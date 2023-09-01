import tensorflow as tf
import keras.backend as K
from tensorflow.keras.layers import Layer
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from keras.models import Model
from keras import regularizers
import tensorflow_probability as tfp


def linear_kernel(source, target):
    return tf.matmul(source, tf.transpose(target))


def polynomial_kernel(source, target, degree=2, coef0=1.0):
    return tf.pow((tf.matmul(source, tf.transpose(target)) + coef0), degree)


def sigmoid_kernel(source, target, coef0=1.0):
    return tf.tanh((tf.matmul(source, tf.transpose(target)) + coef0))


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_s = tf.shape(source)[0]
    n_t = tf.shape(target)[0]
    n_samples = n_s + n_t
    total = tf.concat([source, target], axis=0)
    total0 = tf.expand_dims(total,axis=0)
    total1 = tf.expand_dims(total,axis=1)
    L2_distance = tf.reduce_sum(((total0 - total1) ** 2),axis=2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / tf.cast(n_samples ** 2 - n_samples, tf.float32)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def MMD(source, target, kernels_func=guassian_kernel):    # linear_kernel / polynomial_kernel / sigmoid_kernel / guassian_kernel
    kernels = kernels_func(source, target)
    n_s = tf.shape(source)[0]
    n_t = tf.shape(target)[0]
    XX = tf.reduce_sum(kernels[:n_s, :n_s])/(tf.cast(n_s, tf.float32)**2)
    YY = tf.reduce_sum(kernels[-n_t:, -n_t:])/(tf.cast(n_t, tf.float32)**2)
    XY = tf.reduce_sum(kernels[:n_s, -n_t:])/(tf.cast(n_s, tf.float32)*tf.cast(n_t, tf.float32))
    YX = tf.reduce_sum(kernels[-n_t:, :n_s])/(tf.cast(n_s, tf.float32)*tf.cast(n_t, tf.float32))
    loss = XX + YY - XY - YX
    return tf.abs(loss)


class MMDLayer(Layer):
    def __init__(self, **kwargs):
        super(MMDLayer, self).__init__(**kwargs)

    def call(self, source, target):
        return MMD(source, target)


def wasserstein_distance(x1, x2):
    pairwise_distances = tf.norm(tf.expand_dims(x1, axis=1) - tf.expand_dims(x2, axis=0), axis=-1)
    sorted_distances = tf.sort(pairwise_distances, axis=1)
    cdf_x1 = tf.reduce_mean(tf.cast(sorted_distances <= tf.expand_dims(pairwise_distances, axis=1), tf.float32), axis=1)
    cdf_x2 = tf.reduce_mean(tf.cast(sorted_distances <= tf.expand_dims(pairwise_distances, axis=0), tf.float32), axis=0)
    wasserstein_distance = tf.reduce_mean(tf.abs(cdf_x1 - cdf_x2))
    return wasserstein_distance


def mutual_information(x, y):
    x = tf.nn.softmax(x)
    y = tf.nn.softmax(y)
    px = tf.reduce_mean(x, axis=0)
    py = tf.reduce_mean(y, axis=0)
    mi = tf.reduce_sum(x * tf.math.log(x / px + 1e-8), axis=1) + tf.reduce_sum(y * tf.math.log(y / py + 1e-8), axis=1)
    mi = tf.reduce_mean(mi)
    return mi


def HSIC(x, y):
    shape = tf.shape(x)
    n = shape[0]
    K, Q, H = tf.matmul(x, tf.transpose(x)), tf.matmul(y, tf.transpose(y)), tf.eye(n) - tf.cast(1/n, tf.float32) * tf.ones([n, n])
    hsic = tf.linalg.trace(tf.matmul(tf.matmul(tf.matmul(H, K), H), Q))
    scale = tf.cast(1/((n-1)*(n-1)), dtype=tf.float32)
    return scale * hsic


class ModelTfPrintLayer(Layer):
    def __init__(self, **kwargs):
        super(ModelTfPrintLayer, self).__init__(**kwargs)

    def call(self, input, name):
        print_op1 = tf.print("[ModelTfPrintLayer]", input, name)
        with tf.control_dependencies([print_op1]):
            return tf.identity(input)


def make_PS(input_dim,
            reg_l2=0.001,
            act_fn='elu',
            ):

    inputs = Input(shape=(input_dim+2,), name='input')
    input_x, input_t, input_y = inputs[:, :input_dim], inputs[:, input_dim:input_dim+1], inputs[:, input_dim+1:]

    t_pred = Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(input_x)

    model = Model(inputs=inputs, outputs=t_pred)
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(input_t, t_pred))
    model.add_loss(loss)
    return model


def make_BNN(input_dim,
            hidden_dim=128,
            reg_l2=0.001,
            act_fn='elu',
            ):

    inputs = Input(shape=(input_dim+2,), name='input')
    input_x, input_y = inputs[:, :input_dim+1], inputs[:, input_dim+1:]

    x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(input_x)
    x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(x)
    y_pred = Dense(units=1, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(x)

    ## output
    model = Model(inputs=inputs, outputs=y_pred)

    ## loss
    loss = tf.reduce_mean(tf.square(y_pred - input_y))
    model.add_loss(loss)
    return model


def make_Tarnet(input_dim,
                num_domains=2,
                hidden_dim=128,
                reg_l2=0.001,
                act_fn='elu',
                use_IPW=None,
                use_IPM=None,
                ratio_IPM=1,
                loss_verbose=False,
                ):

    if use_IPW == 'PS':
        inputs = Input(shape=(input_dim+3,), name='input')
        PS = inputs[:, input_dim+2:input_dim+3]
    else:
        inputs = Input(shape=(input_dim+2,), name='input')
    input_x, input_t, input_y = inputs[:, :input_dim], inputs[:, input_dim:input_dim+1], inputs[:, input_dim+1:]
    t = tf.cast(input_t, tf.int32)
    t_onehot = tf.squeeze(tf.one_hot(t, depth=num_domains), axis=1)

    ## representation layers
    x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(input_x)
    x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(x)
    x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(x)

    ## predict layers
    y0_hidden = Dense(units=hidden_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
    y0_hidden = Dense(units=hidden_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
    y1_hidden = Dense(units=hidden_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=hidden_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)
    y_preds = Concatenate(1)([y0_predictions, y1_predictions])
    y_pred = tf.reduce_sum(tf.multiply(y_preds, t_onehot), axis=1, keepdims=True)

    ## output
    model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions]))

    ## loss
    if use_IPW == 'PS':
        t = tf.cast(input_t, tf.float32)
        w = t/(2*PS+0.01) + (1-t)/(2*(1-PS)+0.01)
        loss = tf.reduce_mean(tf.multiply(w, tf.square(y_pred - input_y)))
    elif use_IPW == 'weighted':
        u = tf.cast(tf.reduce_sum(t) / tf.shape(t)[0], tf.float32)
        t = tf.cast(t, tf.float32)
        w = t/(2*u+0.01) + (1-t)/(2*(1-u)+0.01)
        loss = tf.reduce_mean(tf.multiply(w, tf.square(y_pred - input_y)))
    else:
        loss = tf.reduce_mean(tf.square(y_pred - input_y))
    if loss_verbose:
        loss = ModelTfPrintLayer()(loss, 'loss')

    if use_IPM is not None:
        index0, index1 = tf.where(tf.equal(t, 0))[:, 0], tf.where(tf.equal(t, 1))[:, 0]
        feats0, feats1 = tf.gather(x, index0), tf.gather(x, index1)
        if use_IPM == 'MMD':
            loss_IPM = MMDLayer()(feats0, feats1)
        elif use_IPM == 'Wdist':
            loss_IPM = wasserstein_distance(feats0, feats1)
        elif use_IPM == 'HSIC':
            loss_IPM = HSIC(x, t_onehot)
        if loss_verbose:
            loss_IPM = ModelTfPrintLayer()(loss_IPM, 'loss_IPM')
        loss += ratio_IPM * loss_IPM
        
    model.add_loss(loss)
    return model


def make_DR(input_dim,
            num_domains=2,
            hidden_dim=128,
            reg_l2=0.001,
            act_fn='elu',
            use_IPW=None,
            use_IPM=None,
            use_DR=False,
            use_PS=False,
            ratio_IPM=1,
            ratio_DR=1,
            ratio_PS=1,
            loss_verbose=False,
            ):
    
    if use_IPW == 'PS':
        inputs = Input(shape=(input_dim+3,), name='input')
        PS = inputs[:, input_dim+2:input_dim+3]
    else:
        inputs = Input(shape=(input_dim+2,), name='input')
    input_x, input_t, input_y = inputs[:, :input_dim], inputs[:, input_dim:input_dim+1], inputs[:, input_dim+1:]
    t = tf.cast(input_t, tf.int32)
    t_onehot = tf.squeeze(tf.one_hot(t, depth=num_domains), axis=1)

    ## representation layers
    x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(input_x)
    x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(x)
    A = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(x)
    x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(input_x)
    x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(x)
    C = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(x)
    x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(input_x)
    x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(x)
    I = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=regularizers.l2(reg_l2))(x)

    ## PS
    t_pred = Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal')(Concatenate(1)([C, I]))
    
    ## predict layers
    x = Concatenate(1)([A, C])
    y0_hidden = Dense(units=hidden_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
    y0_hidden = Dense(units=hidden_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
    y1_hidden = Dense(units=hidden_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=hidden_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)
    y_preds = Concatenate(1)([y0_predictions, y1_predictions])
    t_onehot = tf.squeeze(tf.one_hot(t, depth=num_domains), axis=1)
    y_pred = tf.reduce_sum(tf.multiply(y_preds, t_onehot), axis=1, keepdims=True)
    
    ## output
    model = Model(inputs=inputs, outputs=y_preds)

    ## loss
    if use_IPW == 'PS':
        t = tf.cast(input_t, tf.float32)
        w = t/(2*PS+0.01) + (1-t)/(2*(1-PS)+0.01)
        loss = tf.reduce_mean(tf.multiply(w, tf.square(y_pred - input_y)))
    elif use_IPW == 'weighted':
        u = tf.cast(tf.reduce_sum(t) / tf.shape(t)[0], tf.float32)
        t = tf.cast(t, tf.float32)
        w = t/(2*u+0.01) + (1-t)/(2*(1-u)+0.01)
        loss = tf.reduce_mean(tf.multiply(w, tf.square(y_pred - input_y)))
    else:
        loss = tf.reduce_mean(tf.square(y_pred - input_y))
    if loss_verbose:
        loss = ModelTfPrintLayer()(loss, 'loss')

    if use_IPM is not None:
        index0, index1 = tf.where(tf.equal(t, 0))[:, 0], tf.where(tf.equal(t, 1))[:, 0]
        feats0, feats1 = tf.gather(A, index0), tf.gather(A, index1)
        if use_IPM == 'MMD':
            loss_IPM = MMDLayer()(feats0, feats1)
        elif use_IPM == 'Wdist':
            loss_IPM = wasserstein_distance(feats0, feats1)
        elif use_IPM == 'HSIC':
            loss_IPM = HSIC(A, t_onehot)
        if loss_verbose:
            loss_IPM = ModelTfPrintLayer()(loss_IPM, 'loss_IPM')
        loss += ratio_IPM * loss_IPM

    if use_DR:
        y = tf.cast(input_y, tf.int32)
        y_onehot = tf.cast(tf.squeeze(tf.one_hot(y, depth=num_domains), axis=1), tf.float32)
        loss_DR = HSIC(A, C) + HSIC(A, I) + HSIC(I, C) + HSIC(I, y_onehot)
        if loss_verbose:
            loss_DR = ModelTfPrintLayer()(loss_DR, 'loss_DR')
        loss += ratio_DR * loss_DR

    if use_PS:
        t = tf.cast(t, tf.int32)
        loss_PS = tf.reduce_mean(tf.keras.losses.binary_crossentropy(t, t_pred))
        if loss_verbose:
            loss_PS = ModelTfPrintLayer()(loss_PS, 'loss_PS')
        loss += ratio_PS * loss_PS

    model.add_loss(loss)
    return model


if __name__ == '__main__':
    d = 10