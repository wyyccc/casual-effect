# %%
import tensorflow as tf
from tensorflow.keras.layers import Layer
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from keras.models import Model

from custom_layers import *


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


def MMD(source, target, kernels_func=linear_kernel):    # linear_kernel / polynomial_kernel / sigmoid_kernel / guassian_kernel
    kernels = kernels_func(source, target)
    n_s = tf.shape(source)[0]
    n_t = tf.shape(target)[0]
    XX = tf.reduce_sum(kernels[:n_s, :n_s])/(tf.cast(n_s, tf.float32)**2)
    YY = tf.reduce_sum(kernels[-n_t:, -n_t:])/(tf.cast(n_t, tf.float32)**2)
    XY = tf.reduce_sum(kernels[:n_s, -n_t:])/(tf.cast(n_s, tf.float32)*tf.cast(n_t, tf.float32))
    YX = tf.reduce_sum(kernels[-n_t:, :n_s])/(tf.cast(n_s, tf.float32)*tf.cast(n_t, tf.float32))
    loss = XX + YY - XY - YX
    return tf.abs(loss)


class MMDLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MMDLayer, self).__init__(**kwargs)

    def call(self, source, target):
        return MMD(source, target)


def loss_Bcauss(feats_list, ps_list):
    loss, term_list = 0, []
    for i in range(len(feats_list)):
        x, ps = feats_list[i], ps_list[i]
        term_list.append(tf.reduce_sum(x/(ps+0.001), axis=0) / tf.reduce_sum(1/(ps+0.001)))
    for i in range(len(feats_list)-1):
        for j in range(i+1, len(feats_list)):
            loss += tf.reduce_sum(tf.square(term_list[i] - term_list[j]))
    return loss / term_list[0].shape[0]


def wasserstein_distance(x1, x2):
    pairwise_distances = tf.norm(tf.expand_dims(x1, axis=1) - tf.expand_dims(x2, axis=0), axis=-1)
    sorted_distances = tf.sort(pairwise_distances, axis=1)
    cdf_x1 = tf.reduce_mean(tf.cast(sorted_distances <= tf.expand_dims(pairwise_distances, axis=1), tf.float32), axis=1)
    cdf_x2 = tf.reduce_mean(tf.cast(sorted_distances <= tf.expand_dims(pairwise_distances, axis=0), tf.float32), axis=0)
    wasserstein_distance = tf.reduce_mean(tf.abs(cdf_x1 - cdf_x2))
    return wasserstein_distance


def HSIC(x, y):
    n, d = tf.shape(x)[0], tf.shape(x)[1]
    K, Q, H = tf.matmul(x, tf.transpose(x)), tf.matmul(y, tf.transpose(y)), tf.eye(n) - tf.cast(1/n, tf.float32) * tf.ones([n, n])
    hsic = tf.linalg.trace(tf.matmul(tf.matmul(tf.matmul(H, K), H), Q))
    scale = tf.cast(1/((n-1)*(n-1)), dtype=tf.float32)
    return scale * hsic


def KL(t1, t2):
    t1 = tf.nn.softmax(t1)
    t2 = tf.nn.softmax(t2)
    KL = tf.reduce_sum(t1*tf.math.log(t1/t2 + 0.001))
    return KL


def JS(t1, t2):
    t1 = tf.nn.softmax(t1)
    t2 = tf.nn.softmax(t2)
    m = 0.5 * (t1 + t2)
    js_divergence = 0.5 * (tf.reduce_sum(t1*tf.math.log(t1/m + 1e-5)) + tf.reduce_sum(t1*tf.math.log(t1/m + 1e-5)))
    return tf.reduce_mean(js_divergence)


def MI(x, y):
    px = tf.reduce_mean(x)
    py = tf.reduce_mean(y)
    pxy = tf.reduce_mean(tf.multiply(x, y))
    mi = tf.reduce_sum(pxy * tf.math.log(pxy / (px * py + 1e-8)))
    return mi


class ModelTfPrintLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ModelTfPrintLayer, self).__init__(**kwargs)

    def call(self, input, name):
        print_op1 = tf.print("[ModelTfPrintLayer]", input, name)
        with tf.control_dependencies([print_op1]):
            return tf.identity(input)


def make_PS(input_dim,
            num_domains=3,
            reg_l2=0.001,
            act_fn='elu',
            ):
    inputs = Input(shape=(input_dim+2,), name='input')
    input_x, input_t, input_y = inputs[:, :input_dim], inputs[:, input_dim:input_dim+1], inputs[:, input_dim+1:]
    t = tf.cast(input_t, tf.int32)
    t_onehot = tf.squeeze(tf.one_hot(t, depth=num_domains), axis=1)

    x = Dense(units=16, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(input_x)
    t_pred = Dense(units=3, activation='softmax', kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(x)

    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(t_onehot, t_pred))

    model = Model(inputs=inputs, outputs=t_pred)
    model.add_loss(loss)

    return model


def make_BNN(input_dim,
                num_domains=3,
                reg_l2=0.001,
                act_fn='elu',
                use_IPW=False,
                use_IPM=False,
                use_infomax=False,
                ratio_IPM=0.1,
                ratio_infomax=0.1,
                loss_verbose=0,
                ):
    if use_IPW == 'PS':
        inputs = Input(shape=(input_dim+3,), name='input')
        PS = inputs[:, input_dim+2:input_dim+3]
    else:
        inputs = Input(shape=(input_dim+2,), name='input')
    input_x, input_t, input_y = inputs[:, :input_dim], inputs[:, input_dim:input_dim+1], inputs[:, input_dim+1:input_dim+2]
    t = tf.cast(input_t, tf.int32)
    t_onehot = tf.squeeze(tf.one_hot(t, depth=num_domains), axis=1)

    ## representation
    x = Concatenate(1)([input_x, tf.cast(t_onehot, tf.float32)])
    x = Dense(units=128, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(x)
    x = Dense(units=128, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(x)
    x = Dense(units=128, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(x)
    x = BatchNormalization()(x)
    y_pred = Dense(units=1, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(x)

    ## loss
    t_onehot = tf.cast(t_onehot, tf.float32)
    if use_IPW is None:
        weight = tf.reduce_sum(tf.multiply(t_onehot, 1/num_domains/tf.constant([0.2, 0.2, 0.2, 0.2, 0.2])), axis=1, keepdims=True)
    elif use_IPW == 'batch':
        weight = tf.reduce_sum(tf.multiply(t_onehot, 1/num_domains/(tf.reduce_mean(t_onehot, axis=0) + 1e-9)), axis=1, keepdims=True)
    elif use_IPW == 'total':
        weight = tf.reduce_sum(tf.multiply(t_onehot, 1/num_domains/tf.constant([0.5872, 0.167, 0.1566, 0.0796, 0.0096])), axis=1, keepdims=True)
    elif use_IPW == 'PS':
        weight = 1 / ((PS + 1) / 4)
    loss = tf.reduce_mean(weight * tf.square(input_y - y_pred))
    if loss_verbose == 1:
        loss = ModelTfPrintLayer()(loss, 'loss')

    feats_list = []
    for i in range(num_domains):
        index = tf.where(tf.equal(t, i))[:, 0]
        feats_list.append(tf.gather(x, index))
    if use_IPM is not None:
        loss_IPM = 0
        for i in range(num_domains-1):
            for j in range(i+1, num_domains):
                if use_IPM == 'MMD':
                    loss_IPM += MMDLayer()(feats_list[i], feats_list[j])
                elif use_IPM == 'Wdist':
                    loss_IPM += wasserstein_distance(feats_list[i], feats_list[j])
        if use_IPM == 'HSIC':
            loss_IPM = HSIC(x, t_onehot)
        if loss_verbose == 1:
            loss_IPM = ModelTfPrintLayer()(loss_IPM, 'loss_IPM')
        loss += ratio_IPM * loss_IPM
    
    if use_infomax:
        R = x
        n = tf.cast(tf.reduce_sum(t_onehot, axis=0), tf.float32) + tf.constant([1e-9]*num_domains)
        S = tf.reduce_sum(tf.multiply(R, tf.cast(t_onehot[:,0:1], tf.float32)), axis=0) / n[0]
        for i in range(1, num_domains):
            S += tf.reduce_sum(tf.multiply(R, tf.cast(t_onehot[:,i:i+1], tf.float32)), axis=0) / n[i]
        S /= tf.reduce_sum(n/(n+1e-9))
        S = tf.sigmoid(S)
        S = tf.tile(tf.expand_dims(S, axis=0), [tf.shape(R)[0], 1])
        loss_info = JS(R,S)
        if loss_verbose == 1:
            loss_info = ModelTfPrintLayer()(loss_info, 'loss_info')
        loss += ratio_infomax * loss_info

    ## model
    model = Model(inputs=inputs, outputs=y_pred)
    # model = Model(inputs=inputs, outputs=Concatenate(1)([y_pred, y_preds]))
    model.add_loss(loss)

    return model


def make_Tarnet(input_dim,
                num_domains=3,
                reg_l2=0.001,
                act_fn='elu',
                use_IPW=False,
                use_IPM=False,
                use_infomax=False,
                ratio_IPM=0.1,
                ratio_infomax=0.1,
                loss_verbose=0,
                ):
    if use_IPW == 'PS':
        inputs = Input(shape=(input_dim+3,), name='input')
        PS = inputs[:, input_dim+2:input_dim+3]
    else:
        inputs = Input(shape=(input_dim+2,), name='input')
    input_x, input_t, input_y = inputs[:, :input_dim], inputs[:, input_dim:input_dim+1], inputs[:, input_dim+1:input_dim+2]
    t = tf.cast(input_t, tf.int32)
    t_onehot = tf.squeeze(tf.one_hot(t, depth=num_domains), axis=1)

    ## representation
    x = Dense(units=128, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(input_x)
    x = Dense(units=128, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(x)
    x = Dense(units=128, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(x)
    x = BatchNormalization()(x)

    ## predict
    y_preds = []
    for _ in range(num_domains):
        y_pred = Dense(units=128, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(x)
        y_pred = Dense(units=128, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(y_pred)
        y_pred = Dense(units=1, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(y_pred)
        y_preds.append(y_pred)
    y_preds = Concatenate(1)(y_preds)
    y_pred = tf.reduce_sum(tf.multiply(y_preds, t_onehot), axis=1, keepdims=True)

    ## loss
    t_onehot = tf.cast(t_onehot, tf.float32)
    if use_IPW is None:
        weight = tf.reduce_sum(tf.multiply(t_onehot, 1/num_domains/tf.constant([0.2, 0.2, 0.2, 0.2, 0.2])), axis=1, keepdims=True)
    elif use_IPW == 'batch':
        weight = tf.reduce_sum(tf.multiply(t_onehot, 1/num_domains/(tf.reduce_mean(t_onehot, axis=0) + 1e-9)), axis=1, keepdims=True)
    elif use_IPW == 'total':
        weight = tf.reduce_sum(tf.multiply(t_onehot, 1/num_domains/tf.constant([0.5872, 0.167, 0.1566, 0.0796, 0.0096])), axis=1, keepdims=True)
    elif use_IPW == 'PS':
        weight = 1 / ((PS + 1) / 4)
    loss = tf.reduce_mean(weight * tf.square(input_y - y_pred))
    if loss_verbose == 1:
        loss = ModelTfPrintLayer()(loss, 'loss')

    feats_list = []
    for i in range(num_domains):
        index = tf.where(tf.equal(t, i))[:, 0]
        feats_list.append(tf.gather(x, index))

    if use_IPM is not None:
        loss_IPM = 0
        for i in range(num_domains-1):
            for j in range(i+1, num_domains):
                if use_IPM == 'MMD':
                    loss_IPM += MMDLayer()(feats_list[i], feats_list[j])
                elif use_IPM == 'Wdist':
                    loss_IPM += wasserstein_distance(feats_list[i], feats_list[j])
        if use_IPM == 'HSIC':
            loss_IPM = HSIC(x, t_onehot)
        if loss_verbose == 1:
            loss_IPM = ModelTfPrintLayer()(loss_IPM, 'loss_IPM')
        loss += ratio_IPM * loss_IPM
    
    if use_infomax:
        R = x
        n = tf.cast(tf.reduce_sum(t_onehot, axis=0), tf.float32) + tf.constant([1e-9]*num_domains)
        S = tf.reduce_sum(tf.multiply(R, tf.cast(t_onehot[:,0:1], tf.float32)), axis=0) / n[0]
        for i in range(1, num_domains):
            S += tf.reduce_sum(tf.multiply(R, tf.cast(t_onehot[:,i:i+1], tf.float32)), axis=0) / n[i]
        S /= tf.reduce_sum(n/(n+1e-9))
        S = tf.sigmoid(S)
        S = tf.tile(tf.expand_dims(S, axis=0), [tf.shape(R)[0], 1])
        loss_info = JS(R,S)
        if loss_verbose == 1:
            loss_info = ModelTfPrintLayer()(loss_info, 'loss_info')
        loss += ratio_infomax * loss_info

    ## model
    model = Model(inputs=inputs, outputs=Concatenate(1)([y_pred, y_preds[:, 1:2]-y_preds[:, 0:1], y_preds[:, 2:3]-y_preds[:, 0:1], y_preds[:, 3:4]-y_preds[:, 0:1], y_preds[:, 4:5]-y_preds[:, 0:1]]))
    # model = Model(inputs=inputs, outputs=Concatenate(1)([y_pred, y_preds]))
    model.add_loss(loss)

    return model


def make_DRCFR(input_dim,
                num_domains=3,
                reg_l2=0.001,
                act_fn='elu',
                use_IPW=False,
                use_DR=False,
                use_IPM=False,
                use_infomax=False,
                ratio_PS=0.1,
                ratio_DR=1e-5,
                ratio_IPM=0.1,
                ratio_infomax=0.1,
                loss_verbose=0,
                ):
    if use_IPW == 'PS':
        inputs = Input(shape=(input_dim+3,), name='input')
        PS = inputs[:, input_dim+2:input_dim+3]
    else:
        inputs = Input(shape=(input_dim+2,), name='input')
    input_x, input_t, input_y = inputs[:, :input_dim], inputs[:, input_dim:input_dim+1], inputs[:, input_dim+1:input_dim+2]
    t = tf.cast(input_t, tf.int32)
    t_onehot = tf.squeeze(tf.one_hot(t, depth=num_domains), axis=1)

    ## DeR layers
    I = Dense(units=128, activation=act_fn, kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(input_x)
    I = Dense(units=128, activation=act_fn, kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(I)
    I = Dense(units=128, activation=act_fn, kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(I)
    C = Dense(units=128, activation=act_fn, kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(input_x)
    C = Dense(units=128, activation=act_fn, kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(C)
    C = Dense(units=128, activation=act_fn, kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(C)
    P = Dense(units=128, activation=act_fn, kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(input_x)
    P = Dense(units=128, activation=act_fn, kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(P)
    P = Dense(units=128, activation=act_fn, kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(P)

    ## PS layers
    t_pred = Dense(units=num_domains, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(Concatenate(1)([I,C]))

    ## predict
    y_preds = []
    for _ in range(num_domains):
        y_pred = Dense(units=128, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(Concatenate(1)([C,P]))
        y_pred = Dense(units=128, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(y_pred)
        y_pred = Dense(units=1, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(y_pred)
        y_preds.append(y_pred)
    y_preds = Concatenate(1)(y_preds)
    y_pred = tf.reduce_sum(tf.multiply(y_preds, t_onehot), axis=1, keepdims=True)

    ## loss
    t_onehot = tf.cast(t_onehot, tf.float32)
    if use_IPW is None:
        weight = tf.reduce_sum(tf.multiply(t_onehot, 1/num_domains/tf.constant([0.333, 0.333, 0.333])), axis=1, keepdims=True)
    elif use_IPW == 'batch':
        weight = tf.reduce_sum(tf.multiply(t_onehot, 1/num_domains/(tf.reduce_mean(t_onehot, axis=0) + 1e-9)), axis=1, keepdims=True)
    elif use_IPW == 'total':
        weight = tf.reduce_sum(tf.multiply(t_onehot, 1/num_domains/tf.constant([0.568, 0.047, 0.385])), axis=1, keepdims=True)
    elif use_IPW == 'PS':
        weight = 1 / ((PS + 1) / 4)
    loss = tf.reduce_mean(weight * tf.square(input_y - y_pred))
    if loss_verbose == 1:
        loss = ModelTfPrintLayer()(loss, 'loss')
    
    # DR
    if use_DR:
        loss_DR = HSIC(P,I) + HSIC(P,C)+ HSIC(I,C)
        if loss_verbose == 1:
            loss_DR = ModelTfPrintLayer()(loss_DR, 'loss_DR')
        loss += ratio_DR * loss_DR
        loss_PS = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(t_onehot, t_pred))
        if loss_verbose == 1:
            loss_PS = ModelTfPrintLayer()(loss_PS, 'loss_PS')
        loss += ratio_PS * loss_PS

    # IPM
    feats_list = []
    for i in range(num_domains):
        index = tf.where(tf.equal(t, i))[:, 0]
        feats_list.append(tf.gather(P, index))
    if use_IPM is not None:
        loss_IPM = 0
        for i in range(num_domains-1):
            for j in range(i+1, num_domains):
                if use_IPM == 'MMD':
                    loss_IPM += MMDLayer()(feats_list[i], feats_list[j])
                elif use_IPM == 'Wdist':
                    loss_IPM += wasserstein_distance(feats_list[i], feats_list[j])
        if use_IPM == 'HSIC':
            loss_IPM = HSIC(P, t_onehot)
        if loss_verbose == 1:
            loss_IPM = ModelTfPrintLayer()(loss_IPM, 'loss_IPM')
        loss += ratio_IPM * loss_IPM
    
    if use_infomax:
        # loss_info = HSIC(input_x, Concatenate(1)([C,A]))
        R = Concatenate(1)([C,P])
        n = tf.cast(tf.reduce_sum(t_onehot, axis=0), tf.float32) + tf.constant([1e-9]*num_domains)
        S = tf.reduce_sum(tf.multiply(R, tf.cast(t_onehot[:,0:1], tf.float32)), axis=0) / n[0]
        for i in range(1, num_domains):
            S += tf.reduce_sum(tf.multiply(R, tf.cast(t_onehot[:,i:i+1], tf.float32)), axis=0) / n[i]
        S /= tf.reduce_sum(n/(n+1e-9))
        S = tf.sigmoid(S)
        S = tf.tile(tf.expand_dims(S, axis=0), [tf.shape(R)[0], 1])
        
        loss_info = JS(R,S)
        if loss_verbose == 1:
            loss_info = ModelTfPrintLayer()(loss_info, 'loss_info')
        loss += ratio_infomax * loss_info

    ## model
    model = Model(inputs=inputs, outputs=Concatenate(1)([y_pred, y_preds[:, 1:2]-y_preds[:, 0:1], y_preds[:, 2:3]-y_preds[:, 0:1]]))
    model.add_loss(loss)

    return model


def make_IDRL(input_dim,
            num_domains=3,
            reg_l2=0.001,
            act_fn='relu',
            use_PS=False,
            use_DR=False,
            use_MI=False,
            ratio_PS=0.1,
            ratio_DR=0.1,
            ratio_MI=0.1,
            loss_verbose=0,
            ):
    inputs = Input(shape=(input_dim+2,), name='input')
    input_x, input_t, input_y = inputs[:, :input_dim], inputs[:, input_dim:input_dim+1], inputs[:, input_dim+1:]
    t = tf.cast(input_t, tf.int32)
    t_onehot = tf.squeeze(tf.one_hot(t, depth=num_domains), axis=1)

    # feats layers
    R = Dense(units=64, activation=act_fn, kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(input_x)
    H = Dense(units=64, activation=act_fn, kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(input_x)
    
    # PS layers
    t_pred = Dense(units=num_domains, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(H)

    # predict
    y_preds = []
    for _ in range(num_domains):
        y_pred = Dense(units=16, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(R)
        y_pred = Dense(units=4, activation=act_fn, kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(y_pred)
        y_pred = Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal', kernel_regularizer=tf.keras.regularizers.l2(reg_l2))(y_pred)
        y_preds.append(y_pred)
    y_preds = Concatenate(1)(y_preds)
    y_pred = tf.reduce_sum(tf.multiply(y_preds, t_onehot), axis=1, keepdims=True)

    # loss
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(input_y, y_pred))
    if loss_verbose == 1:
        loss = ModelTfPrintLayer()(loss, 'loss')
    
    if use_PS:
        loss_PS = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(t_onehot, t_pred))
        if loss_verbose == 1:
            loss_PS = ModelTfPrintLayer()(loss_PS, 'loss_PS')
        loss += ratio_PS * loss_PS
    
    if use_DR:
        loss_DR = HSIC(R, t_onehot)
        if loss_verbose == 1:
            loss_DR = ModelTfPrintLayer()(loss_DR, 'loss_DR')
        loss += ratio_DR * loss_DR
    
    if use_MI:
        n = tf.cast(tf.reduce_sum(t_onehot, axis=0), tf.float32) + tf.constant([1e-9]*num_domains)
        S = tf.reduce_sum(tf.multiply(R, tf.cast(t_onehot[:,0:1], tf.float32)), axis=0) / n[0]
        for i in range(1, num_domains):
            S += tf.reduce_sum(tf.multiply(R, tf.cast(t_onehot[:,i:i+1], tf.float32)), axis=0) / n[i]
        S /= tf.reduce_sum(n/(n+1e-9))
        S = tf.sigmoid(S)
        S = tf.tile(tf.expand_dims(S, axis=0), [tf.shape(R)[0], 1])
        
        # R = ModelTfPrintLayer()(R, 'R')
        # S = ModelTfPrintLayer()(S, 'S')
        # loss_MI = 100 / MI(R,S)
        loss_MI = JS(R,S)
        if loss_verbose == 1:
            loss_MI = ModelTfPrintLayer()(loss_MI, 'loss_MI')
        loss += ratio_MI * loss_MI
    
    # model
    model = Model(inputs=inputs, outputs=Concatenate(1)([y_pred, y_preds]))
    model.add_loss(loss)

    return model


if __name__ == '__main__':
    n, d = 1000, 128
    # a, b = tf.random.normal((200, d), mean=1.0, stddev=2.0), tf.random.normal((400, d), mean=1.0, stddev=2.0)
    # print(MMDLayer()(a, b), wasserstein_distance(a, b))
    # a, b = tf.random.normal((200, d), mean=2.0, stddev=2.0), tf.random.normal((400, d), mean=1.0, stddev=2.0)
    # print(MMDLayer()(a, b), wasserstein_distance(a, b))
    # a, b = tf.random.normal((200, d), mean=10.0, stddev=2.0), tf.random.normal((400, d), mean=1.0, stddev=2.0)
    # print(MMDLayer()(a, b), wasserstein_distance(a, b))
    # a, b = tf.random.normal((200, d), mean=100.0, stddev=2.0), tf.random.normal((400, d), mean=1.0, stddev=2.0)
    # print(MMDLayer()(a, b), wasserstein_distance(a, b))

    # a = tf.random.normal((n, d), mean=1.0, stddev=2.0)
    # b = tf.random.normal((n, d), mean=1.0, stddev=2.0)
    # print(HSIC(a, b), JS(a, b), KL(a, b))
    # a = tf.random.normal((n, d), mean=1.0, stddev=2.0)
    # b = tf.random.normal((n, d), mean=1.0, stddev=2.0) + a
    # print(HSIC(a, b), JS(a, b), KL(a, b))
    # a = tf.random.normal((n, d), mean=1.0, stddev=2.0)
    # b = tf.random.normal((n, d), mean=0.1, stddev=2.0) + a
    # print(HSIC(a, b), JS(a, b), KL(a, b))

    # a = tf.random.normal((d,), mean=1.0, stddev=2.0)
    # b = tf.random.normal((d,), mean=1.0, stddev=2.0)
    # print(MI(a, b), KL(a, b), JS(a, b))

    # a = tf.random.normal((d,), mean=1.0, stddev=2.0)
    # b = a + tf.random.normal((d,), mean=1.0, stddev=2.0)
    # print(MI(a, b), KL(a, b), JS(a, b))

    a = tf.random.normal((n, d), mean=1.0, stddev=2.0)
    b = tf.random.normal((d,), mean=1.0, stddev=2.0)
    b_tiled = tf.tile(tf.expand_dims(b, axis=0), [tf.shape(a)[0], 1])
    print(JS(a, b_tiled))

    a = tf.random.normal((n, d), mean=100.0, stddev=2.0)
    b = tf.random.normal((d,), mean=1.0, stddev=2.0)
    b_tiled = tf.tile(tf.expand_dims(b, axis=0), [tf.shape(a)[0], 1])
    print(JS(a, b_tiled))

    a = tf.random.normal((n, d), mean=1.0, stddev=2.0)
    b = tf.random.normal((d,), mean=1.0, stddev=2.0)
    b_tiled = tf.tile(tf.expand_dims(b, axis=0), [tf.shape(a)[0], 1])
    a += b_tiled
    print(JS(a, b_tiled))

    a = tf.random.normal((n, d), mean=0.1, stddev=2.0)
    b = tf.random.normal((d,), mean=1.0, stddev=2.0)
    b_tiled = tf.tile(tf.expand_dims(b, axis=0), [tf.shape(a)[0], 1])
    a += b_tiled
    print(JS(a, b_tiled))

    a = tf.constant([[305.,0,0,0], [350.,0,0,0], [340.,0.,0.,0.], [315.,0.,0.,0.]])
    b = tf.constant([[1,0.5,0.999,0.5], [1,0.5,0.999,0.5], [1,0.5,0.999,0.5], [1,0.5,0.999,0.5]])
    print(JS(a, b))

    # model = make_Tarnet(40)


# %%
# t = tf.constant([[0.], [1.], [2.], [0.], [1.], [2.]])
# c_pred = tf.constant([[0.5,0.3,0.2], [0.1,0.7,0.2], [0.2,0.5,0.3], [0.5,0.1,0.4], [0.4,0.4,0.2], [0.3,0.3,0.4]])
# x = tf.constant([[1.,1.,2.,1.], [1.,2.,1.,1.], [2.,2.,2.,1.], [1.,3.,2.,1.], [1.,2.,3.,1.], [1.,2.,2.,2.]])

# feats_list, ps_list = [], []
# for i in range(3):
#     index = tf.where(tf.equal(t, i))[:, 0]
#     feats_list.append(tf.gather(x, index))
#     ps_list.append(tf.gather(c_pred[:, i:i+1], index))

# print(loss_Bcauss(feats_list, ps_list))
# %%
