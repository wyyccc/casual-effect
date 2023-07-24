import tensorflow as tf
import keras.backend as K
from tensorflow.keras.layers import Layer
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from keras.models import Model
from keras import regularizers
import tensorflow_probability as tfp

from custom_layers import *


def cal_corr(x, y):
    mean1, mean2 = tf.reduce_mean(x), tf.reduce_mean(y)
    deviation1, deviation2 = tf.subtract(x, mean1), tf.subtract(y, mean2)
    product = tf.reduce_sum(tf.multiply(deviation1, deviation2))
    std_deviation1, std_deviation2 = tf.sqrt(tf.reduce_sum(tf.square(deviation1))), tf.sqrt(tf.reduce_sum(tf.square(deviation2)))
    return tf.divide(product, tf.multiply(std_deviation1, std_deviation2))


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


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_s=tf.shape(source)[0]
    n_s=64 if n_s is None else n_s
    n_t= tf.shape(target)[0]
    n_t=64 if n_t is None else n_t
    n_samples =n_s+n_t
    total = tf.concat([source, target], axis=0)
    total0 = tf.expand_dims(total,axis=0)
    total1 = tf.expand_dims(total,axis=1)
    L2_distance = tf.reduce_sum(((total0 - total1) ** 2),axis=2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = tf.reduce_sum(L2_distance) / float(n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [tf.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def MMD(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    n_s=tf.shape(source)[0]
    n_s=64 if n_s is None else n_s
    n_t=tf.shape(target)[0]
    n_t=64 if n_t is None else n_t
    XX = tf.reduce_sum(kernels[:n_s, :n_s])/float(n_s**2)
    YY = tf.reduce_sum(kernels[-n_t:, -n_t:])/float(n_t**2)
    XY = tf.reduce_sum(kernels[:n_s, -n_t:])/float(n_s*n_t)
    YX = tf.reduce_sum(kernels[-n_t:, :n_s])/float(n_s*n_t)
    loss = XX + YY - XY - YX
    return loss


def loss_regression(label, concat_pred, ratio_bce=1):
    y_true, t_true = label[:, 0:1], label[:, 1:2]
    y0_predictions, y1_predictions = concat_pred[:, 0:1], concat_pred[:, 1:2]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions))
    loss_reg = loss0 + loss1

    return loss_reg


def loss_cls(label, concat_pred, ratio_bce=1):
    y_true, t_true = label[:, 0:1], label[:, 1:2]
    y0_predictions, y1_predictions, t_predictions = concat_pred[:, 0:1], concat_pred[:, 1:2], concat_pred[:, 2:3]
    t_pred = (t_predictions + 0.001) / 1.002

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions))
    loss_reg = loss0 + loss1

    loss_bce = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    return loss_reg + ratio_bce * loss_bce


def loss_treg(label, concat_pred, ratio_bce=1, ratio_epsilon=1):
    y_true, t_true = label[:, 0:1], label[:, 1:2]
    y0_predictions, y1_predictions, t_predictions, epsilons = concat_pred[:, 0:1], concat_pred[:, 1:2], concat_pred[:, 2:3], concat_pred[:, 3:4]
    t_pred = (t_predictions + 0.001) / 1.002

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions))
    loss_reg = loss0 + loss1

    loss_bce = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    y_pred = t_true * y1_predictions + (1 - t_true) * y0_predictions
    h = t_true / t_pred - (1 - t_true) / (1 - t_pred)
    y_pert = y_pred + epsilons * h
    loss_treg = tf.reduce_sum(tf.square(y_true - y_pert))

    return loss_reg + ratio_bce * loss_bce + ratio_epsilon * loss_treg


def loss_bcauss(label, concat_pred, ratio_bcauss=10):
    input_dim = concat_pred.shape[1]-3
    y_true, t_true = label[:, 0:1], label[:, 1:2]
    y0_predictions, y1_predictions, t_predictions, inputs = concat_pred[:, 0:1], concat_pred[:, 1:2], concat_pred[:, 2:3], concat_pred[:, 3:]
    t_pred = (t_predictions + 0.001) / 1.002
        
    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions))
    loss_reg = loss0 + loss1    
    
    ones_to_sum = tf.tile(t_true / t_pred, [1, input_dim]) * inputs
    zeros_to_sum = tf.tile((1 - t_true) / (1 - t_pred), [1, input_dim]) * inputs
    ones_mean = tf.math.reduce_sum(ones_to_sum, 0) / tf.math.reduce_sum(t_true / t_pred, 0)
    zeros_mean = tf.math.reduce_sum(zeros_to_sum, 0) / tf.math.reduce_sum((1 - t_true) / (1 - t_pred), 0)
    loss_bcauss = tf.keras.losses.mean_squared_error(zeros_mean, ones_mean)

    return loss_reg + ratio_bcauss * loss_bcauss


def loss_trans_reg(label, concat_pred, ratio_recon=1):
    y_true, t_true = label[:, 0:1], label[:, 1:2]
    y0_predictions, y1_predictions, recon_term = concat_pred[:, 0:1], concat_pred[:, 1:2], concat_pred[:, 2:3]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions))
    loss_reg = loss0 + loss1

    loss_recon = tf.reduce_sum(recon_term)

    return loss_reg + ratio_recon * loss_recon


def loss_trans_cls(label, concat_pred, ratio_bce=1, ratio_recon=1):
    y_true, t_true = label[:, 0:1], label[:, 1:2]
    y0_predictions, y1_predictions, t_predictions, recon_term = concat_pred[:, 0:1], concat_pred[:, 1:2], concat_pred[:, 2:3], concat_pred[:, 3:]
    t_pred = (t_predictions + 0.001) / 1.002

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions))
    loss_reg = loss0 + loss1

    loss_bce = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    loss_recon = tf.reduce_sum(recon_term)

    return loss_reg + ratio_bce * loss_bce + ratio_recon * loss_recon


def loss_trans_bcauss(label, concat_pred, ratio_recon=1, ratio_bcauss=1):
    input_dim = concat_pred.shape[1]-4
    y_true, t_true = label[:, 0:1], label[:, 1:2]
    y0_predictions, y1_predictions, t_predictions, recon_term, inputs = concat_pred[:, 0:1], concat_pred[:, 1:2], concat_pred[:, 2:3], concat_pred[:, 3:4], concat_pred[:, 4:]
    t_pred = (t_predictions + 0.001) / 1.002
        
    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions))
    loss_reg = loss0 + loss1

    loss_recon = tf.reduce_sum(recon_term)
    
    ones_to_sum = tf.tile(t_true / t_pred, [1, input_dim]) * inputs
    zeros_to_sum = tf.tile((1 - t_true) / (1 - t_pred), [1, input_dim]) * inputs
    ones_mean = tf.math.reduce_sum(ones_to_sum, 0) / tf.math.reduce_sum(t_true / t_pred, 0)
    zeros_mean = tf.math.reduce_sum(zeros_to_sum, 0) / tf.math.reduce_sum((1 - t_true) / (1 - t_pred), 0)
    loss_bcauss = tf.keras.losses.mean_squared_error(zeros_mean, ones_mean)

    return loss_reg + ratio_recon * loss_recon + ratio_bcauss * loss_bcauss


def loss_DeRCFR_cls(label, concat_pred, hidden_dim=100):
    y_true, t_true = label[:, 0:1], label[:, 1:2]
    y0_predictions, y1_predictions, t_predictions, A, I = concat_pred[:, 0:1], concat_pred[:, 1:2], concat_pred[:, 2:3], concat_pred[:, 3:hidden_dim+3], concat_pred[:, hidden_dim+3:2*hidden_dim+3]
    t_pred = (t_predictions + 0.001) / 1.002

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions))
    loss_reg = loss0 + loss1

    loss_bce = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    # loss_corr = 0
    # for i in range(200):
    #     loss_corr += tf.square(cal_corr(I[:, i:i+1], y_true)) + tf.square(cal_corr(A[:, i:i+1], t_true))
    loss_corr = mutual_information(I, y_true) + mutual_information(A, t_true)

    return loss_reg + loss_bce


def loss_DeRCFR_bcauss(label, concat_pred, hidden_dim=100):
    input_dim = concat_pred.shape[1]-(2*hidden_dim+4)
    y_true, t_true = label[:, 0:1], label[:, 1:2]
    y0_predictions, y1_predictions, t_predictions, A, I, c_predictions, inputs = concat_pred[:, 0:1], concat_pred[:, 1:2], concat_pred[:, 2:3], concat_pred[:, 3:hidden_dim+3], concat_pred[:, hidden_dim+3:2*hidden_dim+3], concat_pred[:, 2*hidden_dim+3:2*hidden_dim+4], concat_pred[:, 2*hidden_dim+4:]
    t_pred, c_pred = (t_predictions + 0.001) / 1.002, (c_predictions + 0.001) / 1.002

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions))
    loss_reg = loss0 + loss1

    loss_bce = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    loss_corr = mutual_information(I, y_true) + mutual_information(A, t_true)
    
    ones_to_sum = tf.tile(t_true / c_pred, [1, input_dim]) * inputs
    zeros_to_sum = tf.tile((1 - t_true) / (1 - c_pred), [1, input_dim]) * inputs
    ones_mean = tf.math.reduce_sum(ones_to_sum, 0) / tf.math.reduce_sum(t_true / c_pred, 0)
    zeros_mean = tf.math.reduce_sum(zeros_to_sum, 0) / tf.math.reduce_sum((1 - t_true) / (1 - c_pred), 0)
    loss_bcauss = tf.keras.losses.mean_squared_error(zeros_mean, ones_mean)

    return loss_reg + loss_bce + loss_bcauss


def loss_SCI(label, concat_pred, ratio_recon=0, ratio_depen=0): 
    hidden_dim = (concat_pred.shape[1]-6) // 2
    y_true, t_true = label[:, 0:1], label[:, 1:2]
    y0_pred, y1_pred, y0_predictions, y1_predictions, recon_loss0, recon_loss1, x0, x1 = concat_pred[:, 0:1], concat_pred[:, 1:2], concat_pred[:, 2:3], concat_pred[:, 3:4], concat_pred[:, 4:5], concat_pred[:, 5:6], concat_pred[:, 6:6+hidden_dim], concat_pred[:, 6+hidden_dim:]

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions)) + tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions)) + tf.reduce_sum(t_true * tf.square(y_true - y1_pred))
    loss_reg = loss0 + loss1

    loss_recon = tf.reduce_sum((1-t_true)*recon_loss0 + t_true*recon_loss1)

    loss_depen = HSIC(x0, t_true) + HSIC(x1, t_true)

    return loss_reg + ratio_recon * loss_recon + ratio_depen * loss_depen


def loss_IPW(label, concat_pred, threshold=1):
    y_true, t_true = label[:, 0:1], label[:, 1:2]
    y0_predictions, y1_predictions, t_predictions = concat_pred[:, 0:1], concat_pred[:, 1:2], concat_pred[:, 2:3]
    t_pred = (t_predictions + 0.001) / 1.002

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions) / (1-t_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions) / t_pred)
    loss_reg = loss0 + loss1

    return loss_reg


def loss_IPW_trans(label, concat_pred, ratio_recon=0):
    y_true, t_true = label[:, 0:1], label[:, 1:2]
    y0_predictions, y1_predictions, t_predictions, recon_term = concat_pred[:, 0:1], concat_pred[:, 1:2], concat_pred[:, 2:3], concat_pred[:, 3:4]
    t_pred = (t_predictions + 0.001) / 1.002

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions) * 1/(1-t_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions) / t_pred)
    loss_reg = loss0 + loss1

    loss_recon = tf.reduce_sum(recon_term)

    return loss_reg + ratio_recon * loss_recon


def loss_memento(label, concat_pred, ratio_bce=1.0, ratio_mmd=1.0):
    y_true, t_true = label[:, 0:1], label[:, 1:2]
    y0_predictions, y1_predictions, t_predictions, x = concat_pred[:, 0:1], concat_pred[:, 1:2], concat_pred[:, 2:3], concat_pred[:, 3:]
    t_pred = (t_predictions + 0.001) / 1.002

    loss0 = tf.reduce_sum((1. - t_true) * tf.square(y_true - y0_predictions) / (1-t_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_predictions) / t_pred)
    loss_reg = loss0 + loss1

    loss_bce = tf.reduce_sum(K.binary_crossentropy(t_true, t_pred))

    index_0, index_1 = tf.where(tf.equal(t_true, 0)), tf.where(tf.equal(t_true, 1))
    x0, x1 = tf.gather(x, index_0), tf.gather(x, index_1)
    loss_MMD = MMD(x0, x1)

    return loss_reg + ratio_bce * loss_bce + ratio_mmd * loss_MMD


def make_Dragonnet(input_dim,
                   num_domains=2,
                   reg_l2=0.001,
                   act_fn='relu',
                   rep_mode='MLP',
                   pred_mode='MLP',
                   output_mode='regression',
                   num_experts=4,
                   expert_dim=128,
                   transformer_heads=4,
                   transformer_dim=16,
                   ):

    inputs = Input(shape=(input_dim,), name='input')

    ## representation layers
    if rep_mode == 'MLP':
        x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
        x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(x)
        x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(x)
        if pred_mode != 'PLE':
            t_predictions = Dense(units=1, activation='sigmoid')(x)
    if rep_mode == 'transformer':
        expanded_input = tf.expand_dims(inputs, axis=2)
        expanded_input = tf.tile(expanded_input, [1, 1, transformer_dim])
        transformer_output = TransformerEncoderLayer(d_model=transformer_dim, num_heads=transformer_heads, dff=transformer_dim*2)(expanded_input)
        x = tf.keras.layers.Flatten()(transformer_output)
        t_predictions = Dense(units=1, activation='sigmoid')(x)
        recon_inputs = Dense(units=1, activation=None)(transformer_output)
        recon_inputs = tf.squeeze(recon_inputs, axis = -1)
        diff = tf.square(inputs - recon_inputs)
        sum_recon_diff = tf.reduce_sum(diff, axis=1, keepdims=True)


    ## predict layers
    if pred_mode == 'MLP':
        y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
        y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
        y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
        y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)
        y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
        y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)
    elif pred_mode == 'PLE':
        mmoe_outs = MMOELayer(num_domains+1, num_experts, expert_dim, gate_dropout=0.1)(x)
        expert0 = Dense(units=expert_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
        expert1 = Dense(units=expert_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
        y0_predictions = Dense(units=expert_dim, activation=None, kernel_regularizer=regularizers.l2(reg_l2))(Concatenate(1)([mmoe_outs[0], expert0]))
        y1_predictions = Dense(units=expert_dim, activation=None, kernel_regularizer=regularizers.l2(reg_l2))(Concatenate(1)([mmoe_outs[1], expert1]))
        y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2))(y0_predictions)
        y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2))(y1_predictions)
        t_predictions = Dense(units=1, activation='sigmoid')(mmoe_outs[2])

    ## output
    if output_mode == 'regression':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions]))
    elif output_mode == 'cls':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions]))
    elif output_mode == 'treg':
        dl = EpsilonLayer()
        epsilons = dl(t_predictions, name='epsilon') 
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, epsilons]))
    elif output_mode == 'bcauss':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, inputs]))
    elif output_mode == 'transformer_regression':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, sum_recon_diff]))
    elif output_mode == 'transformer_cls':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, sum_recon_diff]))
    elif output_mode == 'transformer_bcauss':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, sum_recon_diff, inputs]))

    return model


def make_DeRCFR(input_dim,
                num_domains=2,
                hidden_dim=100,
                reg_l2=0.001,
                act_fn='relu',
                rep_mode = 'MLP',
                pred_mode = 'MLP',
                output_mode = 'cls',
                transformer_heads=4,
                transformer_dim=16,
                num_experts=4,
                expert_dim=128):
    
    inputs = Input(shape=(input_dim,), name='input')

    ## representation layers
    if rep_mode == 'MLP':
        x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
        x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(x)
        A = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(x)
        x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
        x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(x)
        C = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(x)
        x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
        x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(x)
        I = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(x)
        t_predictions = Dense(units=1, activation='sigmoid')(Concatenate(1)([C, I]))

    if rep_mode == 'transformer':
        expanded_input = tf.expand_dims(inputs, axis=2)
        expanded_input = tf.tile(expanded_input, [1, 1, transformer_dim])
        transformer_output = TransformerEncoderLayer(d_model=transformer_dim, num_heads=transformer_heads, dff=transformer_dim*2)(expanded_input)
        A = tf.keras.layers.Flatten()(transformer_output)
        recon_inputs = Dense(units=1, activation=None)(transformer_output)
        recon_inputs = tf.squeeze(recon_inputs, axis = -1)
        diff = tf.square(inputs - recon_inputs)
        sum_recon_diff = tf.reduce_sum(diff, axis=1, keepdims=True)
        transformer_output = TransformerEncoderLayer(d_model=transformer_dim, num_heads=transformer_heads, dff=transformer_dim*2)(expanded_input)
        C = tf.keras.layers.Flatten()(transformer_output)
        recon_inputs = Dense(units=1, activation=None)(transformer_output)
        recon_inputs = tf.squeeze(recon_inputs, axis = -1)
        diff = tf.square(inputs - recon_inputs)
        sum_recon_diff += tf.reduce_sum(diff, axis=1, keepdims=True)
        transformer_output = TransformerEncoderLayer(d_model=transformer_dim, num_heads=transformer_heads, dff=transformer_dim*2)(expanded_input)
        I = tf.keras.layers.Flatten()(transformer_output)
        recon_inputs = Dense(units=1, activation=None)(transformer_output)
        recon_inputs = tf.squeeze(recon_inputs, axis = -1)
        diff = tf.square(inputs - recon_inputs)
        sum_recon_diff += tf.reduce_sum(diff, axis=1, keepdims=True)
        t_predictions = Dense(units=1, activation='sigmoid')(Concatenate(1)([I, C]))
    
    
    ## predict layers
    if pred_mode == 'MLP':
        x = Concatenate(1)([A, C])
        y0_hidden = Dense(units=hidden_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
        y1_hidden = Dense(units=hidden_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
        y0_hidden = Dense(units=hidden_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
        y1_hidden = Dense(units=hidden_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)
        y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
        y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)
        if output_mode == 'bcauss':
            c_predictions = Dense(units=1, activation='sigmoid')(C)
    elif pred_mode == 'PLE':
        x = Concatenate(1)([A, C])
        mmoe_outs = MMOELayer(num_domains, num_experts, expert_dim, gate_dropout=0.1)(x)
        expert0 = Dense(units=expert_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
        expert1 = Dense(units=expert_dim, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
        y0_predictions = Dense(units=expert_dim, activation=None, kernel_regularizer=regularizers.l2(reg_l2))(Concatenate(1)([mmoe_outs[0], expert0]))
        y1_predictions = Dense(units=expert_dim, activation=None, kernel_regularizer=regularizers.l2(reg_l2))(Concatenate(1)([mmoe_outs[1], expert1]))
        y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2))(y0_predictions)
        y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2))(y1_predictions)
        if output_mode == 'bcauss':
            c_predictions = Dense(units=1, activation='sigmoid')(C)
    
    ## output
    if output_mode == 'regression':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions]))
    elif output_mode == 'cls':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, A, I]))
    elif output_mode == 'bcauss':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, A, I, c_predictions, inputs]))
    elif output_mode == 'transformer_regression':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, sum_recon_diff]))
    elif output_mode == 'transformer_cls':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, sum_recon_diff]))
    elif output_mode == 'transformer_bcauss':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, sum_recon_diff, inputs]))

    return model
    

def make_SCI(input_dim,
             num_domains=2,
             hidden_dim=100,
             reg_l2=0.001,
             act_fn='relu',
             rep_mode = 'MLP',
             pred_mode = 'MLP',
             output_mode = 'regression'):
    inputs = Input(shape=(input_dim,), name='input')

    ## representation layers
    if rep_mode == 'MLP':
        x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
        x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(x)
        z0 = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(x)
        y0_pred = Dense(units=1, activation=None, kernel_initializer='RandomNormal')(z0)
        x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
        x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(x)
        z1 = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(x)
        y1_pred = Dense(units=1, activation=None, kernel_initializer='RandomNormal')(z1)
        x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
        x = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(x)
        zc = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(x)

    if rep_mode == 'transformer':
        1
    
    
    ## predict layers
    if pred_mode == 'MLP':
        x0 = Concatenate(1)([z0, zc])
        x1 = Concatenate(1)([z1, zc])
        x0_recon = Dense(units=input_dim, activation=None, kernel_regularizer=regularizers.l2(reg_l2))(x0)
        x1_recon = Dense(units=input_dim, activation=None, kernel_regularizer=regularizers.l2(reg_l2))(x1)
        recon_loss0 = tf.reduce_sum(tf.square(inputs - x0_recon), axis=1, keepdims=True)
        recon_loss1 = tf.reduce_sum(tf.square(inputs - x1_recon), axis=1, keepdims=True)
        y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2))(x0)
        y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2))(x1)
    elif pred_mode == 'PLE':
        1
    
    ## output
    if output_mode == 'regression':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_pred, y1_pred, y0_predictions, y1_predictions, recon_loss0, recon_loss1, x0, x1]))
    # elif output_mode == 'bcauss':
    #     model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, A, I, c_predictions, inputs]))
    # elif output_mode == 'transformer_regression':
    #     model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, sum_recon_diff]))
    # elif output_mode == 'transformer_cls':
    #     model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, sum_recon_diff]))
    # elif output_mode == 'transformer_bcauss':
    #     model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, sum_recon_diff, inputs]))

    return model


def make_BWCFR(input_dim,
             num_domains=2,
             hidden_dim=100,
             reg_l2=0.001,
             act_fn='relu',
             rep_mode = 'MLP',
             pred_mode = 'MLP',
             output_mode = 'regression',
             transformer_heads=4,
             transformer_dim=16,):
    inputs = Input(shape=(input_dim,), name='input')

    ## representation layers
    if rep_mode == 'MLP':
        x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
        x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(x)
        x = Dense(units=200, activation=act_fn, kernel_initializer='RandomNormal')(x)

    elif rep_mode == 'transformer':
        expanded_input = tf.expand_dims(inputs, axis=2)
        expanded_input = tf.tile(expanded_input, [1, 1, transformer_dim])
        transformer_output = TransformerEncoderLayer(d_model=transformer_dim, num_heads=transformer_heads, dff=transformer_dim*2)(expanded_input)
        x = tf.keras.layers.Flatten()(transformer_output)
        recon_inputs = Dense(units=1, activation=None)(transformer_output)
        recon_inputs = tf.squeeze(recon_inputs, axis = -1)
        diff = tf.square(inputs - recon_inputs)
        sum_recon_diff = tf.reduce_sum(diff, axis=1, keepdims=True)
    
    x_PS = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
    x_PS = Dense(units=hidden_dim, activation=act_fn, kernel_initializer='RandomNormal')(x_PS)
    t_pred = Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal')(x_PS)
    PS_model = Model(inputs=inputs, outputs=t_pred)


    ## predict layers
    if pred_mode == 'MLP':
        y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
        y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
        y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
        y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)
        y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
        y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)
    elif pred_mode == 'PLE':
        1
    
    ## output
    if output_mode == 'cls':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_pred]))
    elif output_mode == 'transformer_cls':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_pred, sum_recon_diff]))
    # elif output_mode == 'bcauss':
    #     model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, A, I, c_predictions, inputs]))
    # elif output_mode == 'transformer_regression':
    #     model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, sum_recon_diff]))
    # elif output_mode == 'transformer_cls':
    #     model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, sum_recon_diff]))
    # elif output_mode == 'transformer_bcauss':
    #     model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, sum_recon_diff, inputs]))

    return PS_model, model


def make_TransTEE(input_dim,
                   num_domains=2,
                   reg_l2=0.001,
                   act_fn='relu',
                   output_mode='regression',
                   transformer_heads=4,
                   transformer_dim=16,
                   ):

    inputs = Input(shape=(input_dim,), name='inputs')

    ## x representation
    x = tf.expand_dims(inputs, axis=2)
    x = tf.tile(x, [1, 1, transformer_dim])
    x = Dense(units=transformer_dim, activation=act_fn)(x)
    x = TransformerEncoderLayer(d_model=transformer_dim, num_heads=transformer_heads, dff=transformer_dim*2)(x)
    x = Dense(units=1, activation=act_fn)(x)
    x = tf.keras.layers.Flatten()(x)

    ## t representation
    t_predictions = Dense(units=1, activation='sigmoid')(x)

    ## predict layers
    # y0_embed = Dense(units=input_dim, activation=act_fn, name='y0_embed')(tf.constant(np.array([1, 0])))
    # y1_embed = Dense(units=input_dim, activation=act_fn, name='y1_embed')(tf.constant(np.array([0, 1])))

    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(x)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(x)


    ## output
    if output_mode == 'regression':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions]))
    elif output_mode == 'cls':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions]))
    
    return model


def make_Memento(input_dim,
                reg_l2=0.001,
                act_fn='relu',
                output_mode='regression',
                ):
    
    inputs = Input(shape=(input_dim,), name='inputs')

    ## x representation
    x = Dense(units=25, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=25, activation=act_fn, kernel_initializer='RandomNormal')(x)
    x = Dense(units=25, activation=act_fn, kernel_initializer='RandomNormal')(x)
    t_predictions = Dense(units=1, activation='sigmoid')(x)


    ## predict layers
    x = Dense(units=100, activation=act_fn, kernel_initializer='RandomNormal')(inputs)
    x = Dense(units=100, activation=act_fn, kernel_initializer='RandomNormal')(x)
    x = Dense(units=100, activation=act_fn, kernel_initializer='RandomNormal')(x)

    y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
    y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(x)
    y0_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y0_hidden)
    y1_hidden = Dense(units=100, activation=act_fn, kernel_regularizer=regularizers.l2(reg_l2))(y1_hidden)
    y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(y0_hidden)
    y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(y1_hidden)

    ## output
    if output_mode == 'regression':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions]))
    elif output_mode == 'memento':
        model = Model(inputs=inputs, outputs=Concatenate(1)([y0_predictions, y1_predictions, t_predictions, x]))
    
    return model


if __name__ == '__main__':
    d = 10
    sample1 = tf.random.normal([100, d], mean=1.0, stddev=3.0)
    sample2 = tf.random.normal([200, d], mean=1.0, stddev=2.0)
    mmd_value = MMD(sample1, sample2)
    print("MMD:", mmd_value)