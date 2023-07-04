import tensorflow as tf
import keras.backend as K
from tensorflow.keras.layers import Layer
from keras.layers import Input, Dense, Concatenate, BatchNormalization, Dropout
from keras.models import Model
from keras import regularizers

from custom_layers import *


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


def loss_bcauss(label, concat_pred, ratio_bcauss=1):
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
    input_dim = concat_pred.shape[1]-3
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


def make_model(input_dim, 
                 reg_l2=0.01,
                 act_fn='relu',
                 rep_mode='MLP',
                 pred_mode='MLP',
                 output_mode='dragonnet',
                 num_domains=2,
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
        if pred_mode != 'MMOE':
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
    elif pred_mode == 'MMOE':
        mmoe_outs = MMOELayer(num_domains+1, num_experts, expert_dim, gate_dropout=0.1)(x)
        y0_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y0_predictions')(mmoe_outs[0])
        y1_predictions = Dense(units=1, activation=None, kernel_regularizer=regularizers.l2(reg_l2), name='y1_predictions')(mmoe_outs[1])
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