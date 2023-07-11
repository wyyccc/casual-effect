from model import *
from custom_layers import *
import os, time 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
import numpy as np

import warnings
warnings.filterwarnings("ignore")


def get_metrics(t, y_f, y_cf, y0, y1):
    sample_num = t.shape[0]
    y0_true = (1-t)*y_f + t*y_cf
    y1_true = t*y_f + (1-t)*y_cf
    eff_true = y1_true - y0_true
    eff_pred = y1 - y0
    eff_diff = eff_true - eff_pred

    ite_mse = tf.reduce_sum(tf.square(eff_diff)).numpy() / sample_num
    ate = tf.abs(tf.reduce_sum(eff_diff)).numpy() / sample_num
    return ite_mse, ate


def calculate_auuc(uplift, treatment, outcome):
    sorted_indices = tf.argsort(uplift)
    sorted_treatment = tf.gather(treatment, sorted_indices)
    sorted_outcome = tf.gather(outcome, sorted_indices)
    uplift_sum = tf.reduce_sum(sorted_treatment * sorted_outcome)
    total_sum = tf.reduce_sum(outcome)
    auuc = uplift_sum - 0.5 * total_sum

    return auuc.numpy()


def train_and_predict(x_train, t_train, yf_train,
                      x_test, t_test, yf_test, ycf_test,
                      CFG,):
    tf.random.set_seed(123)
    np.random.seed(123)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=CFG.model_type+'_'+CFG.rep_mode+'_'+CFG.pred_mode+'_'+CFG.output_mode+'.h5', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min', verbose=CFG.verbose)]
    if CFG.model_type == 'dragonnet':
        model = make_dragonnet(x_train.shape[1], rep_mode=CFG.rep_mode, pred_mode=CFG.pred_mode, output_mode=CFG.output_mode)
    
    elif CFG.model_type == 'DeRCFR':
        model = make_DeRCFR(x_train.shape[1], rep_mode=CFG.rep_mode, pred_mode=CFG.pred_mode, output_mode=CFG.output_mode)
    
    elif CFG.model_type == 'SCI':
        model = make_SCI(x_train.shape[1], rep_mode=CFG.rep_mode, pred_mode=CFG.pred_mode, output_mode=CFG.output_mode)
    
    elif CFG.model_type == 'IPW':
        PS_model, model = make_IPW(x_train.shape[1], rep_mode=CFG.rep_mode, pred_mode=CFG.pred_mode, output_mode=CFG.output_mode)
        PS_model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=CFG.lr), loss='bce')
        PS_model.fit(x_train, t_train, validation_split=0.3, epochs=CFG.PS_batch_size, batch_size=CFG.batch_size, verbose=CFG.verbose)
        for layer in PS_model.layers:
            layer.trainable = False
    
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=CFG.lr), loss=CFG.loss)
    yt_train = np.concatenate([yf_train.reshape(-1, 1), t_train.reshape(-1, 1)], 1)
    model.fit(x_train, yt_train, callbacks=callbacks, validation_split=0.3, epochs=CFG.epoch, batch_size=CFG.batch_size, verbose=CFG.verbose)

    with tf.keras.utils.custom_object_scope({'EpsilonLayer': EpsilonLayer, 'MMOELayer': MMOELayer, 'TransformerEncoderLayer': TransformerEncoderLayer,
                                             'loss_regression': loss_regression, 'loss_cls': loss_cls, 'loss_treg': loss_treg, 'loss_bcauss': loss_bcauss,
                                             'loss_trans_reg': loss_trans_reg, 'loss_trans_cls': loss_trans_cls, 'loss_trans_bcauss': loss_trans_bcauss,
                                             'loss_DeRCFR_cls': loss_DeRCFR_cls, 'loss_DeRCFR_bcauss': loss_DeRCFR_bcauss, 'loss_SCI': loss_SCI,
                                             'loss_IPW': loss_IPW, 'loss_IPW_trans': loss_IPW_trans,}):
        best_model = tf.keras.models.load_model(filepath=CFG.model_type+'_'+CFG.rep_mode+'_'+CFG.pred_mode+'_'+CFG.output_mode+'.h5')
    
    if CFG.model_type in ['dragonnet', 'DeRCFR', 'IPW']:
        yt_hat_test = best_model.predict(x_test)
        y0_pred = yt_hat_test[:, 0]
        y1_pred = yt_hat_test[:, 1]
    elif CFG.model_type in ['SCI']:
        yt_hat_test = best_model.predict(x_test)
        y0_pred = yt_hat_test[:, 2]
        y1_pred = yt_hat_test[:, 3]

    ite_mse, ate = get_metrics(t_test, yf_test, ycf_test, y0_pred, y1_pred)
    # auuc = calculate_auuc(yt_hat_test[:, 1]-yt_hat_test[:, 0], t_test, yf_test)

    return [ite_mse, ate]


def run(CFG, data_base_dir = 'C:/Users/ouyangyan/Desktop/ce/data/ihdp'):
    train_cv = np.load(os.path.join(data_base_dir,'ihdp_npci_1-1000.train.npz'))
    test = np.load(os.path.join(data_base_dir,'ihdp_npci_1-1000.test.npz'))
    
    X_tr    = train_cv.f.x.copy()
    T_tr    = train_cv.f.t.copy()
    YF_tr   = train_cv.f.yf.copy()
    YCF_tr  = train_cv.f.ycf.copy()
    mu_0_tr = train_cv.f.mu0.copy()
    mu_1_tr = train_cv.f.mu1.copy()
    
    X_te    = test.f.x.copy()
    T_te    = test.f.t.copy()
    YF_te   = test.f.yf.copy()
    YCF_te  = test.f.ycf.copy()
    mu_0_te = test.f.mu0.copy()
    mu_1_te = test.f.mu1.copy()

    ite_mse, ate, AUUC = [], [], []
    for idx in range(X_tr.shape[-1]):
        if idx == CFG.cv:
            break
        print(idx)
        X_train, T_train, YF_train, YCF_train = X_tr[:, :, idx], T_tr[:, idx], YF_tr[:, idx], YCF_tr[:, idx]
        X_test, T_test, YF_test, YCF_test = X_te[:, :, idx], T_te[:, idx], YF_te[:, idx], YCF_te[:, idx]
        start = time.time()
        metrics = (train_and_predict(X_train, T_train, YF_train,
                          X_test, T_test, YF_test, YCF_test,
                          CFG))
        ite_mse.append(metrics[0])
        ate.append(metrics[1])
        # AUUC.append(auuc)
        end = time.time()
        print('elaps: %.4f'%(end-start))
    print('ite mse: ', np.mean(ite_mse), np.std(ite_mse))
    print('ate: ', np.mean(ate), np.std(ate))
    # print('AUUC: ', np.mean(AUUC), np.std(AUUC))
    # print(ite_mse)
    # print(ate)


class CFG:
    lr=1e-3
    epoch=500
    PS_batch_size=80
    batch_size=1024
    verbose=0

    cv=100
    model_type='IPW'        # dragonnet / DeRCFR / SCI / IPW
    rep_mode='transformer'           # MLP / transformer
    pred_mode='MLP'               # MLP / PLE
    output_mode='IPW_transformer'            # regression / cls / treg / bcauss / transformer_regression / transformer_cls / transformer_bcauss
    loss=loss_IPW_trans      # loss_regression / loss_cls / loss_treg / loss_bcauss / loss_trans_reg / loss_trans_cls / loss_trans_bcauss / loss_DeRCFR_cls / loss_DeRCFR_bcauss / loss_IPW / loss_IPW_trans


if __name__ == '__main__':
    run(CFG)
    print('model type: ', CFG.model_type)
    print('rep: ', CFG.rep_mode)
    print('pred: ', CFG.pred_mode)
    print('output: ', CFG.output_mode)