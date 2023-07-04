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


def train_and_predict(x_train, t_train, yf_train,
                      x_test, t_test, yf_test, ycf_test,
                      CFG,):
    tf.random.set_seed(123)
    np.random.seed(123)

    model = make_model(x_train.shape[1], rep_mode=CFG.rep_mode, pred_mode=CFG.pred_mode, output_mode=CFG.output_mode)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=CFG.rep_mode+' '+CFG.pred_mode+' '+CFG.output_mode+'.h5', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min', verbose=CFG.verbose)]
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=CFG.lr), loss=CFG.loss)
    yt_train = np.concatenate([yf_train.reshape(-1, 1), t_train.reshape(-1, 1)], 1)
    history = model.fit(x_train, yt_train,
                    callbacks=callbacks,
                    validation_split=0.3,
                    epochs=CFG.epoch,
                    batch_size=CFG.batch_size, 
                    verbose=CFG.verbose)

    with tf.keras.utils.custom_object_scope({'EpsilonLayer': EpsilonLayer, 'MMOELayer': MMOELayer, 'TransformerEncoderLayer': TransformerEncoderLayer,
                                             'loss_regression': loss_regression, 'loss_cls': loss_cls, 'loss_treg': loss_treg, 'loss_bcauss': loss_bcauss,
                                             'loss_trans_reg': loss_trans_reg, 'loss_trans_cls': loss_trans_cls, 'loss_trans_bcauss': loss_trans_bcauss,}):
        best_model = tf.keras.models.load_model(CFG.rep_mode+' '+CFG.pred_mode+' '+CFG.output_mode+'.h5')
    yt_hat_test = best_model.predict(x_test)

    ite_mse, ate = get_metrics(t_test, yf_test, ycf_test, yt_hat_test[:, 0], yt_hat_test[:, 1])

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

    # i = 1
    # print(X_tr[i,:,0])
    # print(T_tr[i,0])
    # print(YF_tr[i,0])
    # print(YCF_tr[i,0])
    # print(mu_0_tr[i,0])
    # print(mu_1_tr[i,0])

    ite_mse = []
    ate = []
    for idx in range(X_tr.shape[-1]):
        print(idx)
        if idx == CFG.cv:
            break
        X_train, T_train, YF_train, YCF_train = X_tr[:, :, idx], T_tr[:, idx], YF_tr[:, idx], YCF_tr[:, idx]
        X_test, T_test, YF_test, YCF_test = X_te[:, :, idx], T_te[:, idx], YF_te[:, idx], YCF_te[:, idx]
        start = time.time()
        metrics = (train_and_predict(X_train, T_train, YF_train,
                          X_test, T_test, YF_test, YCF_test,
                          CFG))
        ite_mse.append(metrics[0])
        ate.append(metrics[1])
        end = time.time()
        print('elaps: %.4f'%(end-start))
    print('ite mse: ', np.mean(ite_mse), np.std(ite_mse))
    print('ate: ', np.mean(ate), np.std(ate))
    # print(ite_mse)
    # print(ate)


class CFG:
    lr=1e-3
    epoch=300
    batch_size=1024
    verbose=0

    cv=100
    rep_mode='MLP'           # MLP / transformer
    pred_mode='MMOE'               # MLP / MMOE
    output_mode='bcauss'             # regression / cls / treg / bcauss / transformer_regression / transformer_cls / transformer_bcauss
    loss=loss_bcauss          # loss_regression / loss_cls / loss_treg / loss_bcauss / loss_trans_reg / loss_trans_cls / loss_trans_bcauss


if __name__ == '__main__':
    run(CFG)
    print('rep: ', CFG.rep_mode)
    print('pred: ', CFG.pred_mode)
    print('output: ', CFG.output_mode)