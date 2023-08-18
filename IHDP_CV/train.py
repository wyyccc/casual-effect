from model import *
import os, time 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
import numpy as np
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


def get_metrics(t, y_f, y_cf, y0, y1, mu_0, mu_1):
    sample_num = t.shape[0]
    y0_true = (1-t)*y_f + t*y_cf
    y1_true = t*y_f + (1-t)*y_cf
    eff_true = y1_true - y0_true
    eff_pred = y1 - y0
    eff_diff = eff_true - eff_pred

    pehe = tf.sqrt(tf.reduce_sum(tf.square(eff_diff)).numpy() / sample_num).numpy()
    ate = tf.abs(tf.reduce_sum(mu_1 - mu_0 - eff_pred)).numpy() / sample_num

    return pehe, ate


def train_and_predict(x_train, t_train, yf_train, ycf_train, mu_0_train, mu_1_train,
                      x_test, t_test, yf_test, ycf_test, mu_0_test, mu_1_test,
                      CFG,):
    tf.random.set_seed(123)
    np.random.seed(123)

    y_unscaled = np.concatenate([yf_train.reshape(-1, 1), yf_test.reshape(-1, 1)], axis=0)
    y_scaler = StandardScaler().fit(y_unscaled)

    yf_train = y_scaler.transform(yf_train.reshape(-1, 1))

    # print(np.concatenate([t_train.reshape(-1, 1), yf_train.reshape(-1, 1), ycf_train.reshape(-1, 1)], 1)[:20])
    # assert 1==2

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='model/'+CFG.model_type+CFG.model_name+'.h5', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min', verbose=CFG.verbose)]
    
    if CFG.model_type == 'BNN':
        model = make_BNN(x_train.shape[1])
    elif CFG.model_type == 'Tarnet':
        model = make_Tarnet(x_train.shape[1], use_IPM=CFG.use_IPM, ratio_IPM=CFG.ratio_IPM, loss_verbose=CFG.loss_verbose)
    elif CFG.model_type == 'DeRCFR':
        model = make_DR(x_train.shape[1])
    
    model.compile(optimizer=tf.keras.optimizers.Nadam(tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=CFG.lr, decay_steps=1, decay_rate=0.999)))
    # model.summary()

    xty_train = np.concatenate([x_train, t_train.reshape(-1, 1), yf_train.reshape(-1, 1)], 1)
    xty_test = np.concatenate([x_test, t_test.reshape(-1, 1), yf_test.reshape(-1, 1)], 1)
    model.fit(xty_train, yf_train, callbacks=callbacks, validation_split=0.2, epochs=CFG.epoch, batch_size=CFG.batch_size, verbose=CFG.verbose)

    with tf.keras.utils.custom_object_scope({'ModelTfPrintLayer': ModelTfPrintLayer, 'MMDLayer': MMDLayer,}):
        best_model = tf.keras.models.load_model(filepath='model/'+CFG.model_type+CFG.model_name+'.h5')
    if CFG.model_type == 'BNN':
        xty_test_0 = np.concatenate([x_test, np.zeros((1,1)), np.zeros((1,1))], 1)
        xty_test_1 = np.concatenate([x_test, np.ones((1,1)), np.zeros((1,1))], 1)
        yt_hat_test = np.concatenate([best_model.predict(xty_test_0, verbose=CFG.verbose).reshape(-1, 1), best_model.predict(xty_test_1, verbose=CFG.verbose).reshape(-1, 1)], 1)
    else:
        yt_hat_test = best_model.predict(xty_test, verbose=CFG.verbose)

    y0_pred = y_scaler.inverse_transform(yt_hat_test[:, 0:1]).reshape(-1)
    y1_pred = y_scaler.inverse_transform(yt_hat_test[:, 1:2]).reshape(-1)
    yf_train = y_scaler.inverse_transform(yf_train).reshape(-1)
    
    if CFG.insample:
        if CFG.model_type == 'BNN':
            xty_test_0 = np.concatenate([x_test, np.zeros((1,1)), np.zeros((1,1))], 1)
            xty_test_1 = np.concatenate([x_test, np.ones((1,1)), np.zeros((1,1))], 1)
            yt_hat_test = np.concatenate([best_model.predict(xty_test_0, verbose=CFG.verbose).reshape(-1, 1), best_model.predict(xty_test_1, verbose=CFG.verbose).reshape(-1, 1)], 1)
        else:
            yt_hat_train = best_model.predict(xty_train)
            y0_pred_train = y_scaler.inverse_transform(yt_hat_train[:, 0:1]).reshape(-1)
            y1_pred_train = y_scaler.inverse_transform(yt_hat_train[:, 1:2]).reshape(-1)
        # pehe_insample, ate_insample = get_metrics(t_train, yf_train, ycf_train, y0_pred_train, y1_pred_train, mu_0_train, mu_1_train)
        pehe_insample, ate_insample = 1, 1

    pehe, ate = get_metrics(t_test, yf_test, ycf_test, y0_pred, y1_pred, mu_0_test, mu_1_test)
    # pehe, ate = get_metrics(t_test, mu_0_test, mu_1_test, y0_pred, y1_pred)

    if CFG.insample:
        return [pehe, ate], [pehe_insample, ate_insample]
    return [pehe, ate]


def run(CFG):
    data = np.genfromtxt(CFG.data_dir, delimiter=',')
    data = data[~np.isnan(data).any(axis=1)].astype(float)
    
    T, YF, YCF, mu_0, mu_1, X = data[:, 0:1].astype(int), data[:, 1:2], data[:, 2:3], data[:, 3:4], data[:, 4:5], data[:, 5:]

    pehe, ate = [], []
    pehe_insample, ate_insample = [], []
    for i in range(CFG.cv):
        if i > 747:
            break
        print(i)
        X_train, T_train, YF_train, YCF_train, mu_0_train, mu_1_train = np.delete(X,i,axis=0), np.delete(T,i,axis=0), np.delete(YF,i,axis=0), np.delete(YCF,i,axis=0), np.delete(mu_0,i,axis=0), np.delete(mu_1,i,axis=0)
        X_test, T_test, YF_test, YCF_test, mu_0_test, mu_1_test = X[i:i+1], T[i:i+1], YF[i:i+1], YCF[i:i+1], mu_0[i:i+1], mu_1[i:i+1]
        start = time.time()
        if CFG.insample:
            metrics, metrics_insample = (train_and_predict(X_train, T_train, YF_train, YCF_train, mu_0_train, mu_1_train,
                          X_test, T_test, YF_test, YCF_test, mu_0_test, mu_1_test,
                          CFG))
            pehe.append(metrics[0])
            ate.append(metrics[1])
            pehe_insample.append(metrics_insample[0])
            ate_insample.append(metrics_insample[1])
        else:
            metrics = (train_and_predict(X_train, T_train, YF_train, YCF_train, mu_0_train, mu_1_train,
                          X_test, T_test, YF_test, YCF_test, mu_0_test, mu_1_test,
                          CFG))
            pehe.append(metrics[0])
            ate.append(metrics[1])
        end = time.time()
        print('elaps: %.4f'%(end-start))
    print('pehe: ', np.mean(pehe), np.std(pehe))
    print('ate: ', np.mean(ate), np.std(ate))

    print('pehe (insample): ', np.mean(pehe_insample), np.std(pehe_insample))
    print('ate (insample): ', np.mean(ate_insample), np.std(ate_insample))


class CFG:
    data_dir = 'C:/Users/ouyangyan/Desktop/CE1/data/data.csv'
    np.random.seed(123)
    cv=747
    insample=False

    lr=1e-2
    epoch=200
    batch_size=1024
    verbose=0
    loss_verbose=False    

    model_type='Tarnet'         # 'BNN' / 'Tarnet' / 'DR'
    model_name='_2'
    use_IPW=None
    use_IPM=None            # None / 'MMD' / 'Wdist' / 'HSIC'
    ratio_IPM=1


if __name__ == '__main__':
    run(CFG)
    print('model type: ', CFG.model_type)