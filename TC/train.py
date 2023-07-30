from model import *
from custom_layers import *
from sklift.metrics import uplift_auc_score
import os, time 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD, Adam
import numpy as np
from sklearn.metrics import accuracy_score
import random
from tensorflow.keras.callbacks import LambdaCallback


import warnings
warnings.filterwarnings("ignore")


def get_acc(y_true, y_pred):
    y_pred = 1*(y_pred>0.5)
    y_true = y_true
    return accuracy_score(y_true, y_pred)


def get_bce(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))


def calculate_auuc(y_true, y_preds, t_true):
    t_true = tf.squeeze(t_true)
    index_01, index_02, index_12 = tf.where(tf.not_equal(t_true, 2)), tf.where(tf.not_equal(t_true, 1)), tf.where(tf.not_equal(t_true, 0))
    y_01, tau_01, t_01 = tf.squeeze(tf.gather(y_true, index_01)), tf.squeeze(tf.gather(y_preds[:, 1:2] - y_preds[:, 0:1], index_01)), tf.squeeze(tf.gather(t_true, index_01))
    y_02, tau_02, t_02 = tf.squeeze(tf.gather(y_true, index_02)), tf.squeeze(tf.gather(y_preds[:, 2:3] - y_preds[:, 0:1], index_02)), tf.squeeze(tf.gather(t_true, index_02))
    y_12, tau_12, t_12 = tf.squeeze(tf.gather(y_true, index_12)), tf.squeeze(tf.gather(y_preds[:, 2:3] - y_preds[:, 1:2], index_12)), tf.squeeze(tf.gather(t_true, index_12))
    t_02 = tf.where(tf.equal(t_02, 2), 1, t_02)
    t_12 = tf.where(tf.equal(t_12, 1), 0, t_12)
    t_12 = tf.where(tf.equal(t_12, 2), 1, t_12)
    auuc_01, auuc_02, auuc_12 = uplift_auc_score(y_01, tau_01, t_01), uplift_auc_score(y_02, tau_02, t_02), uplift_auc_score(y_12, tau_12, t_12)
    return [auuc_01, auuc_02, auuc_12]


def train_and_predict(x_train, t_train, y_train,
                      x_test, t_test, y_test,
                      CFG, time):
    tf.random.set_seed(time)
    np.random.seed(time)

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='model/'+CFG.model_name+'.h5', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min', verbose=CFG.verbose)]
    
    if CFG.model_type == 'Tarnet':
        model = make_Tarnet(x_train.shape[1], use_IPW=CFG.use_IPW, use_MMD=CFG.use_MMD, use_Wdist=CFG.use_Wdist, use_BCAUSS=CFG.use_BCAUSS, ratio_ce=CFG.ratio_ce, ratio_mmd=CFG.ratio_MMD, ratio_Wdist=CFG.ratio_Wdist, ratio_bcauss=CFG.ratio_BCAUSS)
    elif CFG.model_type == 'DRCFR':
        model = make_DRCFR(x_train.shape[1], use_IPW=CFG.use_IPW, use_MIM=CFG.use_MIM, use_MMD=CFG.use_MMD, use_Wdist=CFG.use_Wdist, use_BCAUSS=CFG.use_BCAUSS, ratio_ce=CFG.ratio_ce, ratio_MIM=CFG.ratio_MIM, ratio_mmd=CFG.ratio_MMD, ratio_Wdist=CFG.ratio_Wdist, ratio_bcauss=CFG.ratio_BCAUSS)

    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=CFG.lr))
    # model.summary()

    xty_train = np.concatenate([x_train, t_train, y_train], 1)
    xty_test = np.concatenate([x_test, t_test, y_test], 1)

    #print(model.predict(xty_train[:10]))
    #assert 1==2
    model.fit(xty_train, y_train, callbacks=callbacks, validation_split=0.3, epochs=CFG.epoch, batch_size=CFG.batch_size, verbose=CFG.verbose)
    #print(model.predict(xty_train[:10]))

    with tf.keras.utils.custom_object_scope({'EpsilonLayer': EpsilonLayer, 'MMOELayer': MMOELayer, 'TransformerEncoderLayer': TransformerEncoderLayer,
                                             'MMDLayer': MMDLayer,}):
        best_model = tf.keras.models.load_model(filepath='model/'+CFG.model_name+'.h5')
    
    if CFG.insample:
        preds = best_model.predict(xty_train)
        y_pred, y_preds = preds[:, 0:1], preds[:, 1:]
        acc_insample = get_acc(y_train, y_pred)
        bce_insample = get_bce(y_train, y_pred)
        auuc_insample = calculate_auuc(y_train, y_preds, t_train)
    preds = best_model.predict(xty_test)
    y_pred, y_preds = preds[:, 0:1], preds[:, 1:]
    acc = get_acc(y_test, y_pred)
    bce = get_bce(y_test, y_pred)
    auuc = calculate_auuc(y_test, y_preds, t_test)

    if CFG.insample:
        return acc, acc_insample, bce, bce_insample, auuc, auuc_insample
    return acc, bce, auuc


def run(CFG, data_base_dir = 'C:/Users/ouyangyan/Desktop/TC/data/data.csv'):
    data = np.genfromtxt(data_base_dir, delimiter=',')[1:]
    data = data[~np.isnan(data).any(axis=1)].astype(float)
    data[:, -1] = 1*(data[:, -1]>10)
    train_cv = data[:int(len(data)*0.8), :]
    test = data[int(len(data)*0.8):, :]

    np.random.seed(111)
    random.seed(111)
    change_count = int(len(data) * 0.1)
    change_indexes = random.sample(range(len(data)), change_count)
    for index in change_indexes:
        if data[index, -1] == 0:
            data[index, -1] = 1
        else:
            data[index, -1] = 0
    
    X_tr = train_cv[:, :40]
    T_tr = train_cv[:, 40:41].astype(int)
    Y_tr = train_cv[:, 41:]
    
    X_te  = test[:, :40]
    T_te  = test[:, 40:41].astype(int)
    Y_te  = test[:, 41:]

    acc, acc_insample, bce, bce_insample, auuc_01, auuc_02, auuc_12, auuc_01_insample, auuc_02_insample, auuc_12_insample = [], [], [], [], [], [], [], [], [], []
    for t in range(CFG.times):
        print(t)
        start = time.time()
        metrics = train_and_predict(X_tr, T_tr, Y_tr, X_te, T_te, Y_te, CFG, t)
        if CFG.insample:
            acc.append(metrics[0])
            acc_insample.append(metrics[1])
            bce.append(metrics[2])
            bce_insample.append(metrics[3])
            auuc_01.append(metrics[4][0])
            auuc_02.append(metrics[4][1])
            auuc_12.append(metrics[4][2])
            auuc_01_insample.append(metrics[5][0])
            auuc_02_insample.append(metrics[5][1])
            auuc_12_insample.append(metrics[5][2])
        end = time.time()
        print('elaps: %.4f'%(end-start))
    print('acc: ', np.mean(acc), np.std(acc))
    print('acc_insample: ', np.mean(acc_insample), np.std(acc_insample))
    print('bce: ', np.mean(bce), np.std(bce))
    print('bce_insample: ', np.mean(bce_insample), np.std(bce_insample))
    print('AUUC_01: ', np.mean(auuc_01), np.std(auuc_01))
    print('AUUC_02: ', np.mean(auuc_02), np.std(auuc_02))
    print('AUUC_12: ', np.mean(auuc_12), np.std(auuc_12))
    print('AUUC_01_insample: ', np.mean(auuc_01_insample), np.std(auuc_01_insample))
    print('AUUC_02_insample: ', np.mean(auuc_02_insample), np.std(auuc_02_insample))
    print('AUUC_12_insample: ', np.mean(auuc_12_insample), np.std(auuc_12_insample))


class CFG:
    lr=1e-3
    epoch=200
    batch_size=1024
    verbose=1

    times=1
    insample=True

    model_type='DRCFR'        # Tarnet / DRCFR
    model_name= model_type + ''

    use_MIM=True
    ratio_MIM=1e-9

    use_IPW=False
    ratio_ce=1e-1

    use_MMD=True
    ratio_MMD=1e-3

    use_Wdist=False
    ratio_Wdist=1e-2

    use_BCAUSS=False
    ratio_BCAUSS=1e-1

    # kernels_func=guassian_kernel              # linear_kernel(1e-3) / polynomial_kernel(1e-5) / sigmoid_kernel(1e-1) / guassian_kernel(1e-1)
    

if __name__ == '__main__':
    run(CFG)
    print('model type: ', CFG.model_type)