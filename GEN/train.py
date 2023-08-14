from model import *
from custom_layers import *
import time 
import tensorflow as tf
import keras
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

# import logging
# logger = logging.getLogger()

import warnings
warnings.filterwarnings("ignore")


def get_metrics(ce_true, ce_preds):
    mean1, mean2, mean3, mean4 = tf.reduce_mean(ce_true[:,0]), tf.reduce_mean(ce_true[:,1]), tf.reduce_mean(ce_true[:,2]), tf.reduce_mean(ce_true[:,3])
    rmse1, rmse2, rmse3, rmse4 = tf.sqrt(tf.reduce_mean(tf.square(ce_true[:,0] - ce_preds[:,0]))), tf.sqrt(tf.reduce_mean(tf.square(ce_true[:,1] - ce_preds[:,1]))), tf.sqrt(tf.reduce_mean(tf.square(ce_true[:,2] - ce_preds[:,2]))), tf.sqrt(tf.reduce_mean(tf.square(ce_true[:,3] - ce_preds[:,3])))
    nrmse1, nrmse2, nrmse3, nrmse4 = rmse1 / mean1, rmse2 / mean2, rmse3 / mean3, rmse4 / mean4
    return [nrmse1, nrmse2, nrmse3, nrmse4]


def train_and_predict(x_train, t_train, y_train, ce_train,
                      x_test, ce_test,
                      CFG):
    tf.random.set_seed(CFG.seed)
    np.random.seed(CFG.seed)

    xty_train = np.concatenate([x_train, t_train, y_train], 1)
    t_dummy, y_dummy = np.ones((x_test.shape[0],1)), np.ones((x_test.shape[0],1))
    xty_test = np.concatenate([x_test, t_dummy, y_dummy], 1)

    if CFG.use_IPW == 'PS':
        PS_model = make_PS(x_train.shape[1], num_domains=CFG.num_domains)
        PS_model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=CFG.lr))
        #PS_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=CFG.lr, momentum=CFG.momentum, nesterov=True))
        callbacks_PS = [tf.keras.callbacks.ModelCheckpoint(filepath='model/'+CFG.model_name+'_PS'+'.h5', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min', verbose=CFG.verbose)]
        PS_model.fit(xty_train, y_train, callbacks=callbacks_PS, validation_split=0.3, epochs=CFG.PS_epoch, batch_size=CFG.batch_size, verbose=CFG.verbose)
        with tf.keras.utils.custom_object_scope({'MMDLayer': MMDLayer, 'ModelTfPrintLayer': ModelTfPrintLayer}):
            best_model = tf.keras.models.load_model(filepath='model/'+CFG.model_name+'_PS'+'.h5')
        PS = tf.reduce_sum(best_model.predict(xty_train) * tf.squeeze(tf.one_hot(t_train, depth=3), axis=1), axis=1, keepdims=True)
        xty_train = np.concatenate([xty_train, PS], 1)
        xty_test = np.concatenate([xty_test, np.ones((x_test.shape[0],1))], 1)

    if CFG.model_type == 'BNN':
        model = make_BNN(x_train.shape[1], num_domains=CFG.num_domains, use_IPW=CFG.use_IPW, use_IPM=CFG.use_IPM, use_infomax=CFG.use_infomax
                            , ratio_IPM=CFG.ratio_IPM, ratio_infomax=CFG.ratio_infomax
                            , loss_verbose=CFG.loss_verbose)
    elif CFG.model_type == 'Tarnet':
        model = make_Tarnet(x_train.shape[1], num_domains=CFG.num_domains, use_IPW=CFG.use_IPW, use_IPM=CFG.use_IPM, use_infomax=CFG.use_infomax
                            , ratio_IPM=CFG.ratio_IPM, ratio_infomax=CFG.ratio_infomax
                            , loss_verbose=CFG.loss_verbose)
    elif CFG.model_type == 'DRCFR':
        model = make_DRCFR(x_train.shape[1], num_domains=CFG.num_domains, use_IPW=CFG.use_IPW, use_DR=CFG.use_DR, use_IPM=CFG.use_IPM, use_infomax=CFG.use_infomax
                           , ratio_PS=CFG.ratio_PS, ratio_DR=CFG.ratio_DR, ratio_IPM=CFG.ratio_IPM, ratio_infomax=CFG.ratio_infomax
                           , loss_verbose=CFG.loss_verbose)
    elif CFG.model_type == 'IDRL':
        model = make_IDRL(x_train.shape[1], num_domains=CFG.num_domains, use_MI=CFG.use_MI, use_DR=CFG.use_DR, use_PS=CFG.use_PS, ratio_MI=CFG.ratio_MI, ratio_DR=CFG.ratio_DR, ratio_PS=CFG.ratio_PS
                         , loss_verbose=CFG.loss_verbose)

    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=CFG.lr))
    # model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=CFG.lr, momentum=CFG.momentum, nesterov=True))
    # model.summary()
    
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='model/'+CFG.model_name+'.h5', monitor='val_loss', save_best_only=True, save_weights_only=False, mode='min', verbose=CFG.verbose)
                 , keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=CFG.verbose, mode='auto', min_delta=1e-8, cooldown=10, min_lr=1e-5)]
    model.fit(xty_train, y_train, callbacks=callbacks, validation_split=0.3, epochs=CFG.epoch, batch_size=CFG.batch_size, verbose=CFG.verbose)
    # print(model.predict(xty_train[:10]))
    # assert 1==2

    with tf.keras.utils.custom_object_scope({'MMDLayer': MMDLayer, 'ModelTfPrintLayer': ModelTfPrintLayer}):
        best_model = tf.keras.models.load_model(filepath='model/'+CFG.model_name+'.h5')
    
    if CFG.insample:
        if CFG.model_type == 'BNN':
            y0 = best_model.predict(np.concatenate([x_train, 0*np.ones((x_train.shape[0],1)), np.ones((x_train.shape[0],1))], axis=1), batch_size=CFG.batch_size)
            y1 = best_model.predict(np.concatenate([x_train, 1*np.ones((x_train.shape[0],1)), np.ones((x_train.shape[0],1))], axis=1), batch_size=CFG.batch_size)
            y2 = best_model.predict(np.concatenate([x_train, 2*np.ones((x_train.shape[0],1)), np.ones((x_train.shape[0],1))], axis=1), batch_size=CFG.batch_size)
            y3 = best_model.predict(np.concatenate([x_train, 3*np.ones((x_train.shape[0],1)), np.ones((x_train.shape[0],1))], axis=1), batch_size=CFG.batch_size)
            y4 = best_model.predict(np.concatenate([x_train, 4*np.ones((x_train.shape[0],1)), np.ones((x_train.shape[0],1))], axis=1), batch_size=CFG.batch_size)
            preds = np.concatenate([y0.reshape(-1,1), y1.reshape(-1,1)-y0.reshape(-1,1), y2.reshape(-1,1)-y0.reshape(-1,1), y3.reshape(-1,1)-y0.reshape(-1,1), y4.reshape(-1,1)-y0.reshape(-1,1)], axis=1)
        else:
            preds = best_model.predict(xty_train, batch_size=CFG.batch_size)
        y_pred, ce_preds = preds[:, 0:1], preds[:, 1:]
        metrics_insample = get_metrics(ce_train, ce_preds)
    if CFG.model_type == 'BNN':
        y0 = best_model.predict(np.concatenate([x_test, 0*np.ones((x_test.shape[0],1))], axis=1), batch_size=CFG.batch_size)
        y1 = best_model.predict(np.concatenate([x_test, 1*np.ones((x_test.shape[0],1))], axis=1), batch_size=CFG.batch_size)
        y2 = best_model.predict(np.concatenate([x_test, 2*np.ones((x_test.shape[0],1))], axis=1), batch_size=CFG.batch_size)
        y3 = best_model.predict(np.concatenate([x_test, 3*np.ones((x_test.shape[0],1))], axis=1), batch_size=CFG.batch_size)
        y4 = best_model.predict(np.concatenate([x_test, 4*np.ones((x_test.shape[0],1))], axis=1), batch_size=CFG.batch_size)
        preds = np.concatenate([y0.reshape(-1,1), y1.reshape(-1,1)-y0.reshape(-1,1), y2.reshape(-1,1)-y0.reshape(-1,1), y3.reshape(-1,1)-y0.reshape(-1,1), y4.reshape(-1,1)-y0.reshape(-1,1)], axis=1)
    else:
        preds = best_model.predict(xty_test, batch_size=CFG.batch_size)
    y_pred, ce_preds = preds[:, 0:1], preds[:, 1:]
    metrics = get_metrics(ce_test, ce_preds)

    if CFG.insample:
        return metrics_insample, metrics
    return metrics


def run(CFG, train_dir='C:/Users/ouyangyan/Desktop/multi/data/train.csv', test_dir='C:/Users/ouyangyan/Desktop/multi/data/test.csv', target_dir='C:/Users/ouyangyan/Desktop/multi/data/target.csv'):
    # train = np.genfromtxt(train_dir, delimiter=',')[1:]
    # test = np.genfromtxt(test_dir, delimiter=',')[1:]
    # target = np.genfromtxt(target_dir, delimiter=',')[1:]

    # train = np.concatenate([train, target[:-5000]], 1)
    # test = np.concatenate([test, target[-5000:]], 1)

    # train = train[~np.isnan(train).any(axis=1)].astype(float)
    # test = test[~np.isnan(test).any(axis=1)].astype(float)

    
    # X_tr = train[:, :-4]
    # T_tr = train[:, -4:-3].astype(int)
    # Y_tr = train[:, -3:-2]
    
    # X_te  = test[:, :-2]

    # X_scaler = StandardScaler().fit(np.concatenate([X_tr, X_te], axis=0))
    # X_tr, X_te = X_scaler.transform(X_tr), X_scaler.transform(X_te)

    # CE_tr = train[:, -2:]
    # CE_te = test[:, -2:]

    np.random.seed(CFG.seed)
    mean = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cov = np.array([[1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
                    [0.2, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
                    [0.2, 0.2, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
                    [0.2, 0.2, 0.2, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 
                    [0.2, 0.2, 0.2, 0.2, 1, 0.2, 0.2, 0.2, 0.2, 0.2], 
                    [0.2, 0.2, 0.2, 0.2, 0.2, 1, 0.2, 0.2, 0.2, 0.2], 
                    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1, 0.2, 0.2, 0.2], 
                    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1, 0.2, 0.2], 
                    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1, 0.2], 
                    [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 1]])
    X = np.random.multivariate_normal(mean, cov, 5000)

    T = X[:, 0:1] + 0.5*X[:, 2:3] + 0.1*(X[:, 4:5]**2) + 0.2*(X[:, 6:7]**2) + 0.2*X[:, 7:8] + 0.1*X[:, 9:10] + np.random.normal(0, 4, 5000).reshape(-1, 1)
    T[T < 1] = 0
    T[(T > 1) & (T <= 3)] = 1
    T[(T > 3) & (T <= 6)] = 2
    T[(T > 6) & (T <= 10)] = 3
    T[T > 10] = 4
    T = T.astype(int)
    T_onehot = np.zeros((5000, 5))
    T_onehot[np.where(T == 0)[0]] = np.array([1, 0, 0, 0, 0])
    T_onehot[np.where(T == 1)[0]] = np.array([0, 1, 0, 0, 0])
    T_onehot[np.where(T == 2)[0]] = np.array([0, 0, 1, 0, 0])
    T_onehot[np.where(T == 3)[0]] = np.array([0, 0, 0, 1, 0])
    T_onehot[np.where(T == 4)[0]] = np.array([0, 0, 0, 0, 1])

    Y0 = 0 + 1.0 * X[:, 0:1]*X[:, 1:2] + 0.5 * X[:, 1:2] + 1.5*(X[:, 2:3]+0.5)**2 + 0.4 * (X[:, 4:5]**3) + 0.6 * X[:, 6:7] + 0.6 * (X[:, 9:10]**2) + np.random.normal(0, 4, 5000).reshape(-1, 1)
    Y1 = 1 + 1.2 * X[:, 0:1]*X[:, 1:2] + 0.8 * X[:, 1:2] + 1.5*(X[:, 2:3]+0.5)**2 + 0.6 * (X[:, 4:5]**3) + 0.4 * X[:, 6:7] + 0.8 * (X[:, 9:10]**2) + np.random.normal(0, 4, 5000).reshape(-1, 1)
    Y2 = 2 + 1.5 * X[:, 0:1]*X[:, 1:2] + 1.0 * X[:, 1:2] + 1.5*(X[:, 2:3]+0.5)**2 + 1.8 * (X[:, 4:5]**3) + 0.2 * X[:, 6:7] + 0.8 * (X[:, 9:10]**2) + np.random.normal(0, 4, 5000).reshape(-1, 1)
    Y3 = 3 + 1.6 * X[:, 0:1]*X[:, 1:2] + 1.2 * X[:, 1:2] + 1.8*(X[:, 2:3]+0.5)**2 + 1.0 * (X[:, 4:5]**3) + 0.6 * X[:, 6:7] + 1.0 * (X[:, 9:10]**2) + np.random.normal(0, 4, 5000).reshape(-1, 1)
    Y4 = 4 + 1.8 * X[:, 0:1]*X[:, 1:2] + 1.5 * X[:, 1:2] + 1.8*(X[:, 2:3]+0.5)**2 + 1.2 * (X[:, 4:5]**3) + 0.4 * X[:, 6:7] + 1.2 * (X[:, 9:10]**2) + np.random.normal(0, 4, 5000).reshape(-1, 1)
    Y = Y0*T_onehot[:, 0:1] + Y1*T_onehot[:, 1:2] + Y2*T_onehot[:, 2:3] + Y3*T_onehot[:, 3:4] + Y4*T_onehot[:, 4:5]

    CE1 = Y1 - Y0
    CE2 = Y2 - Y0
    CE3 = Y3 - Y0
    CE4 = Y4 - Y0
    CE = np.concatenate([CE1, CE2, CE3, CE4], axis=1)

    train_size = 4000
    X_tr = X[:train_size, :]
    T_tr = T[:train_size, :]
    Y_tr = Y[:train_size, :]

    X_te  = X[train_size:, :]

    CE_tr = CE[:train_size, :]
    CE_te = CE[train_size:, :]

    # print(np.mean(CE1))
    # print(np.mean(CE2))
    # print(np.mean(CE3))
    # print(np.mean(CE4))

    # from collections import Counter
    # counter = Counter(T.reshape(-1))
    # distribution = counter.most_common()
    # distribution.sort()
    # print(distribution)
    # assert 1==2

    # [(0, 2936), (1, 835), (2, 783), (3, 398), (4, 48)]

    NRMSE1, NRMSE2, NRMSE3, NRMSE4 = [], [], [], []
    NRMSE1_insample, NRMSE2_insample, NRMSE3_insample, NRMSE4_insample = [], [], [], []
    
    for t in range(CFG.times):
        print(t)
        np.random.seed(t)
        random.seed(t)
        start = time.time()

        if CFG.insample:
            metrics_insample, metrics = train_and_predict(X_tr, T_tr, Y_tr, CE_tr, X_te, CE_te, CFG)
            NRMSE1_insample.append(metrics_insample[0])
            NRMSE2_insample.append(metrics_insample[1])
            NRMSE3_insample.append(metrics_insample[2])
            NRMSE4_insample.append(metrics_insample[3])
        else:
            metrics = train_and_predict(X_tr, T_tr, Y_tr, CE_tr, X_te, CE_te, CFG)
        NRMSE1.append(metrics[0])
        NRMSE2.append(metrics[1])
        NRMSE3.append(metrics[2])
        NRMSE4.append(metrics[3])

        end = time.time()
        print('elaps: ', end-start)

    print('NRMSE1: ', np.mean(NRMSE1), np.std(NRMSE1))
    print('NRMSE2: ', np.mean(NRMSE2), np.std(NRMSE2))
    print('NRMSE3: ', np.mean(NRMSE3), np.std(NRMSE3))
    print('NRMSE4: ', np.mean(NRMSE4), np.std(NRMSE4))

    print('NRMSE1_insample: ', np.mean(NRMSE1_insample), np.std(NRMSE1_insample))
    print('NRMSE2_insample: ', np.mean(NRMSE2_insample), np.std(NRMSE2_insample))
    print('NRMSE3_insample: ', np.mean(NRMSE3_insample), np.std(NRMSE3_insample))
    print('NRMSE4_insample: ', np.mean(NRMSE4_insample), np.std(NRMSE4_insample))


class CFG:
    lr=1e-3
    momentum=0.9
    PS_epoch=300
    epoch=500
    batch_size=1024
    verbose=0
    loss_verbose=0

    times=10
    seed=123
    insample=True
    num_domains=5

    model_type='BNN'        # BNN / Tarnet / DRCFR / IIB / IDRL
    model_name= model_type + '_'

    use_IPW=None       # None / batch / total / PS  

    use_IPM=None        # None / MMD / Wdist / HSIC
    ratio_IPM=1e-1

    use_infomax=False        # True / False
    ratio_infomax=1e-0

    use_DR=False        # True / False
    ratio_DR=1e-3
    ratio_PS=1e-1


if __name__ == '__main__':
    run(CFG)
    print('model type: ', CFG.model_type)