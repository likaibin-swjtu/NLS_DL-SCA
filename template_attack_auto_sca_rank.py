import sys

from keras.optimizers import *

## for visualizing
import matplotlib
import matplotlib.pyplot as plt
import Util.SCA_dataset as datasets
import Util.Attack as Attack

import Util.Template_attack as TA
import Util.Triplet_loss as losses
from keras.optimizers import adam_v2
from scipy import stats

import tensorflow.keras as tk
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.python.keras import backend as K

import keras_tuner as kt
from kerastuner.tuners import *

import numpy as np
import scipy.stats as ss
import random
import math
from sklearn.preprocessing import StandardScaler

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


class acc_Metric(tk.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
        super(acc_Metric, self).__init__(name=name, **kwargs)
        self.m = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.m.update_state(K.equal(K.argmax(y_true[:, :num_classes], axis=-1), K.argmax(y_pred, axis=-1)))

    def result(self):
        return self.m.result()

    def reset_states(self):
        self.m.reset_states()


def rank_compute(prediction, att_plt, byte, output_rank):
    hw = [bin(x).count("1") for x in range(256)]
    (nb_traces, nb_hyp) = prediction.shape

    key_log_prob = np.zeros(256)
    prediction = np.log(prediction + 1e-40)
    rank_evol = np.full(nb_traces, 255)

    for i in range(nb_traces):
        for k in range(256):
            if leakage_model == 'ID':
                key_log_prob[k] += prediction[i, AES_Sbox[k ^ int(att_plt[i, byte])]]
            else:
                key_log_prob[k] += prediction[i, hw[AES_Sbox[k ^ int(att_plt[i, byte])]]]
        rank_evol[i] = rk_key(key_log_prob, correct_key)

    if output_rank:
        return rank_evol
    else:
        return key_log_prob


def perform_attacks(nb_traces, predictions, plt_attack, nb_attacks=1, byte=2, shuffle=True, output_rank=False):
    (nb_total, nb_hyp) = predictions.shape
    all_rk_evol = np.zeros((nb_attacks, nb_traces))

    for i in range(nb_attacks):
        if shuffle:
            l = list(zip(predictions, plt_attack))
            random.shuffle(l)
            sp, splt = list(zip(*l))
            sp = np.array(sp)
            splt = np.array(splt)
            att_pred = sp[:nb_traces]
            att_plt = splt[:nb_traces]

        else:
            att_pred = predictions[:nb_traces]
            att_plt = plt_attack[:nb_traces]

        key_log_prob = rank_compute(att_pred, att_plt, byte, output_rank)
        if output_rank:
            all_rk_evol[i] = key_log_prob

    if output_rank:
        return np.mean(all_rk_evol, axis=0)
    else:
        return np.float32(key_log_prob)


def calculate_key_prob(y_true, y_pred):
    plt_attack = y_true[:, num_classes:]
    if plt_attack[0][0] == 1:  # check if data is from validation set, then compute GE
        GE = perform_attacks(nb_traces_attacks, y_pred, plt_attack[:, 1:], nb_attacks, byte=2)
    else:  # otherwise, return zeros
        GE = np.float32(np.zeros(256))
    return GE


@tf.function
def tf_calculate_key_prob(y_true, y_pred):
    _ret = tf.numpy_function(calculate_key_prob, [y_true, y_pred], tf.float32)
    return _ret


def calculate_rank(y_pred):
    pred_rank = ss.rankdata(y_pred, axis=1) - 1
    return pred_rank / 255


# Objective: GE
def rk_key(rank_array, key):
    key_val = rank_array[key]
    final_rank = np.float32(np.where(np.sort(rank_array)[::-1] == key_val)[0][0])
    if math.isnan(float(final_rank)) or math.isinf(float(final_rank)):
        return np.float32(256)
    else:
        return np.float32(final_rank)


def calculate_Lm(key_prob):
    key_rank = 256 - ss.rankdata(key_prob)
    corr, _ = stats.pearsonr(ranked_LDD, key_rank)
    if math.isnan(float(corr)) or math.isinf(float(corr)):
        return np.float32(0)
    else:
        return np.float32(corr)


class Rank(tf.keras.callbacks.Callback):
    def __init__(self, validation=None):
        super(Rank, self).__init__()
        self.validation = validation

    def set_params(self, params):
        super(Rank, self).set_params(params)
        # if 'my_metric_val' not in self.params['metrics']:
        #     self.params['metrics'].append('my_metric_val')

    def on_epoch_end(self, epoch, logs=None):

        if self.validation:
            logs['my_metric_val'] = float('inf')
            X_profiling_metric, Y_profiling_metric, X_attack_metric = self.validation[0], self.validation[1], \
                self.validation[2]
            x_profiling_embeddings_metric = self.model.predict(X_profiling_metric)
            x_embeddings_metric = self.model.predict(X_attack_metric)
            mean_v_metric, cov_v_metric, classes_metric = TA.template_training(x_profiling_embeddings_metric,
                                                                               Y_profiling_metric, pool=False)
            TA_prediction_metric = TA.template_attacking_proba(mean_v_metric, cov_v_metric, x_embeddings_metric,
                                                               classes_metric)
            avg_rank_metric, all_rank_metric = Attack.perform_attacks(nb_traces_attacks, TA_prediction_metric,
                                                                      correct_key, plt_attack,
                                                                      log=True, dataset=dataset, nb_attacks=nb_attacks)
            if avg_rank_metric[-1] > 0:
                logs['my_metric_val'] = nb_traces_attacks
            else:
                logs['my_metric_val'] = np.argmax(avg_rank_metric < 1)

            return logs['my_metric_val']

    def on_train_end(self, logs=None):
        print(logs)  # None


class Lm_Metric(tk.metrics.Metric):
    def __init__(self, name='lm', **kwargs):
        super(Lm_Metric, self).__init__(name=name, **kwargs)
        self.acc_sum = self.add_weight(name='acc_sum', shape=(256), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.acc_sum.assign_add(tf_calculate_key_prob(y_true, y_pred))

    def result(self):
        return tf.numpy_function(calculate_Lm, [self.acc_sum], tf.float32)

    def reset_states(self):
        self.acc_sum.assign(K.zeros(256))


class key_rank_Metric(tk.metrics.Metric):
    def __init__(self, name='key_rank', **kwargs):
        super(key_rank_Metric, self).__init__(name=name, **kwargs)
        self.acc_sum = self.add_weight(
            name='acc_sum', shape=(256), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.acc_sum.assign_add(tf_calculate_key_prob(y_true, y_pred))

    def result(self):
        return tf.numpy_function(rk_key, [self.acc_sum, correct_key], tf.float32)

    def reset_states(self):
        self.acc_sum.assign(K.zeros(256))


def build_model_mlp(hp):
    activation = hp.Choice('activation', values=["relu", "tanh", "selu", "elu"])

    model = tk.models.Sequential()
    model.add(layers.Dense(units=hp.Int('dense_' + str(0) + '_units',
                                        min_value=100,
                                        max_value=400,
                                        step=100),
                           activation=activation,
                           input_shape=(input_length,)))

    for j in range(hp.Int('n_dense_layers', 1, 10)):
        model.add(layers.Dense(units=hp.Int('dense_' + str(j) + '_units',
                                            min_value=100,
                                            max_value=400,
                                            step=100),
                               activation=activation,
                               ))
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    loss = losses.lifted_struct_loss(alpha_value, margin, num_classes)
    model.compile(optimizer=adam_v2.Adam(lr=hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5])),
                  loss=loss, metrics=metric)

    model.summary()
    return model


def build_model(hp):
    """tunning network"""
    # activation = hp.Choice('activation', values=["relu", "tanh", "selu", "elu"])
    #
    # model = tk.models.Sequential()
    # model.add(Conv1D(hp.Int('conv_0_filters', min_value=56, max_value=88, step=8),
    #                  hp.Int('conv_0_kernal_size', min_value=12, max_value=16, step=1), padding='same',
    #                  input_shape=(input_length, 1)))
    # model.add(Activation(activation))
    # model.add(AveragePooling1D(15, strides=15))
    # # model.add(AveragePooling1D(hp.Int('averagePooling_size', min_value=12, max_value=17, step=1),
    # #                            hp.Int('averagePooling_stride', min_value=12, max_value=17, step=1)))
    # model.add(Conv1D(hp.Int('conv_' + str(1) + '_filters', min_value=120, max_value=168, step=8),
    #                  hp.Int('conv_' + str(1) + '_kernal_size', min_value=2, max_value=6, step=1),
    #                  padding='same'))
    # model.add(Activation(activation))
    # model.add(AveragePooling1D(2, strides=2))
    # # model.add(AveragePooling1D(hp.Int('averagePooling_size', min_value=2, max_value=4, step=1),
    # #                            hp.Int('averagePooling_stride', min_value=2, max_value=4, step=1)))
    # model.add(Flatten())
    # # model.add(layers.Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16)))
    # model.add(Dense(embedding_size))
    # loss = losses.lifted_struct_loss(alpha_value, margin, num_classes)
    # model.compile(optimizer=adam_v2.Adam(lr=hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5])),
    #               loss=loss)
    #
    # model.summary()
    '''original network'''
    model = Sequential()
    model.add(Conv1D(input_shape=(input_length, 1), filters=64, kernel_size=15, padding="same", activation="selu"))
    model.add(AveragePooling1D(pool_size=12, strides=12))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="selu"))
    model.add(AveragePooling1D(pool_size=2, strides=2))
    # model.add(Conv1D(filters=128, kernel_size=3, padding="same", kernel_regularizer=l2(0.0001), activation="selu"))
    # model.add(AveragePooling1D(pool_size=2,strides=2))
    model.add(Flatten(name='Flatten'))
    model.add(Dense(embedding_size))
    alpha_value = hp.Float('alpha_value', min_value=0.1, max_value=1, step=0.1)
    margin = hp.Float('margin', min_value=0.2, max_value=2, step=0.1)
    loss = losses.lifted_struct_loss(alpha_value, margin, num_classes)
    model.compile(optimizer=adam_v2.Adam(lr=hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5])),
                  loss=loss)
    return model


def calculate_LDD(k_c=34, mode='HW'):
    p = range(256)
    hw = [bin(x).count("1") for x in range(256)]
    k_all = range(256)
    container = np.zeros((len(k_all), len(p)), int)
    variance = np.zeros((256,))

    if mode == 'HW':
        for i in range(len(p)):
            for j in range(len(k_all)):
                container[j][i] = hw[labelize(p[i], k_all[j])]
        for k in range(256):
            variance[k] = np.sum(
                abs(np.power(container[k_c] - container[k], 2)))

    elif mode == 'ID':
        for i in range(len(p)):
            for j in range(len(k_all)):
                container[j][i] = labelize(p[i], k_all[j])
        for k in range(256):
            variance[k] = np.sum(abs(np.power(container[k_c] - container[k], 2)))

    else:
        for i in range(len(p)):
            for j in range(len(k_all)):
                container[j][i] = calculate_MSB(labelize(p[i], k_all[j]))
        for k in range(256):
            variance[k] = np.sum(abs(container[k_c] - container[k]))
    return variance


def labelize(plaintexts, keys):
    return AES_Sbox[plaintexts ^ keys]


def calculate_MSB(data):
    if isinstance(data, (list, tuple, np.ndarray)):
        container = np.zeros((np.shape(data)), int)
        for i in range(len(data)):
            if data[i] >= 128:
                container[i] = 1
            else:
                container[i] = 0
    else:
        if data >= 128:
            container = 1
        else:
            container = 0
    return container


if __name__ == "__main__":  # File root for dataset and results
    data_root = 'Data/'
    model_root = 'Model/'
    result_root = 'Result/'
    save_root = 'root the save the searching history'

    # Dataset settings
    # Note: Use code in Util/Generate_ASCAD_X.py to generate ASCAD datasets used in the paper (4,000 features, raw traces required)
    # Note: Attacking on the default ASCAD_F (700 features) or ASCAD_R (1,400 features) would work as well!
    datasetss = ['ASCAD_rand']  # ['ASCAD','ASCAD_rand','AESHD']
    leakage_models = ['HW']  # ['HW','ID']
    profiling_traces = 0  # 0: default
    noise_type = 'desync'
    noise_level = 100

    # 搜索方式及其目标函数
    searching_method = 'BO'
    objective = 'my_metric_val'  # 'val_lm/val_key_rank/val_acc/val_key_rank_new'
    max_trails = 10

    # Triplet settings
    classifier = 'Triplet'
    embedding_size = 48  # Dimension of output features
    # alpha_values = 0.7  # 0: optimal
    # margin = 1  # Triplet margin
    beta = 0.002

    # Training settings
    batch_size = 256  # Triplet training batch size
    epochs = 10  # Triplet training epoch
    train_flag = True  # True: train a model; False: load a model
    nb_attacks = 10  # number of attacks for GE calculation
    attack_model = 'CNN'  # 所要学习的网络模型

    # Saving settings
    index = 0  # naming index
    save_folder = 'test'

    matplotlib.rcParams.update({'font.size': 12})
    # alpha_value = alpha_values
    # The data, split between train and test sets
    for dataset in datasetss:
        if dataset == 'ASCAD':
            correct_key = 224
            nb_traces_attacks = 10000
            (X_profiling, X_attack), (Y_profiling_ID, Y_attack_ID), (plt_profiling, plt_attack) = datasets.load_ascad(
                profiling_traces=profiling_traces, leakage_model='ID')
        elif dataset == 'ASCAD_rand':
            correct_key = 34
            nb_traces_attacks = 10000
            (X_profiling, X_attack), (Y_profiling_ID, Y_attack_ID), (
                plt_profiling, plt_attack) = datasets.load_ascad_rand(profiling_traces=profiling_traces,
                                                                      leakage_model='ID')
        elif dataset == 'AESHD':
            correct_key = 200
            nb_traces_attacks = 5000
            (X_profiling, X_attack), (Y_profiling_ID, Y_attack_ID), (plt_profiling, plt_attack) = datasets.load_aeshd(
                data_root + '/', profiling_traces=profiling_traces, leakage_model='ID')

        if noise_type == 'gnoise':
            X_profiling = datasets.addGussianNoise(X_profiling, noise_level)
            X_attack = datasets.addGussianNoise(X_attack, noise_level)
        if noise_type == 'desync':
            X_profiling = datasets.addDesync(X_profiling, int(noise_level))
            X_attack = datasets.addDesync(X_attack, int(noise_level))

        # 给定网络模型的输入格式并进行归一化
        input_length = len(X_profiling[0])
        scaler = StandardScaler()
        X_profiling = scaler.fit_transform(X_profiling)
        X_attack = scaler.transform(X_attack)

        # # set alpha values
        # if alpha_value == 0:
        #     if dataset == 'AESHD':
        #         alpha_value = 0.6
        #     else:
        #         alpha_value = 0.1
        # else:
        #     alpha_value = alpha_values

        if attack_model == 'MLP':
            X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
            X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
            build_model = build_model_mlp
        else:
            X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
            X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
            build_model = build_model

        for leakage_model in leakage_models:
            # calculate ideal key rank
            ranked_LDD = ss.rankdata(calculate_LDD(correct_key, mode=leakage_model))

            # select the objective function
            if objective == 'val_accuracy':
                metric = [acc_Metric(), key_rank_Metric()]
                direction = 'max'
            elif objective == 'val_loss':
                direction = 'min'
            elif objective == 'val_lm':
                metric = [Lm_Metric(), key_rank_Metric()]
                direction = 'max'
            elif objective == 'val_key_rank':
                metric = [key_rank_Metric()]
                direction = 'min'
            elif objective == 'my_metric_val':
                metric = [Rank()]
                direction = 'min'
            else:
                print('No objective function defined!')

            if leakage_model == 'HW':
                num_classes = 9
                Y_profiling = datasets.calculate_HW(Y_profiling_ID)
                Y_attack = datasets.calculate_HW(Y_attack_ID)
            else:
                num_classes = 256
                Y_profiling = Y_profiling_ID
                Y_attack = Y_attack_ID

            if searching_method == 'BO':
                print('Searching method: BO')
                tuner = BayesianOptimization(hypermodel=build_model,
                                             objective=kt.Objective(objective, direction=direction),
                                             max_trials=max_trails,
                                             executions_per_trial=1,
                                             directory=save_root,
                                             project_name=dataset + '_' + attack_model + '_' + leakage_model + '_' + searching_method + '_' + objective + '_' + str(
                                                 2),
                                             overwrite=True)
            else:
                print('Searching method: RA')
                tuner = RandomSearch(build_model,
                                     objective=kt.Objective(objective, direction=direction),
                                     max_trials=max_trails,
                                     executions_per_trial=1,
                                     directory=save_root,
                                     project_name=dataset + '_' + attack_model + '_' + leakage_model + '_' + searching_method + '_' + objective + '_' + str(
                                         2),
                                     overwrite=True)

            tuner.search_space_summary()
            tuner.search(x=X_profiling, y=Y_profiling, epochs=10, batch_size=batch_size,
                         callbacks=[Rank(validation=(X_profiling, Y_profiling, X_attack))], verbose=2)
            tuner.results_summary()

            print('Retrain the best model with 10 epochs...')
            best_hp = tuner.get_best_hyperparameters()[0]
            model = tuner.hypermodel.build(best_hp)
            model.fit(x=X_profiling, y=Y_profiling, batch_size=batch_size, verbose=2, epochs=epochs,
                      callbacks=[Rank(validation=(X_profiling, Y_profiling, X_attack))])

            # Tamplate attack
            print('======Tamplate attack======')
            try:
                x_profiling_embeddings = model.predict(X_profiling)
                x_embeddings = model.predict(X_attack)
                mean_v, cov_v, classes = TA.template_training(x_profiling_embeddings, Y_profiling, pool=False)
                TA_prediction = TA.template_attacking_proba(mean_v, cov_v, x_embeddings, classes)
                avg_rank, all_rank = Attack.perform_attacks(nb_traces_attacks, TA_prediction, correct_key, plt_attack,
                                                            log=True, dataset=dataset, nb_attacks=nb_attacks)

                # val_t_ge0 = np.argmax(avg_rank < 1)
                # if val_t_ge0 == 0:
                #     val_t_ge0_value.append([nb_traces_attacks])
                #     print('GE smaller than 1:', nb_traces_attacks)
                # else:
                #     val_t_ge0_value.append([val_t_ge0])
                #     print('GE smaller than 1:', val_t_ge0)

                print('GE: ', avg_rank[-1])
                print('GE smaller than 1: ', np.argmax(avg_rank < 1))
                print('GE smaller than 5: ', np.argmax(avg_rank < 5))
                print('GE smaller than 10: ', np.argmax(avg_rank < 10))
                print('Print and save GE TA')
                # write2file("./Result/ASCAD.txt",
                #            f"Alpha: {alpha_value}\n"
                #            f"margin: {margin}\n"
                #            f"GE: {np.argmax(avg_rank [-1])}\n"
                #            f"GE smaller than 1: {np.argmax(avg_rank < 1)}\n"
                #            f"GE smaller than 5: {np.argmax(avg_rank < 5)}\n"
                #            f"GE smaller than 10: {np.argmax(avg_rank < 10)}\n"
                #            )

                plt.plot(avg_rank)
                plt.xlabel('Number of Attack Traces')
                plt.ylabel('Guessing Entropy')
                # plt.savefig(result_root + save_folder + '/GE_{}.png'.format(test_info))
                # np.save(result_root + save_folder + '/GE_{}.npy'.format(test_info), avg_rank)
                plt.savefig(result_root + dataset + '-' + leakage_model + '-desync' + str(
                    noise_level) + 'Guessing Entropy_lf' + 'alpha_value' + str(
                    best_hp.values['alpha_value']) + 'magin' + str(
                    best_hp.values['margin']) + '.jpg', dpi=1200, bbox_inches='tight')
                # plt.show()
                plt.close()

                # plt.clf()
                # np.save(result_root + 'TGE0_' + saving_name + '.npy', np.array(val_t_ge0_value).flatten())

            except np.linalg.LinAlgError as e:
                print(e)
                print('Tamplate attack error, singular matrix maybe')

# if __name__ == '__main__':
#     # for alpha_values in np.arange(0.0, 0.1, 1.1, dtype=np.float32):
#     main()
