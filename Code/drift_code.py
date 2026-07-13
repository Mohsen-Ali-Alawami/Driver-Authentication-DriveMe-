# import sklearn.ensemble as ensemble
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import timeit
from sklearn.svm import OneClassSVM
import pickle
from os import path
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, RepeatVector, Activation, TimeDistributed, LSTM, Bidirectional, SimpleRNN, GRU
from tensorflow import keras
from tensorflow.keras import regularizers
###########################################################################################


def Tr_Ts_drfit_split_process(normal_data, normal_data_new, attack_data, split_ratio):
    normal_data = normal_data.fillna(0)
    normal_data_new = normal_data_new.fillna(0)
    attack_data = attack_data.fillna(0)
    # ##################### MinMaxScaler ##############################
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(normal_data)
    normal_data = pd.DataFrame(scaler.transform(normal_data))
    scaler = scaler.fit(normal_data_new)
    normal_data_new = pd.DataFrame(scaler.transform(normal_data_new))
    scaler = scaler.fit(attack_data)
    attack_data = pd.DataFrame(scaler.transform(attack_data))
    ###################################################
    # splitting dataframe by row index
    n_train = int(len(normal_data) * split_ratio)
    n_test = int(len(attack_data) * (1 - split_ratio))
    train_normal = normal_data.iloc[:n_train, :]
    # print("Train_normal shape: ", train_normal.shape)
    test_normal = normal_data_new.iloc[:n_test, :]
    test_normal[len(test_normal.columns)] = 0  # Add "0" as attack labels to all test data
    # print("test_normal shape: ", test_normal.shape)
    test_attack = attack_data.iloc[:n_test, :]
    test_attack[len(test_attack.columns)] = 1  # Add "1" as attack labels to all test data
    # print("test_attack shape: ", test_attack.shape)
    return train_normal, test_normal, test_attack
