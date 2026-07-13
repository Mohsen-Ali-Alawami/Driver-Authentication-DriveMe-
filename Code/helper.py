import os
import re
import sklearn as sk
from sklearn.model_selection import train_test_split
from natsort import natsorted

import numpy as np
# import sklearn as sk
# import sklearn.preprocessing as prep
# import sklearn.model_selection as model_selection
# import sklearn.metrics as metrics
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
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, RepeatVector, Activation, TimeDistributed, LSTM, Bidirectional, SimpleRNN, GRU
# from tensorflow import keras
# from tensorflow.keras import regularizers
###########################################################################################

def load_Belt_data(Belt_path):
    iterations= ["1", "10", "2", "3", "4", "5", "6", "7", "8", "9"]
    index = 0
    data_one_user = []
    for root, dirs, files in os.walk(Belt_path):
        for name in files:
            df = pd.read_csv(os.path.join(Belt_path, name))
            # print(os.path.join(Belt_path, name))
            df.columns = [1, 2, 3, 4, 5,
                          6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15,
                          16, 17, 18, 19, 20,
                          21, 22, 23, 24, 25,
                          26, 27, 28, 29, 30
                          ]
            # print(df.shape)
            # df.to_csv(name + ".csv")
            data_one_user.append(df)
    one_user_df = pd.concat(data_one_user, ignore_index=True)
    # print(one_user_df)
    # print(one_user_df.shape)
    return one_user_df


def convert_bytes(size, unit=None):
    if unit == "KB":
        return print('Model size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('Model size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('Model size: ' + str(size) + ' bytes')


def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size


def load_Seat_data(Seat_path):
    df_list= []
    for root, dirs, files in os.walk(Seat_path):
        for name in files:
            df = pd.read_csv(os.path.join(Seat_path, name), index_col=None)
            print(name)
            print(df.shape)
            del df[df.columns[0]]#delet time column (first column)
            df.drop(columns=df.columns[-1], axis=1, inplace=True)#delete the 31th col, to be same as Belt (i.e. 30 cols)
            # print(os.path.join(Seat_path, name))
            # print(df.shape)
            # print(df)
            df = pd.DataFrame(df.values)
            df_list.append(df)
    one_user_df = pd.concat(df_list, ignore_index=True)
    # print(one_user_df.shape)
    return one_user_df


def Data_process(data):# split each user's pressure data into dataframe into train/test and then contact them
    X_train =[]
    X_test = []
    y_train = []
    y_test = []
    for i in range(len(data)):#
        data_df_ith = data[i]
        # print(data_df_ith.shape)
        X = data_df_ith.iloc[:,:-1]
        y = data_df_ith.iloc[:, -1:]

        X_train_ith, X_test_ith, y_train_ith, y_test_ith = train_test_split(X, y, test_size=0.4, random_state=0)
        # print(X_train_ith.shape)
        # print(X_test_ith.shape)
        X_train.append(X_train_ith)
        X_test.append(X_test_ith)
        y_train.append(y_train_ith)
        y_test.append(y_test_ith)
    final_X_train = pd.concat(X_train, ignore_index=True,sort=False)
    final_X_test = pd.concat(X_test, ignore_index=True)
    # print(final_X_train.shape)
    # final_X_train.to_csv("final_X_train.csv")
    # print(final_X_test.shape)
    final_y_train = pd.concat(y_train, ignore_index=True)
    final_y_test = pd.concat(y_test, ignore_index=True)
    return final_X_train, final_X_test, final_y_train, final_y_test

def Data_process2(normal_data, attack_data, split_ratio):
    normal_data = normal_data.fillna(0)
    attack_data = attack_data.fillna(0)
    # ##################### MinMaxScaler ##############################
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(normal_data)
    normal_data = pd.DataFrame(scaler.transform(normal_data))
    scaler = scaler.fit(attack_data)
    attack_data = pd.DataFrame(scaler.transform(attack_data))
    ###################################################
    # splitting dataframe by row index
    n_train = int(len(normal_data) * split_ratio)
    n_test = int(len(attack_data)* (1-split_ratio))
    train_normal = normal_data.iloc[:n_train, :]
    # print("Train_normal shape: ", train_normal.shape)
    test_normal = normal_data.iloc[n_train + 1:, :]
    test_normal[len(test_normal.columns)]= 0  # Add "0" as attack labels to all test data
    # print("test_normal shape: ", test_normal.shape)
    test_attack = attack_data.iloc[:n_test, :]
    test_attack[len(test_attack.columns)] = 1  # Add "1" as attack labels to all test data
    # print("test_attack shape: ", test_attack.shape)
    return train_normal, test_normal, test_attack

def Data_process3(train_normal, test_normal, test_attack):
    train_normal = train_normal.fillna(0)
    test_normal = test_normal.fillna(0)
    test_attack = test_attack.fillna(0)

    # ##################### MinMaxScaler ##############################
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(train_normal)
    train_normal = pd.DataFrame(scaler.transform(train_normal))
    scaler = scaler.fit(test_normal)
    test_normal = pd.DataFrame(scaler.transform(test_normal))
    scaler = scaler.fit(test_attack)
    test_attack = pd.DataFrame(scaler.transform(test_attack))
    ###################################################

    test_normal[len(test_normal.columns)] = 0  # Add "0" as attack labels to all test data
    # print("test_normal shape: ", test_normal.shape)
    test_attack[len(test_attack.columns)] = 1  # Add "1" as attack labels to all test data
    return train_normal, test_normal, test_attack





def calc_metrics(actual, pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    if len(pred) != len(actual):
        raise ValueError("predict and true samples must have the same length")
    predict = np.asarray(pred)
    actual = np.asarray(actual)
    for sample in range(len(predict)):
        if actual[sample] == 0 and predict[sample] == 0:
            TN += 1
        elif actual[sample] == 0 and predict[sample] == 1:
            FP += 1
        elif actual[sample] == 1 and predict[sample] == 0:
            FN += 1
        elif actual[sample] == 1 and predict[sample] == 1:
            TP += 1

    if TP == 0 or FP == 0 or FN == 0:
        if TP == 0:
            TP = 1
        elif FP == 0:
            FP = 1
        elif FN == 0:
            FN = 1

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall, TN, FN, TP, FP

def save_metrics(resutls, filename, Results_path):
    if (not os.path.isdir(Results_path)):  # Create the directory if not exist
        os.makedirs(Results_path)

    file_dir  = os.path.join(Results_path, filename)
    if path.exists(file_dir):
        resutls.to_csv(file_dir, mode = 'a', header = False, index = False)
    else:
        resutls.to_csv(file_dir, header = True, index = False)


# def run_OCSVM(train_normal, test_normal, test_attack, ratio, gamma, nu_1, i, j, model_output_path, filename_model, Results_path, filename):
def run_OCSVM(train_normal, test_normal, test_attack, ratio, gamma, nu_1, i, j, model_output_path, filename_model):
    train_start = timeit.default_timer()
    Test_normal_attack = pd.concat([test_normal, test_attack], ignore_index=True)
    print("Test_normal_attack:", Test_normal_attack.shape)
    actual_labels = Test_normal_attack.iloc[:, -1:]
    # print(actual)
    print("Actual labels: ", len(actual_labels))
    Test_normal_attack = Test_normal_attack.iloc[:, :-1]  # drop last col [labels] from normal data
    print("Test_normal_attack:", Test_normal_attack.shape)
    # svm = OneClassSVM(kernel='rbf')
    svm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu_1)
    print(svm)
    svm.fit(train_normal)
    train_end = timeit.default_timer()
    Train_elapsed_time = (train_end - train_start)
    # print("Training time for the iteration: ", Train_elapsed_time)
    model_path = os.path.join(model_output_path, filename_model)  # build a path for each file in the input folder
    pickle.dump(svm, open(model_path, 'wb'))
    print(convert_bytes(get_file_size(model_path), "KB"))

    # test_start = timeit.default_timer()
    # pred = svm.predict(Test_normal_attack)
    # print("len(pred)" , len(pred))
    # # print("pred:  ", pred)
    # pred = [0 if i == 1 else 1 for i in pred]
    # print("len(pred)", len(pred))
    # # print("pred:  ", pred)
    # # normal_pred = pred[:len(test_normal)]
    # # attack_pred = pred[len(test_normal)+1:]
    # f1, precision, recall, TN, FN, TP, FP = calc_metrics(actual_labels, pred)
    # test_end = timeit.default_timer()
    # Test_elapsed_time = (test_end - test_start)
    # # print("Testing time for the iteration: ", Test_elapsed_time, '% >>>>> Iteration done!')
    # print(f1, precision, recall, TN, FN, TP, FP)
    # output = [str(i), str(i),str(j), train_normal.shape[0], test_normal.shape[0], test_attack.shape[0],
    #           TN, FN, TP, FP, f1, precision, recall, Train_elapsed_time, Test_elapsed_time, gamma, nu_1, ratio]
    # results = pd.DataFrame([output], columns=['Train user', 'Test user1', 'Test user2', 'Train Normal size', 'Test Normal size', 'Test attack size',
    #                                           'TN', 'FN', 'TP', 'FP', 'F1-score', 'precision', 'Recall',
    #                                           'train_time (sec)', 'test_time (sec)', 'gamma', 'nu', 'Ratio'])
    # save_metrics(results, filename, Results_path)



# def run_IF(train_normal, test_normal, test_attack, ratio, estimator, i, j, model_output_path, filename_model, Results_path, filename):
def run_IF(train_normal, test_normal, test_attack, ratio, estimator, i, j, model_output_path, filename_model):
    train_start = timeit.default_timer()
    Test_normal_attack = pd.concat([test_normal, test_attack], ignore_index=True)
    print("Test_normal_attack:", Test_normal_attack.shape)
    actual_labels = Test_normal_attack.iloc[:, -1:]
    # print(actual)
    print("Actual labels: ", len(actual_labels))
    Test_normal_attack = Test_normal_attack.iloc[:, :-1]  # drop last col [labels] from normal data
    print("Test_normal_attack:", Test_normal_attack.shape)
    clf = IsolationForest(n_estimators=estimator)
    print(clf)
    clf.fit(train_normal)
    train_end = timeit.default_timer()
    Train_elapsed_time = (train_end - train_start)
    # print("Training time for the iteration: ", Train_elapsed_time)
    model_path = os.path.join(model_output_path, filename_model)  # build a path for each file in the input folder
    pickle.dump(clf, open(model_path, 'wb'))
    print(convert_bytes(get_file_size(model_path), "KB"))

    # test_start = timeit.default_timer()
    # pred = clf.predict(Test_normal_attack)
    # print("len(pred)" , len(pred))
    # # print("pred:  ", pred)
    # pred = [0 if i == 1 else 1 for i in pred]
    # print("len(pred)", len(pred))
    # # print("pred:  ", pred)
    # # normal_pred = pred[:len(test_normal)]
    # # attack_pred = pred[len(test_normal)+1:]
    # f1, precision, recall, TN, FN, TP, FP = calc_metrics(actual_labels, pred)
    # test_end = timeit.default_timer()
    # Test_elapsed_time = (test_end - test_start)
    # # print("Testing time for the iteration: ", Test_elapsed_time, '% >>>>> Iteration done!')
    # print(f1, precision, recall, TN, FN, TP, FP)
    # output = [str(i), str(i),str(j), train_normal.shape[0], test_normal.shape[0], test_attack.shape[0],
    #           TN, FN, TP, FP, f1, precision, recall, Train_elapsed_time, Test_elapsed_time, estimator, ratio]
    # results = pd.DataFrame([output], columns=['Train user', 'Test user1', 'Test user2', 'Train Normal size', 'Test Normal size', 'Test attack size',
    #                                           'TN', 'FN', 'TP', 'FP', 'F1-score', 'precision', 'Recall',
    #                                           'train_time (sec)', 'test_time (sec)', 'estimator', 'Ratio'])
    # save_metrics(results, filename, Results_path)


# def run_LOF(train_normal, test_normal, test_attack, ratio, n_neighbors, i, j, model_output_path, filename_model, Results_path, filename):
def run_LOF(train_normal, test_normal, test_attack, ratio, n_neighbors, i, j, model_output_path, filename_model):
    train_start = timeit.default_timer()
    Test_normal_attack = pd.concat([test_normal, test_attack], ignore_index=True)
    print("Test_normal_attack:", Test_normal_attack.shape)
    actual_labels = Test_normal_attack.iloc[:, -1:]
    # print(actual)
    print("Actual labels: ", len(actual_labels))
    Test_normal_attack = Test_normal_attack.iloc[:, :-1]  # drop last col [labels] from normal data
    print("Test_normal_attack:", Test_normal_attack.shape)
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    print(clf)
    clf.fit(train_normal)
    train_end = timeit.default_timer()
    Train_elapsed_time = (train_end - train_start)
    # print("Training time for the iteration: ", Train_elapsed_time)
    model_path = os.path.join(model_output_path, filename_model)  # build a path for each file in the input folder
    pickle.dump(clf, open(model_path, 'wb'))
    print(convert_bytes(get_file_size(model_path), "KB"))

    # test_start = timeit.default_timer()
    # pred = clf.predict(Test_normal_attack)
    # print("len(pred)" , len(pred))
    # # print("pred:  ", pred)
    # pred = [0 if i == 1 else 1 for i in pred]
    # print("len(pred)", len(pred))
    # # print("pred:  ", pred)
    # # normal_pred = pred[:len(test_normal)]
    # # attack_pred = pred[len(test_normal)+1:]
    # f1, precision, recall, TN, FN, TP, FP = calc_metrics(actual_labels, pred)
    # test_end = timeit.default_timer()
    # Test_elapsed_time = (test_end - test_start)
    # # print("Testing time for the iteration: ", Test_elapsed_time, '% >>>>> Iteration done!')
    # print(f1, precision, recall, TN, FN, TP, FP)
    # output = [str(i), str(i),str(j), train_normal.shape[0], test_normal.shape[0], test_attack.shape[0],
    #           TN, FN, TP, FP, f1, precision, recall, Train_elapsed_time, Test_elapsed_time, n_neighbors, ratio]
    # results = pd.DataFrame([output], columns=['Train user', 'Test user1', 'Test user2', 'Train Normal size', 'Test Normal size', 'Test attack size',
    #                                           'TN', 'FN', 'TP', 'FP', 'F1-score', 'precision', 'Recall',
    #                                           'train_time (sec)', 'test_time (sec)', 'n_neighbors', 'Ratio'])
    # save_metrics(results, filename, Results_path)



# def run_EE(train_normal, test_normal, test_attack, ratio, contaminations, i, j, model_output_path, filename_model, Results_path, filename):
def run_EE(train_normal, test_normal, test_attack, ratio, contaminations, i, j, model_output_path, filename_model):
    train_start = timeit.default_timer()
    Test_normal_attack = pd.concat([test_normal, test_attack], ignore_index=True)
    print("Test_normal_attack:", Test_normal_attack.shape)
    actual_labels = Test_normal_attack.iloc[:, -1:]
    # print(actual)
    print("Actual labels: ", len(actual_labels))
    Test_normal_attack = Test_normal_attack.iloc[:, :-1]  # drop last col [labels] from normal data
    print("Test_normal_attack:", Test_normal_attack.shape)
    clf = EllipticEnvelope(random_state=0, contamination= contaminations).fit(train_normal)
    print(clf)
    # clf.fit(train_normal)
    train_end = timeit.default_timer()
    Train_elapsed_time = (train_end - train_start)
    # print("Training time for the iteration: ", Train_elapsed_time)
    model_path = os.path.join(model_output_path, filename_model)  # build a path for each file in the input folder
    pickle.dump(clf, open(model_path, 'wb'))
    print(convert_bytes(get_file_size(model_path), "KB"))

    # test_start = timeit.default_timer()
    # pred = clf.predict(Test_normal_attack)
    # print("len(pred)" , len(pred))
    # # print("pred:  ", pred)
    # pred = [0 if i == 1 else 1 for i in pred]
    # print("len(pred)", len(pred))
    # # print("pred:  ", pred)
    # # normal_pred = pred[:len(test_normal)]
    # # attack_pred = pred[len(test_normal)+1:]
    # f1, precision, recall, TN, FN, TP, FP = calc_metrics(actual_labels, pred)
    # test_end = timeit.default_timer()
    # Test_elapsed_time = (test_end - test_start)
    # # print("Testing time for the iteration: ", Test_elapsed_time, '% >>>>> Iteration done!')
    # print(f1, precision, recall, TN, FN, TP, FP)
    # output = [str(i), str(i),str(j), train_normal.shape[0], test_normal.shape[0], test_attack.shape[0],
    #           TN, FN, TP, FP, f1, precision, recall, Train_elapsed_time, Test_elapsed_time, contaminations, ratio]
    # results = pd.DataFrame([output], columns=['Train user', 'Test user1', 'Test user2', 'Train Normal size', 'Test Normal size', 'Test attack size',
    #                                           'TN', 'FN', 'TP', 'FP', 'F1-score', 'precision', 'Recall',
    #                                           'train_time (sec)', 'test_time (sec)', 'contaminations', 'Ratio'])
    # save_metrics(results, filename, Results_path)


def create_sequences(X, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
    return np.array(Xs)
def Average(lst):
    return sum(lst) / len(lst)


def compute_train_mae(model, x_train ):
    x_train_pred = model.predict(x_train)
    # print("x_train: \n", x_train[0:3])
    # print("x_train: \n", x_train.shape)
    # print("x_train_pred: \n", x_train_pred[0:3])
    # print("x_train_pred: \n", x_train_pred.shape)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)#get the MAE matrix from training
    # print("train_mae_loss: \n", train_mae_loss[0:3])
    # print("train_mae_loss: \n", train_mae_loss.shape)
    train_mae_loss_avg = train_mae_loss.mean(axis=0)  # axis=0: Apply the calculation “column-wise” , axis=1: Apply the calculation “row-wise”.
    # print("Train_mae_loss of all rows [mean]:  ", train_mae_loss_avg.shape)
    # print("Train_mae_loss of all rows [mean]:  ", train_mae_loss_avg)

    train_mae_loss_avg1 = train_mae_loss.mean(axis=1)  # axis=0: Apply the calculation “column-wise” , axis=1: Apply the calculation “row-wise”.
    # print("Train_mae_loss of all cols [mean]:  ", train_mae_loss_avg1.shape)
    # print("Train_mae_loss of all cols [mean]:  ", train_mae_loss_avg1)

    Threshold_row = max(train_mae_loss_avg) #col_wise
    # print("Threshold: ", Threshold_row)

    Threshold_col = max(train_mae_loss_avg1)
    # print("Threshold1: ", Threshold_col)
    threshold =Average([Threshold_row, Threshold_col])

    # Get reconstruction loss threshold.
    # threshold = np.max(train_mae_loss)
    print("Reconstruction of threshold R_C case: ", threshold)
    return threshold


def fit_LSTM_AE_Bidirectional(X_train, nb_epoch, batch_size, no_neurons, Drop_out, learning_rate):
    #create model
    no_neurons_layer2 = int(no_neurons / 2)

    # model2 = keras.Sequential()
    model2 = Sequential()
    #Encoder
    model2.add(Bidirectional(LSTM(units=no_neurons,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, name = 'First_layer')))
    model2.add(Dropout(rate=Drop_out))
    model2.add(Bidirectional(LSTM(units=no_neurons_layer2, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), name='Second_layer')))
    model2.add(Dropout(rate=Drop_out))
    model2.add(RepeatVector(n=X_train.shape[1]))

    # Decoder
    model2.add(Bidirectional(LSTM(units=no_neurons, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),return_sequences=True, name = 'Third_layer')))
    model2.add(Dropout(rate=Drop_out))
    model2.add(Bidirectional(LSTM(units=no_neurons_layer2, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),return_sequences=True, name='Fourth_layer')))
    model2.add(Dropout(rate=Drop_out))
    model2.add(TimeDistributed(Dense(units=X_train.shape[2])))

    # compile model
    model2.compile(loss=keras.losses.mae, optimizer=keras.optimizers.Adam(learning_rate=learning_rate), metrics=["mae", "mse"])

    # fit model
    history = model2.fit(X_train, X_train, epochs=nb_epoch, batch_size=batch_size, validation_split=0.2, shuffle=False, verbose=0)
    # model2.summary()
    # net_type = "_Bi_"
    return history, model2

def prediction(model3 , X_test, test_lables, threshold, timesteps):
    X_test_pred = model3.predict(X_test)
    # print("X_test: \n", X_test[0:3])
    print("X_test: \n", X_test.shape)
    # print("X_test_pred: \n", X_test_pred[0:3])
    print("X_test_pred: \n", X_test_pred.shape)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
    # print("test_mae_loss: \n", test_mae_loss[0:3])
    # print("test_mae_loss: \n", test_mae_loss.shape)
    #
    # # test_mae_loss = test_mae_loss.mean(axis=0)  # axis=0: Apply the calculation “column-wise” , axis=1: Apply the calculation “row-wise”.
    # # print("test_mae_loss of all rows [mean]:  ", test_mae_loss.shape)
    # # print("test_mae_loss of all rows [mean]:  ", test_mae_loss)
    #
    test_mae_loss_avg1 = test_mae_loss.mean(axis=1)  # axis=0: Apply the calculation “column-wise” , axis=1: Apply the calculation “row-wise”.
    # print("test_mae_loss_avg1 of all cols [mean]:  ", test_mae_loss_avg1.shape)
    # print("test_mae_loss_avg1 of all cols [mean]:  ", test_mae_loss_avg1)
    #
    # output_df = pd.DataFrame({'Test_error': test_mae_loss_avg1,'Test_lables': test_lables[:-timesteps]})
    output_df = pd.DataFrame({'Test_error': test_mae_loss_avg1})
    output_df['Test_lables'] = test_lables[:-timesteps]
    output_df =output_df.assign(Threshold=threshold)
    output_df['anomaly_state'] = output_df.Test_error > output_df.Threshold

    # anomalies = output_df[output_df.anomaly_state == True]
    # print("output_df:  \n", output_df)
    # print("output_df:  ", output_df.shape)
    # print("anomalies:  ", anomalies.shape)
    # print("Anomalies:  \n", anomalies)
    return output_df

def calc_metrics_AE(output_df):
    TP=0
    TN=0
    FP=0
    FN=0
    actual = output_df.Test_lables
    predict =  output_df.anomaly_state
    if len(predict) != len(actual):
        raise ValueError("predict and true samples must have the same length")
    predict = np.asarray(predict)
    actual = np.asarray(actual)
    for sample in range(len(predict)):
        if actual[sample] ==0 and predict[sample]==False:
            TN +=1
        elif actual[sample] ==0 and predict[sample]==True:
            FP += 1
        elif actual[sample] ==1 and predict[sample]==False:
            FN += 1
        elif actual[sample] ==1 and predict[sample]==True:
            TP += 1


    if TP == 0 or FP == 0 or FN == 0:
        if TP == 0:
            TP =1
        elif FP ==0:
            FP=1
        elif FN==0:
            FN=1

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall, TN, FN, TP, FP


def run_AE(train_normal, test_normal, test_attack, ratio, i, j, Results_path, filename):
    train_start = timeit.default_timer()
    Test_normal_attack = pd.concat([test_normal, test_attack], ignore_index=True)
    print("Test_normal_attack:", Test_normal_attack.shape)
    test_lables = Test_normal_attack.iloc[:, -1:]
    # print(actual)
    print("Actual labels: ", test_lables.shape)
    # print("Actual labels: ", test_lables)


    # print("Actual labels: ", len(test_lables))
    Test_normal_attack = Test_normal_attack.iloc[:, :-1]  # drop last col [labels] from normal data
    print("Test_normal_attack:", Test_normal_attack.shape)
    time_steps = [5]
    n_epochs = [100]
    batch_size = [64]
    no_neurons = [128]
    Drop_out = [0.2]
    learning_rate = [0.001]
    # time_steps = [5, 10]
    # n_epochs = [5, 10, 50, 100]
    # batch_size = [32, 64, 128, 256]
    # no_neurons = [128, 256, 512]
    # Drop_out = [0.2, 0.3]
    # learning_rate = [0.001, 0.01, 0.1]
    for neurons in range(len(no_neurons)):
        for ii in range(len(n_epochs)):
            for jj in range(len(batch_size)):
                for kk in range (len(learning_rate)):
                    for dropout in range (len(Drop_out)):
                        for hh in range(len(time_steps)):
                            print('%..................An iteration evaluation ..........% >>>>> Running!')
                            train_start = timeit.default_timer()
                            X_train = create_sequences(train_normal, time_steps[hh])
                            print("Reshaped training data with time_steps: ", X_train.shape)
                            X_test = create_sequences(Test_normal_attack, time_steps[hh])
                            print("Reshaped testing data with time_steps: ", X_test.shape)
                            tf.keras.backend.clear_session()
                            history, model = fit_LSTM_AE_Bidirectional(X_train, n_epochs[ii], batch_size[jj], no_neurons[neurons], Drop_out[dropout], learning_rate[kk])

                            # plot_train(history)
                            threshold = compute_train_mae(model, X_train)
                            train_end = timeit.default_timer()
                            Train_elapsed_time = (train_end - train_start)
                            print("Training time for the iteration: ", Train_elapsed_time)
                            test_start = timeit.default_timer()
                            prediction(model, X_test, test_lables, threshold, time_steps[hh])
                            f1, precision, recall, TN, FN, TP, FP = calc_metrics_AE(prediction(model, X_test, test_lables, threshold, time_steps[hh]))
                            test_end = timeit.default_timer()
                            Test_elapsed_time = (test_end - test_start)
                            print("Testing time for the iteration: ", Test_elapsed_time, '% >>>>> Iteration done!')
                            print(f1, precision, recall, TN, FN, TP, FP)
                            output = [str(i), str(i), str(j), train_normal.shape[0], test_normal.shape[0],
                                      test_attack.shape[0], threshold, time_steps[hh], n_epochs[ii], batch_size[jj],
                                      no_neurons[neurons], Drop_out[dropout], learning_rate[kk], TN, FN, TP, FP, f1, precision, recall, Train_elapsed_time, Test_elapsed_time,ratio]
                            results = pd.DataFrame([output], columns=['Train user', 'Test user1', 'Test user2',
                                                                      'Train Normal size', 'Test Normal size',
                                                                      'Test attack size','threshold', 'time_steps','n_epochs', 'BZ', 'no_neurons', 'Drop_out','learning_rate',
                                                                      'TN', 'FN', 'TP', 'FP', 'F1-score', 'precision',
                                                                      'Recall','train_time (sec)', 'test_time (sec)', 'Ratio'])
                            save_metrics(results, filename, Results_path)










def run2(train_normal, test_normal, test_attack, ratio, gamma, nu_1, i, Results_path, filename):
    train_start = timeit.default_timer()
    Test_normal_attack = pd.concat([test_normal, test_attack], ignore_index=True)
    print("Test_normal_attack:", Test_normal_attack.shape)
    actual_labels = Test_normal_attack.iloc[:, -1:]
    # print(actual)
    print("Actual labels: ", len(actual_labels))
    Test_normal_attack = Test_normal_attack.iloc[:, :-1]  # drop last col [labels] from normal data
    print("Test_normal_attack:", Test_normal_attack.shape)
    # svm = OneClassSVM(kernel='rbf')
    svm = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu_1)
    print(svm)
    svm.fit(train_normal)
    train_end = timeit.default_timer()
    Train_elapsed_time = (train_end - train_start)
    # print("Training time for the iteration: ", Train_elapsed_time)

    test_start = timeit.default_timer()
    pred = svm.predict(Test_normal_attack)
    print("len(pred)", len(pred))
    # print("pred:  ", pred)
    pred = [0 if i == 1 else 1 for i in pred]
    print("len(pred)", len(pred))
    # print("pred:  ", pred)
    # normal_pred = pred[:len(test_normal)]
    # attack_pred = pred[len(test_normal)+1:]
    f1, precision, recall, TN, FN, TP, FP = calc_metrics(actual_labels, pred)
    test_end = timeit.default_timer()
    Test_elapsed_time = (test_end - test_start)
    # print("Testing time for the iteration: ", Test_elapsed_time, '% >>>>> Iteration done!')
    print(f1, precision, recall, TN, FN, TP, FP)
    output = [str(i), str(i), "All other users", train_normal.shape[0], test_normal.shape[0], test_attack.shape[0],
              TN, FN, TP, FP, f1, precision, recall, Train_elapsed_time, Test_elapsed_time, gamma, nu_1, ratio]
    results = pd.DataFrame([output],
                           columns=['Train owner', 'Test owner', 'Test attacks', 'Train Normal size', 'Test Normal size',
                                    'Test attack size',
                                    'TN', 'FN', 'TP', 'FP', 'F1-score', 'precision', 'Recall',
                                    'train_time (sec)', 'test_time (sec)', 'gamma', 'nu', 'Ratio'])
    save_metrics(results, filename, Results_path)

def mean_median_list(list_n):
    n = len(list_n)
    print()
    get_sum = sum(list_n)
    mean = get_sum / n
    list_n.sort()
    if n % 2 == 0:
        median1 = list_n[n // 2]
        median2 = list_n[n // 2 - 1]
        median = (median1 + median2) / 2
    else:
        median = list_n[n // 2]
    return mean, median


def mean_median_max_min_list(list_n):
    n = len(list_n)
    print()
    get_sum = sum(list_n)
    mean = get_sum / n
    max_value = max(list_n)
    min_value = min(list_n)
    list_n.sort()
    if n % 2 == 0:
        median1 = list_n[n // 2]
        median2 = list_n[n // 2 - 1]
        median = (median1 + median2) / 2
    else:
        median = list_n[n // 2]
    return mean, median, max_value, min_value


def compute_brute_force_result_1_OCSVM():
    source_path= r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\OCSVM\per_user_results\Belt_results_5_second'
    files_in_path = natsorted(os.listdir(source_path))  # list of users in the server
    print("Number of users in Server:", len(files_in_path))
    # print("List of users:", files_in_path)
    users = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    # users = ["0"]
    for n in range(len(users)):
        file_names = []
        for k in range(len(files_in_path)):
            import re
            file_name = re.findall(r'\d+', files_in_path[k])
            # file_name = files_in_path[k].split()
            # print(file_name[-1])
            if users[n] == file_name[-1]:
                print(file_name[-1])
                file_names.append(files_in_path[k])
        print(file_names)

        for i in range(len(file_names)):#len(files_in_path)
            path_one_file = os.path.join(source_path, file_names[i])  # build path for each user
            data = pd.read_csv(path_one_file)
            list_Train_user = list(data['Train user'])
            list_gamma = list(data['gamma'])
            list_nu = list(data['nu'])
            list_ratio = list(data['Ratio'])
            list_F1 = list(data['F1-score'])
            list_train_time = list(data['train_time (sec)'])
            list_test_time = list(data['test_time (sec)'])
            mean_Train_user, median_Train_user = mean_median_list(list_Train_user)
            if "scale" in list_gamma or "auto" in list_gamma:
                mean_gamma = list_gamma[0]
                median_gamma = list_gamma[0]
            else:
                mean_gamma, median_gamma = mean_median_list(list_gamma)
            mean_nu, median_nu = mean_median_list(list_nu)
            mean_ratio, median_ratio = mean_median_list(list_ratio)
            mean_F1, median_F1 =mean_median_list(list_F1)
            mean_F1, median_F1, max_f1, min_f1 = mean_median_max_min_list(list_F1)
            mean_train_time, median_train_time = mean_median_list(list_train_time)
            mean_test_time, median_test_time = mean_median_list(list_test_time)
            output = [mean_Train_user, mean_gamma, mean_nu, mean_ratio, max_f1, min_f1, mean_F1, mean_train_time, mean_test_time]

            results = pd.DataFrame([output], columns=['mean_Train_user', 'mean_gamma', 'mean_nu','mean_ratio', 'max_f1', 'min_f1','mean_F1', 'mean_train_time', 'mean_test_time'])
            # filename = "Brut_force_results_OCSVM_5_seconds_Seat_user" + str(n) +".csv"
            filename = "Brut_force_results_OCSVM_5_seconds_Belt_user" + str(n) + ".csv"
            # Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\OCSVM\average_results\fusion_average_results\fusion_average_2_seconds'
            Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\OCSVM\average_results2\Belt_average_results\Belt_average_5_seconds'
            save_metrics(results, filename, Results_path)


def compute_brute_force_result_1_LOF():
    source_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\LOF\per_user_results\Fusion_results_2_seconds'
    files_in_path = natsorted(os.listdir(source_path))  # list of users in the server
    print("Number of users in Server:", len(files_in_path))
    print("List of users:", files_in_path)
    users = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    for n in range(len(users)):
        file_names = []
        for k in range(len(files_in_path)):
            import re
            file_name = re.findall(r'\d+', files_in_path[k])
            # file_name = files_in_path[k].split()
            # print(file_name[-1])
            if users[n] == file_name[-1]:
                # print(file_name[-1])
                file_names.append(files_in_path[k])
        print(len(file_names))
        for i in range(len(file_names)):#len(files_in_path)
            path_one_file = os.path.join(source_path, file_names[i])  # build path for each user
            data = pd.read_csv(path_one_file)
            list_Train_user = list(data['Train user'])
            list_F1 = list(data['F1-score'])
            list_ratio = list(data['Ratio'])
            list_n_neighbors = list(data['n_neighbors'])
            list_train_time = list(data['train_time (sec)'])
            list_test_time = list(data['test_time (sec)'])
            mean_Train_user, median_Train_user = mean_median_list(list_Train_user)
            mean_ngh, median_gh = mean_median_list(list_n_neighbors)
            mean_ratio, median_ratio = mean_median_list(list_ratio)
            mean_F1, median_F1, max_f1, min_f1 = mean_median_max_min_list(list_F1)
            mean_train_time, median_train_time = mean_median_list(list_train_time)
            mean_test_time, median_test_time = mean_median_list(list_test_time)
            output = [mean_Train_user, mean_ngh, mean_ratio, max_f1, min_f1, mean_F1, mean_train_time, mean_test_time]
            results = pd.DataFrame([output],
                                   columns=['mean_Train_user', 'mean_ngh', 'mean_ratio', 'max_f1', 'min_f1', 'mean_F1','mean_train_time', 'mean_test_time'])
            filename = "Brut_force_results_LOF_2_seconds_Fusion_user" + str(n) + ".csv"
            Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\LOF\average_results\Fusion_average_results\Fusion_average_2_seconds'
            save_metrics(results, filename, Results_path)


def compute_brute_force_result_1_IF():
    source_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\IF\per_user_results\Belt_results_5_seconds'
    files_in_path = natsorted(os.listdir(source_path))  # list of users in the server
    print("Number of users in Server:", len(files_in_path))
    print("List of users:", files_in_path)
    users = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    for n in range(len(users)):
        file_names = []
        for k in range(len(files_in_path)):
            import re
            file_name = re.findall(r'\d+', files_in_path[k])
            # file_name = files_in_path[k].split()
            # print(file_name[-1])
            if users[n] == file_name[-1]:
                # print(file_name[-1])
                file_names.append(files_in_path[k])
        print(len(file_names))
        for i in range(len(file_names)):#len(files_in_path)
            path_one_file = os.path.join(source_path, file_names[i])  # build path for each user
            data = pd.read_csv(path_one_file)
            list_Train_user = list(data['Train user'])
            list_F1 = list(data['F1-score'])
            list_ratio = list(data['Ratio'])
            list_estimators = list(data['estimator'])
            list_train_time = list(data['train_time (sec)'])
            list_test_time = list(data['test_time (sec)'])
            mean_Train_user, median_Train_user = mean_median_list(list_Train_user)
            mean_estimators, median_estimators = mean_median_list(list_estimators)
            mean_ratio, median_ratio = mean_median_list(list_ratio)
            mean_F1, median_F1, max_f1, min_f1 = mean_median_max_min_list(list_F1)
            mean_train_time, median_train_time = mean_median_list(list_train_time)
            mean_test_time, median_test_time = mean_median_list(list_test_time)
            output = [mean_Train_user, mean_estimators, mean_ratio, max_f1, min_f1, mean_F1, mean_train_time, mean_test_time]
            results = pd.DataFrame([output],
                                   columns=['mean_Train_user', 'mean_estimators', 'mean_ratio', 'max_f1', 'min_f1', 'mean_F1','mean_train_time', 'mean_test_time'])
            filename = "Brut_force_results_IF_5_seconds_Belt_user" + str(n) + ".csv"
            Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\IF\average_results\Belt_average_results\Belt_average_5_seconds'
            save_metrics(results, filename, Results_path)


def compute_brute_force_result_1_EE():
    source_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\EE\per_user_results\Fusion_results_2_seconds'
    files_in_path = natsorted(os.listdir(source_path))  # list of users in the server
    print("Number of users in Server:", len(files_in_path))
    print("List of users:", files_in_path)
    users = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    for n in range(len(users)):
        file_names = []
        for k in range(len(files_in_path)):
            import re
            file_name = re.findall(r'\d+', files_in_path[k])
            # file_name = files_in_path[k].split()
            # print(file_name[-1])
            if users[n] == file_name[-1]:
                # print(file_name[-1])
                file_names.append(files_in_path[k])
        print(len(file_names))
        for i in range(len(file_names)):#len(files_in_path)
            path_one_file = os.path.join(source_path, file_names[i])  # build path for each user
            data = pd.read_csv(path_one_file)
            list_Train_user = list(data['Train user'])
            list_F1 = list(data['F1-score'])
            list_ratio = list(data['Ratio'])
            list_contaminations = list(data['contaminations'])
            list_train_time = list(data['train_time (sec)'])
            list_test_time = list(data['test_time (sec)'])
            mean_Train_user, median_Train_user = mean_median_list(list_Train_user)
            mean_contaminations, median_contaminations = mean_median_list(list_contaminations)
            mean_ratio, median_ratio = mean_median_list(list_ratio)
            mean_F1, median_F1, max_f1, min_f1 = mean_median_max_min_list(list_F1)
            mean_train_time, median_train_time = mean_median_list(list_train_time)
            mean_test_time, median_test_time = mean_median_list(list_test_time)
            output = [mean_Train_user, mean_contaminations, mean_ratio, max_f1, min_f1, mean_F1, mean_train_time, mean_test_time]
            results = pd.DataFrame([output],
                                   columns=['mean_Train_user', 'mean_contaminations', 'mean_ratio', 'max_f1', 'min_f1', 'mean_F1','mean_train_time', 'mean_test_time'])
            filename = "Brut_force_results_EE_2_seconds_Fusion_user" + str(n) + ".csv"
            Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\EE\average_results\Fusion_average_results\Fusion_average_2_seconds'
            save_metrics(results, filename, Results_path)

def compute_brute_force_result_1_AE():
    source_path= r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\AE\per_user_results\Belt_results_3_seconds'
    files_in_path = natsorted(os.listdir(source_path))  # list of users in the server
    print("Number of users in Server:", len(files_in_path))
    print("Number of users in Server:", files_in_path)
    all_paths = []
    for i in range(len(files_in_path)):  # len(files_in_path)
        path_one_file = os.path.join(source_path, files_in_path[i])  # build path for each user
        print(path_one_file)
        all_paths.append(path_one_file)
    # print("all_paths", all_paths)
    users = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    path1 = r"D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\AE\per_user_results\other2\Belt_3_seconds"
    name= "Belt_AE_result_ratio_0.9_Tr_user_"
    for ii in range(12):
        print("ii: ", ii)
        one_user_data = []
        for jj in range(11):
            # print((ii*11)+jj)
            path_one_file = os.path.join(source_path, all_paths[(ii*11)+jj])  # build path for each user
            # print(path_one_file)
            data = pd.read_csv(path_one_file)
            # print(data.shape)
            one_user_data.append(data)
            # print(len(one_user_data))
        data_df = pd.concat(one_user_data)
        path_one_user = os.path.join(path1, name + users[ii] + ".csv")
        data_df.to_csv(path_one_user)

def sort_average_results():
    source_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\OCSVM\average_results2\Belt_average_results\Belt_average_2_seconds'
    out_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\OCSVM\best_hyperparamters'
    # source_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\LOF\average_results\Fusion_average_results\Fusion_average_5_seconds'
    # out_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\LOF\best_hyperparamters'
    # source_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\IF\average_results\Fusion_average_results\Fusion_average_2_seconds'
    # out_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\IF\best_hyperparamters'
    # source_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\EE\average_results\Belt_average_results\Belt_average_5_seconds'
    # out_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\EE\best_hyperparamters'
    file_name = 'Best_Belt_2_seconds.csv'
    files_in_path = natsorted(os.listdir(source_path))  # list of users in the server
    print("Number of users in Server:", len(files_in_path))
    print("List of users:", files_in_path)
    # path_one_file = os.path.join(source_path, files_in_path[0])  # build path for each user
    # data = pd.read_csv(path_one_file)
    # print(data.shape)
    # data = data.sort_values('mean_F1', ascending=False)
    # print(data.iloc[0,:])
    best_f1_list = []
    for i in range(len(files_in_path)):
        path_one_file = os.path.join(source_path, files_in_path[i])  # build path for each user
        data = pd.read_csv(path_one_file)
        print(data.shape)
        data = data.sort_values('mean_F1', ascending=False)
        best_f1_list.append(data.iloc[0,:])
    best_f1_list_df = pd.concat(best_f1_list, axis=1, ignore_index=True)
    best_f1_list_df = best_f1_list_df.T
    path_output_file = os.path.join(out_path, file_name)
    best_f1_list_df.to_csv(path_output_file)







def compute_brute_force_result_2(): # for Scenario1_case2
    source_path= r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case2\per_user_results'
    files_in_path = natsorted(os.listdir(source_path))  # list of users in the server
    print("Number of users in Server:", len(files_in_path))
    users = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
    for n in range(len(users)):
        file_names = []
        for k in range(len(files_in_path)):
            import re
            file_name = re.findall(r'\d+', files_in_path[k])
            # file_name = files_in_path[k].split()
            print(file_name[-1])
            if users[n] == file_name[-1]:
                file_names.append(files_in_path[k])
        print(file_names)

        for i in range(len(file_names)):#len(files_in_path)
            path_one_file = os.path.join(source_path, file_names[i])  # build path for each user
            data = pd.read_csv(path_one_file)
            list_Train_user = list(data['Train owner'])
            list_gamma = list(data['gamma'])
            list_nu = list(data['nu'])
            list_ratio = list(data['Ratio'])
            list_F1 = list(data['F1-score'])
            list_train_time = list(data['train_time (sec)'])
            list_test_time = list(data['test_time (sec)'])
            mean_Train_user, median_Train_user = mean_median_list(list_Train_user)
            if "scale" in list_gamma or "auto" in list_gamma:
                mean_gamma = list_gamma[0]
                median_gamma = list_gamma[0]
            else:
                mean_gamma, median_gamma = mean_median_list(list_gamma)
            mean_nu, median_nu = mean_median_list(list_nu)
            mean_ratio, median_ratio = mean_median_list(list_ratio)
            mean_F1, median_F1 =mean_median_list(list_F1)
            mean_train_time, median_train_time = mean_median_list(list_train_time)
            mean_test_time, median_test_time = mean_median_list(list_test_time)
            output = [mean_Train_user, mean_gamma, mean_nu, mean_ratio, mean_F1, mean_train_time, mean_test_time]

            results = pd.DataFrame([output], columns=['mean_Train_user', 'mean_gamma', 'mean_nu','mean_ratio', 'mean_F1', 'mean_train_time', 'mean_test_time'])
            filename = "Brut_force_results_OCSVM_5_seconds_Seat_user" + str(n) +".csv"
            Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case2\average_results'
            save_metrics(results, filename, Results_path)

# def generate_n_grams(data_df, n ):
# 	from nltk.util import ngrams
# 	import itertools
# 	train_data_grams = []
# 	for a in data_df.values:
# 		l = list(ngrams(a, n=n))
# 		l2 = list(itertools.chain(*l))
# 		train_data_grams.append(l2)
# 	train_data_grams_df = pd.DataFrame(train_data_grams)
# 	return train_data_grams_df

# data normalization and preprossing
def data_preprocessing(train_data, test_data, pca=False, bigrams=False, trigrams=False):
	"""
	:param train_data:
	:param test_data:
	:return: normalized train data and test data
	"""


	# ##################### MinMaxScaler ##############################
	from sklearn.preprocessing import StandardScaler, MinMaxScaler
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler = scaler.fit(train_data)
	train_data = pd.DataFrame(scaler.transform(train_data))
	scaler = scaler.fit(test_data)
	test_data = pd.DataFrame(scaler.transform(test_data))
	###################################################


	# bigram
	if bigrams:
		# for bigrams
		train_data = generate_n_grams(train_data, 2)
		test_data = generate_n_grams(test_data, 2)

	# trigram
	if trigrams and not bigrams:
		train_data = generate_n_grams(train_data, 3)
		test_data = generate_n_grams(test_data, 3)

	# PCA
	if pca:
		pca = sk.decomposition.PCA(n_components=0.95)
		train_data = pca.fit_transform(train_data)
		test_data = pca.transform(test_data)

	return train_data, test_data

