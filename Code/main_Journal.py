import os
import re

import numpy as np
import sklearn as sk
# import sklearn.preprocessing as prep
# import sklearn.model_selection as model_selection
# import sklearn.metrics as metrics
# import sklearn.ensemble as ensemble
import pandas as pd
import sklearn.ensemble as ensemble

# import math
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from numpy import mean
# from numpy import std
# import pandas
# from keras.models import Sequential
# from keras.layers import Dense
# # from keras.wrappers.scikit_learn import KerasClassifier
# # from keras.utils import np_utils
# from scikeras.wrappers import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from helper_all import *
import matplotlib.pyplot as plt
import timeit
# from drift_code import *
##################################################################################################################

def classifier(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    train_start = timeit.default_timer()
    logreg = LogisticRegression(random_state=12).fit(X_train, y_train)
    train_end = timeit.default_timer()
    Train_elapsed_time = (train_end - train_start)
    print("Training time for the iteration: ", Train_elapsed_time)
    test_start = timeit.default_timer()
    y_pred = logreg.predict(X_test)
    print(logreg.score(X_train, y_train)*100)
    print(logreg.score(X_test, y_test)*100)
    print(y_pred)
    from sklearn import metrics
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)
    test_end = timeit.default_timer()
    Test_elapsed_time = (test_end - test_start)
    print("Testing time for the iteration: ", Test_elapsed_time, '% >>>>> Iteration done!')

    # import seaborn as sns
    # plt.figure(figsize=(9, 9))
    # sns.heatmap(cm, annot=True, fmt='0.0f', linewidths=0.5, square=True, cbar=False)
    # plt.ylabel('Actual users',  fontsize=22)
    # plt.xlabel('Predicted users',  fontsize=22)
    # plt.title('Belt-based data with LR model (Tr:0.9, Ts:0.1)',  fontsize=22)
    # # plt.title('Seat-based data with LR model (Tr:0.6, Ts:0.4)',  fontsize=22)
    # plt.show()
    print(metrics.classification_report(y_test, y_pred))

def classifier_2 (X_train, X_test, y_train, y_test):
    models = [
        ensemble.RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=0),
    ]
    names = ['RandomForest']  # 'GradientBoosting' consumes large train time
    for name, model in zip(names, models):
        train_start = timeit.default_timer()
        model.fit(X_train, y_train)
        train_end = timeit.default_timer()
        Train_elapsed_time = (train_end - train_start)
        print("Training time for the iteration: ", Train_elapsed_time)
        test_start = timeit.default_timer()
        y_pred = model.predict(X_test)
        print(model.score(X_train, y_train) * 100)
        print(model.score(X_test, y_test) * 100)
        print(y_pred)
        from sklearn import metrics
        cm = metrics.confusion_matrix(y_test, y_pred)
        print(cm)
        test_end = timeit.default_timer()
        Test_elapsed_time = (test_end - test_start)
        print("Testing time for the iteration: ", Test_elapsed_time, '% >>>>> Iteration done!')
        # import seaborn as sns
        # plt.figure(figsize=(9, 9))
        # sns.heatmap(cm, annot=True, fmt='0.0f', linewidths=0.5, square=True, cbar=False, annot_kws={"fontsize":15})
        # plt.ylabel('Actual users', fontsize=28)
        # plt.xlabel('Predicted users', fontsize=28)
        # # plt.title('Belt-based data with RF model (Tr:0.6, Ts:0.4)', fontsize=22)
        # # plt.title('Seat-based data with RF model (Tr:0.6, Ts:0.4)', fontsize=22)
        # plt.xticks(fontsize=16)
        # plt.yticks(fontsize=16)
        # plt.show()
        print(metrics.classification_report(y_test, y_pred))

def scenario1_case1_OCSVM(data_all_users):# take one user as owner to train the model, and test agnist each other user as attacker data (one vs one)
    for i in range(len(data_all_users)):  # len(data_all_users)
        for j in range(len(data_all_users)):  # len(data_all_users)
            if j != i:# take one user as owner to train the model, and test agnist each other user as attacker data
                normal_data = data_all_users[i]
                attack_data = data_all_users[j]

                # OCSVM model
                # split_ratio = [0.5, 0.6, 0.7, 0.8, 0.9]
                # gamma = ['scale', 'auto', 0.001, 0.01, 0.1]
                # nu = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7]
                split_ratio = [0.9]
                gamma = ['scale']
                nu = [0.7]
                for k in range(len(split_ratio)):
                    for g in range(len(gamma)):
                        for n in range(len(nu)):
                            X_tr_N, X_ts_N, X_ts_A = Data_process2(normal_data, attack_data, split_ratio[k])
                            print("X_tr_N:  ", X_tr_N.shape)
                            # print("X_tr_N:  ", X_tr_N)
                            print("X_ts_N:  ", X_ts_N.shape)
                            # print("X_ts_N:  ", X_ts_N)
                            print("X_ts_A:  ", X_ts_A.shape)
                            # print("X_ts_A:  ", X_ts_A)
                            # model_output_path = r'D:\Sejong_Univ_03_01_2025\research\Driver_authentication\results\Journal_results\Models_files\OCSVM\Belt_models\SizeD'
                            # filename_model = 'Belt_OCSVM_model_ratio_' + str(split_ratio[k]) + '_gamma_' + str(gamma[g]) + '_nu_' + str(nu[n]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.pkl'

                            # model_output_path = r'D:\Sejong_Univ_03_01_2025\research\Driver_authentication\results\Journal_results\Models_files\OCSVM\Seat_models\SizeD'
                            # filename_model = 'Seat_OCSVM_model_ratio_' + str(split_ratio[k]) + '_gamma_' + str(gamma[g]) + '_nu_' + str(nu[n]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.pkl'

                            model_output_path = r'D:\Sejong_Univ_03_01_2025\research\Driver_authentication\results\Journal_results\Models_files\OCSVM\Fusion_models\SizeD'
                            filename_model = 'Fusion_OCSVM_model_ratio_' + str(split_ratio[k]) + '_gamma_' + str(gamma[g]) + '_nu_' + str(nu[n]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.pkl'


                            # Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\per_user_results\Seat_results_2_second'
                            # Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\per_user_results\Belt_results_2_seconds'
                            # Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\per_user_results\Fusion_results_2_second'
                            # Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\using_best_hyperprameters\Seat_results_2_second'
                            # Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\using_best_hyperprameters\Belt_results_2_second'
                            # Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\using_best_hyperprameters\Fusion_results_2_second'

                            # filename = 'Seat_OCSVM_result_' + str(split_ratio[k]) + '_' + str(gammas[g]) + '_' + str(nu[n]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.csv'
                            # filename = 'Fusion_OCSVM_result_ratio_' + str(split_ratio[k]) + '_gamma_' + str(gamma[g]) + '_nu_' + str(nu[n]) + '_Tr_user' + str(i) + '.csv'
                            # filename = 'Belt_OCSVM_result_ratio_' + str(split_ratio[k]) + '_gamma_' + str(gamma[g]) + '_nu_' + str(nu[n]) + '_Tr_user' + str(i) + '.csv'

                            # run_OCSVM(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], gamma[g], nu[n], i, j, model_output_path, filename_model, Results_path, filename)
                            run_OCSVM(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], gamma[g], nu[n], i, j, model_output_path, filename_model)





def scenario1_case1_IF(data_all_users):  # take one user as owner to train the model, and test agnist each other user as attacker data (one vs one)
    for i in range(len(data_all_users)):  # len(data_all_users)
        for j in range(len(data_all_users)):  # len(data_all_users)
            if j != i:  # take one user as owner to train the model, and test agnist each other user as attacker data
                normal_data = data_all_users[i]
                attack_data = data_all_users[j]

                # IF model
                # split_ratio = [0.5, 0.6, 0.7, 0.8, 0.9]
                # estimators= [10, 20, 50, 100, 500, 1000, 10000]
                # warm_start=['True', 'False']
                split_ratio = [0.9]
                estimators = [1000]
                for k in range(len(split_ratio)):
                    for g in range(len(estimators)):
                        X_tr_N, X_ts_N, X_ts_A = Data_process2(normal_data, attack_data, split_ratio[k])
                        print("X_tr_N:  ", X_tr_N.shape)
                        # print("X_tr_N:  ", X_tr_N)
                        print("X_ts_N:  ", X_ts_N.shape)
                        # print("X_ts_N:  ", X_ts_N)
                        print("X_ts_A:  ", X_ts_A.shape)
                        # print("X_ts_A:  ", X_ts_A)
                        model_output_path = r'D:\Sejong_Univ_03_01_2025\research\Driver_authentication\results\Journal_results\Models_files\IF\Fusion_models\SizeD'
                        filename_model = 'Fusion_IF_model_ratio_' + str(split_ratio[k]) + '_estimators_' + str(estimators[g]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.pkl'
                        # Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\IF\per_user_results\Fusion_results_2_seconds'
                        # filename = 'Fusion_IF_result_ratio_' + str(split_ratio[k]) + '_estimators_' + str(estimators[g]) + '_Tr_user' + str(i) + '.csv'
                        # run_IF(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], estimators[g], i, j, model_output_path, filename_model, Results_path, filename)
                        run_IF(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], estimators[g], i, j, model_output_path, filename_model)

def scenario1_case1_LOF(data_all_users):  # take one user as owner to train the model, and test agnist each other user as attacker data (one vs one)
    for i in range(len(data_all_users)):  # len(data_all_users)
        for j in range(len(data_all_users)):  # len(data_all_users)
            if j != i:  # take one user as owner to train the model, and test agnist each other user as attacker data
                normal_data = data_all_users[i]
                attack_data = data_all_users[j]

                # LOF model
                # split_ratio = [0.5, 0.6, 0.7, 0.8, 0.9]
                # neighbors = [10, 20, 50, 100, 1000, 10000]
                split_ratio = [0.9]
                neighbors = [1000]
                for k in range(len(split_ratio)):
                    for g in range(len(neighbors)):
                        X_tr_N, X_ts_N, X_ts_A = Data_process2(normal_data, attack_data, split_ratio[k])
                        print("X_tr_N:  ", X_tr_N.shape)
                        # print("X_tr_N:  ", X_tr_N)
                        print("X_ts_N:  ", X_ts_N.shape)
                        # print("X_ts_N:  ", X_ts_N)
                        print("X_ts_A:  ", X_ts_A.shape)
                        # print("X_ts_A:  ", X_ts_A)
                        model_output_path = r'D:\Sejong_Univ_03_01_2025\research\Driver_authentication\results\Journal_results\Models_files\LOF\Fusion_models\SizeD'
                        filename_model = 'Fusion_LOF_model_ratio_' + str(split_ratio[k]) + '_neighbors_' + str(neighbors[g]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.pkl'
                        # Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\LOF\per_user_results\Fusion_results_2_seconds'

                        # filename = 'Fusion_LOF_result_ratio_' + str(split_ratio[k]) + '_neighbors_' + str(neighbors[g]) + '_Tr_user' + str(i) + '.csv'
                        # run_LOF(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], neighbors[g], i, j, model_output_path, filename_model, Results_path, filename)
                        run_LOF(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], neighbors[g], i, j, model_output_path, filename_model)



def scenario1_case1_EE(data_all_users):  # take one user as owner to train the model, and test agnist each other user as attacker data (one vs one)
    for i in range(len(data_all_users)):  # len(data_all_users)
        for j in range(len(data_all_users)):  # len(data_all_users)
            if j != i:  # take one user as owner to train the model, and test agnist each other user as attacker data
                normal_data = data_all_users[i]
                attack_data = data_all_users[j]

                # EE model
                # split_ratio = [0.5, 0.6, 0.7, 0.8, 0.9]
                # contaminations = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
                split_ratio = [0.9]
                contaminations = [0.0001]
                for k in range(len(split_ratio)):
                    for g in range(len(contaminations)):
                        X_tr_N, X_ts_N, X_ts_A = Data_process2(normal_data, attack_data, split_ratio[k])
                        print("X_tr_N:  ", X_tr_N.shape)
                        # print("X_tr_N:  ", X_tr_N)
                        print("X_ts_N:  ", X_ts_N.shape)
                        # print("X_ts_N:  ", X_ts_N)
                        print("X_ts_A:  ", X_ts_A.shape)
                        # print("X_ts_A:  ", X_ts_A)
                        model_output_path = r'D:\Sejong_Univ_03_01_2025\research\Driver_authentication\results\Journal_results\Models_files\EE\Fusion_models\SizeD'
                        filename_model = 'Fusion_EE_model_ratio_' + str(split_ratio[k]) + '_contaminations_' + str(contaminations[g]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.pkl'

                        # Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\EE\per_user_results\Fusion_results_2_seconds'
                        # filename = 'Fusion_EE_result_ratio_' + str(split_ratio[k]) + '_contaminations_' + str(contaminations[g]) + '_Tr_user' + str(i) + '.csv'
                        # run_EE(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], contaminations[g], i, j, model_output_path, filename_model, Results_path, filename)
                        run_EE(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], contaminations[g], i, j, model_output_path, filename_model)


def scenario1_case1_AE(data_all_users):  # take one user as owner to train the model, and test agnist each other user as attacker data (one vs one)
    for i in range(len(data_all_users)):  # len(data_all_users)
        for j in range(len(data_all_users)):  # len(data_all_users)
            if j != i:  # take one user as owner to train the model, and test agnist each other user as attacker data
                normal_data = data_all_users[i]
                attack_data = data_all_users[j]

                # IF model
                # split_ratio = [0.5, 0.6, 0.7, 0.8, 0.9]
                split_ratio = [0.9]
                for k in range(len(split_ratio)):
                    X_tr_N, X_ts_N, X_ts_A = Data_process2(normal_data, attack_data, split_ratio[k])
                    print("X_tr_N:  ", X_tr_N.shape)
                    # print("X_tr_N:  ", X_tr_N)
                    print("X_ts_N:  ", X_ts_N.shape)
                    # print("X_ts_N:  ", X_ts_N)
                    print("X_ts_A:  ", X_ts_A.shape)
                    # print("X_ts_A:  ", X_ts_A)
                    # model_output_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Models_files\EE\Belt_models'
                    # filename_model = 'Belt_EE_model_ratio_' + str(split_ratio[k]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.pkl'
                    Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case1\AE\per_user_results\Fusion_results_2_seconds'

                    filename = 'Fusion_AE_result_ratio_' + str(split_ratio[k]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.csv'
                    run_AE(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], i, j, Results_path, filename)


# def scenario2_drift_OCSVM(data_all_users, data_all_users_new):
#     for i in range(len(data_all_users)):  # len(data_all_users)
#         normal_data = data_all_users[i]
#         normal_data_new = data_all_users_new[i]
#         for j in range(len(data_all_users_new)):  # len(data_all_users)
#             if j != i:# take one user as owner to train the model, and test agnist each other user as attacker data
#                 attack_data = data_all_users_new[j]
#                 split_ratio = [0.9]
#                 gamma = ['scale']
#                 nu = [0.1]
#                 for k in range(len(split_ratio)):
#                     for g in range(len(gamma)):
#                         for n in range(len(nu)):
#                             X_tr_N, X_ts_N, X_ts_A = Tr_Ts_drfit_split_process(normal_data, normal_data_new, attack_data, split_ratio[k])
#                             print("X_tr_N:  ", X_tr_N.shape)
#                             # print("X_tr_N:  ", X_tr_N)
#                             print("X_ts_N:  ", X_ts_N.shape)
#                             # print("X_ts_N:  ", X_ts_N)
#                             print("X_ts_A:  ", X_ts_A.shape)
#                             # print("X_ts_A:  ", X_ts_A)
#                             Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario2_drift\OCSVM\using_best_hyperprameters\Fusion_results_2_second'
#                             model_output_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Models_files'
#                             filename_model = 'Seat_OCSVM_model_ratio_' + str(split_ratio[k]) + '_gamma_' + str(gamma[g]) + '_nu_' + str(nu[n]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.pkl'
#                             filename = 'Fusion_OCSVM_drift_result_ratio_' + str(split_ratio[k]) + '_gamma_' + str(gamma[g]) + '_nu_' + str(nu[n]) + '_Tr_user' + str(i) + '.csv'
#                             run_OCSVM(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], gamma[g], nu[n], i, j, model_output_path,filename_model, Results_path, filename)


# def scenario2_drift_IF(data_all_users, data_all_users_new):
#     for i in range(len(data_all_users)):  # len(data_all_users)
#         normal_data = data_all_users[i]
#         normal_data_new = data_all_users_new[i]
#         for j in range(len(data_all_users_new)):  # len(data_all_users)
#             if j != i:# take one user as owner to train the model, and test agnist each other user as attacker data
#                 attack_data = data_all_users_new[j]
#                 split_ratio = [0.9]
#                 estimators = [1000]
#                 for k in range(len(split_ratio)):
#                     for g in range(len(estimators)):
#                         X_tr_N, X_ts_N, X_ts_A = Tr_Ts_drfit_split_process(normal_data, normal_data_new, attack_data, split_ratio[k])
#                         print("X_tr_N:  ", X_tr_N.shape)
#                         # print("X_tr_N:  ", X_tr_N)
#                         print("X_ts_N:  ", X_ts_N.shape)
#                         # print("X_ts_N:  ", X_ts_N)
#                         print("X_ts_A:  ", X_ts_A.shape)
#                         # print("X_ts_A:  ", X_ts_A)
#                         Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario2_drift\IF\using_best_hyperprameters\Fusion_results_2_second'
#                         model_output_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Models_files\IF\Belt_models'
#                         filename_model = 'Fusion_IF_model_ratio_' + str(split_ratio[k]) + '_estimators_' + str(estimators[g]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.pkl'
#                         filename = 'Fusion_IF_drift_result_ratio_' + str(split_ratio[k]) + '_estimators_' + str(estimators[g]) + '_Tr_user' + str(i) + '.csv'
#                         run_IF(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], estimators[g], i, j, model_output_path, filename_model, Results_path, filename)


# def scenario2_drift_LOF(data_all_users, data_all_users_new):
#     for i in range(len(data_all_users)):  # len(data_all_users)
#         normal_data = data_all_users[i]
#         normal_data_new = data_all_users_new[i]
#         for j in range(len(data_all_users_new)):  # len(data_all_users)
#             if j != i:# take one user as owner to train the model, and test agnist each other user as attacker data
#                 attack_data = data_all_users_new[j]
#                 split_ratio = [0.9]
#                 neighbors = [1000]
#                 for k in range(len(split_ratio)):
#                     for g in range(len(neighbors)):
#                         X_tr_N, X_ts_N, X_ts_A = Tr_Ts_drfit_split_process(normal_data, normal_data_new, attack_data, split_ratio[k])
#                         print("X_tr_N:  ", X_tr_N.shape)
#                         # print("X_tr_N:  ", X_tr_N)
#                         print("X_ts_N:  ", X_ts_N.shape)
#                         # print("X_ts_N:  ", X_ts_N)
#                         print("X_ts_A:  ", X_ts_A.shape)
#                         # print("X_ts_A:  ", X_ts_A)
#                         Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario2_drift\LOF\using_best_hyperprameters\Belt_results_5_second'
#                         model_output_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Models_files\IF\Belt_models'
#                         filename_model = 'Fusion_LOF_model_ratio_' + str(split_ratio[k]) + '_neighbors_' + str(neighbors[g]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.pkl'
#                         filename = 'Belt_LOF_drift_result_ratio_' + str(split_ratio[k]) + '_neighbors_' + str(neighbors[g]) + '_Tr_user' + str(i) + '.csv'
#                         run_LOF(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], neighbors[g], i, j, model_output_path, filename_model, Results_path, filename)


# def scenario2_drift_EE(data_all_users, data_all_users_new):
#     for i in range(len(data_all_users)):  # len(data_all_users)
#         normal_data = data_all_users[i]
#         normal_data_new = data_all_users_new[i]
#         for j in range(len(data_all_users_new)):  # len(data_all_users)
#             if j != i:# take one user as owner to train the model, and test agnist each other user as attacker data
#                 attack_data = data_all_users_new[j]
#                 split_ratio = [0.9]
#                 contaminations = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
#                 for k in range(len(split_ratio)):
#                     for g in range(len(contaminations)):
#                         X_tr_N, X_ts_N, X_ts_A = Tr_Ts_drfit_split_process(normal_data, normal_data_new, attack_data, split_ratio[k])
#                         print("X_tr_N:  ", X_tr_N.shape)
#                         # print("X_tr_N:  ", X_tr_N)
#                         print("X_ts_N:  ", X_ts_N.shape)
#                         # print("X_ts_N:  ", X_ts_N)
#                         print("X_ts_A:  ", X_ts_A.shape)
#                         # print("X_ts_A:  ", X_ts_A)
#                         Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario2_drift\EE\using_best_hyperprameters\Belt_results_5_second'
#                         model_output_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Models_files\IF\Belt_models'
#                         filename_model = 'Belt_EE_model_ratio_' + str(split_ratio[k]) + '_contaminations_' + str(contaminations[g]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.pkl'
#                         filename = 'Belt_EE_drift_result_ratio_' + str(split_ratio[k]) + '_contaminations_' + str(contaminations[g]) + '_Tr_user' + str(i) + '.csv'
#                         run_LOF(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], contaminations[g], i, j, model_output_path, filename_model, Results_path, filename)






def start():
    # paths_belt = [r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U1\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U2\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U3\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U4\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U5\Belt",
    #               # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U6\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U7\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U8\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U9\Belt",
    #               # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U10\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U11\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U12\Belt"
    #               ]


    # paths_belt = [r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U1\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U2\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U3\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U4\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U5\Belt",
    #               # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U6\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U7\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U8\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U9\Belt",
    #               # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U10\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U11\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U12\Belt"
    #               ]

    # paths_belt = [r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U1\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U2\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U3\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U4\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U5\Belt",
    #               # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U6\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U7\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U8\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U9\Belt",
    #               # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U10\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U11\Belt",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U12\Belt"
    #               ]

    paths_belt = [r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U1\Belt",
                  r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U2\Belt",
                  r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U3\Belt",
                  r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U4\Belt",
                  r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U5\Belt",
                  # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U6\Belt",
                  r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U7\Belt",
                  r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U8\Belt",
                  r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U9\Belt",
                  # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U10\Belt",
                  r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U11\Belt",
                  r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U12\Belt"
                  ]


    # paths_seat = [
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U1\Seat",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U2\Seat",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U3\Seat",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U4\Seat",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U5\Seat",
    #               # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U6\Seat",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U7\Seat",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U8\Seat",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U9\Seat",
    #               # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U10\Seat",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U11\Seat",
    #               r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\5_seconds\U12\Seat"
    #               ]

    # paths_seat = [
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U1\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U2\Seat",
    #     # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U3\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U4\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U5\Seat",
    #     # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U6\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U7\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U8\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U9\Seat",
    #     # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U10\Seat",
    #     # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U11\Seat",
    #     # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\4_seconds\U12\Seat"
    # ]


    # paths_seat = [
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U1\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U2\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U3\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U4\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U5\Seat",
    #     # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U6\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U7\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U8\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U9\Seat",
    #     # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U10\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U11\Seat",
    #     r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\3_seconds\U12\Seat"
    # ]

    paths_seat = [
        r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U1\Seat",
        r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U2\Seat",
        r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U3\Seat",
        r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U4\Seat",
        r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U5\Seat",
        # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U6\Seat",
        r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U7\Seat",
        r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U8\Seat",
        r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U9\Seat",
        # r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U10\Seat",
        r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U11\Seat",
        r"D:\Sejong_Univ_03_01_2025\research\Driver_authentication\datasets\2_seconds\U12\Seat"
        ]




    # paths_belt_new = [r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U1-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U2-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U3-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U4-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U5-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U7-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U8-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U9-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U11-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U12-1\Belt"
    #                   ]


    # paths_belt_new = [r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U1-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U2-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U3-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U4-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U5-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U7-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U8-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U9-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U11-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U12-1\Belt"
    #                   ]


    # paths_belt_new = [
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U1-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U2-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U3-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U4-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U5-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U7-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U8-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U9-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U11-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U12-1\Belt"
    #                   ]

    # paths_belt_new = [r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U1-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U2-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U3-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U4-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U5-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U7-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U8-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U9-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U11-1\Belt",
    #                   r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U12-1\Belt"
    #                   ]

    # paths_seat_new = [
    #               r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U1-1\Seat",
    #               r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U2-1\Seat",
    #               r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U3-1\Seat",
    #               r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U4-1\Seat",
    #               r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U5-1\Seat",
    #               r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U7-1\Seat",
    #               r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U8-1\Seat",
    #               r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U9-1\Seat",
    #               r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U11-1\Seat",
    #               r"D:\Sejong_Univ\research\Driver_authentication\datasets\5_seconds\U12-1\Seat"
    #               ]

    # paths_seat_new = [
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U1-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U2-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U3-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U4-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U5-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U7-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U8-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U9-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U11-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\4_seconds\U12-1\Seat"
    # ]

    # paths_seat_new = [
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U1-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U2-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U3-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U4-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U5-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U7-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U8-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U9-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U11-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\3_seconds\U12-1\Seat"
    # ]


    # paths_seat_new = [
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U1-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U2-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U3-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U4-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U5-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U7-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U8-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U9-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U11-1\Seat",
    #     r"D:\Sejong_Univ\research\Driver_authentication\datasets\2_seconds\U12-1\Seat"
    # ]

    ########### BELT DATA #########################
    belt_data_all_users = []
    for i in range(len(paths_belt)):
        one_user_df = load_Belt_data(paths_belt[i])
        one_user_df[len(one_user_df.columns)] = i + 1  # Add index as labels to all record_df rows
        # print(one_user_df.shape)
        belt_data_all_users.append(one_user_df)
    print('belt_data_all_users', len(belt_data_all_users))
    ########## SEAT DATA #########################
    seat_data_all_users = []
    for i in range(len(paths_seat)):
        one_user_df = load_Seat_data(paths_seat[i])
        # one_user_df[len(one_user_df.columns)] = i + 1  # Add index as labels to all record_df rows
        print(one_user_df.shape)
        seat_data_all_users.append(one_user_df)
    print('seat_data_all_users' , len(seat_data_all_users))
    ############# FUSION DATA #####################################
    fusion_data_all_users = []
    for i in range(len(seat_data_all_users)):#
        one_user_seat_df = seat_data_all_users[i]
        print(one_user_seat_df.shape)
        one_user_seat_df.columns = [''] * len(one_user_seat_df.columns)
        one_user_belt_df = belt_data_all_users[i]
        print(one_user_belt_df.shape)
        one_user_belt_df.columns = [''] * len(one_user_belt_df.columns)
        one_user_fusion_df = pd.concat([one_user_seat_df, one_user_belt_df], axis=0, ignore_index=True) #concatnate verstically both seat and belt data
        print(one_user_fusion_df.shape)
        # one_user_fusion_df.to_csv("one_user_fusion_df.csv")
        fusion_data_all_users.append(one_user_fusion_df)
    print('fusion_data_all_users', len(fusion_data_all_users))
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    ########### BELT DATA NEW #########################
    # belt_data_all_users_NEW = []
    # for i in range(len(paths_belt_new)):
    #     one_user_df_NEW = load_Belt_data(paths_belt_new[i])
    #     one_user_df_NEW[len(one_user_df_NEW.columns)] = i + 1  # Add index as labels to all record_df rows
    #     # print(one_user_df_NEW.shape)
    #     belt_data_all_users_NEW.append(one_user_df_NEW)
    # print('belt_data_all_users_NEW', len(belt_data_all_users_NEW))
    ########### SEAT DATA NEW  #########################
    # seat_data_all_users_NEW = []
    # for i in range(len(paths_seat_new)):
    #     one_user_df_NEW = load_Seat_data(paths_seat_new[i])
    #     # one_user_df_NEW[len(one_user_df_NEW.columns)] = i + 1  # Add index as labels to all record_df rows
    #     print("User number : >>>>> ", i, ">>>>", one_user_df_NEW.shape)
    #     seat_data_all_users_NEW.append(one_user_df_NEW)
    # print('seat_data_all_users_NEW' , len(seat_data_all_users_NEW))

    ############# FUSION DATA NEW #####################################
    # fusion_data_all_users_NEW = []
    # for i in range(len(seat_data_all_users_NEW)):#
    #     one_user_seat_df_new = seat_data_all_users_NEW[i]
    #     print(one_user_seat_df_new.shape)
    #     one_user_seat_df_new.columns = [''] * len(one_user_seat_df_new.columns)
    #     one_user_belt_df_new = belt_data_all_users_NEW[i]
    #     print(one_user_belt_df_new.shape)
    #     one_user_belt_df_new.columns = [''] * len(one_user_belt_df_new.columns)
    #     one_user_fusion_df_new = pd.concat([one_user_seat_df_new, one_user_belt_df_new], axis=0, ignore_index=True) #concatnate verstically both seat and belt data
    #     print(one_user_fusion_df_new.shape)
    #     # one_user_fusion_df.to_csv("one_user_fusion_df.csv")
    #     fusion_data_all_users_NEW.append(one_user_fusion_df_new)
    # print('fusion_data_all_users_NEW', len(fusion_data_all_users_NEW))
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # scenario1_case1_OCSVM(belt_data_all_users)
    # scenario1_case1_OCSVM(seat_data_all_users)
    # scenario1_case1_OCSVM(fusion_data_all_users)

    # scenario1_case1_IF(belt_data_all_users)
    # scenario1_case1_IF(seat_data_all_users)
    # scenario1_case1_IF(fusion_data_all_users)

    # scenario1_case1_LOF(belt_data_all_users)
    # scenario1_case1_LOF(seat_data_all_users)
    # scenario1_case1_LOF(fusion_data_all_users)

    # scenario1_case1_EE(belt_data_all_users)
    # scenario1_case1_EE(seat_data_all_users)
    scenario1_case1_EE(fusion_data_all_users)


    # scenario1_case1_AE(belt_data_all_users)
    # scenario1_case1_AE(seat_data_all_users)
    # scenario1_case1_AE(fusion_data_all_users)

    # scenario2_drift_OCSVM(belt_data_all_users, belt_data_all_users_NEW)
    # scenario2_drift_OCSVM(seat_data_all_users, seat_data_all_users_NEW)
    # scenario2_drift_OCSVM(fusion_data_all_users, fusion_data_all_users_NEW)
    # scenario2_drift_IF(belt_data_all_users, belt_data_all_users_NEW)
    # scenario2_drift_IF(seat_data_all_users, seat_data_all_users_NEW)
    # scenario2_drift_IF(fusion_data_all_users, fusion_data_all_users_NEW)
    # scenario2_drift_LOF(belt_data_all_users, belt_data_all_users_NEW)
    # scenario2_drift_LOF(seat_data_all_users, seat_data_all_users_NEW)
    # scenario2_drift_LOF(fusion_data_all_users, fusion_data_all_users_NEW)
    # scenario2_drift_EE(belt_data_all_users, belt_data_all_users_NEW)


    # compute_brute_force_result_1_OCSVM() # for scenario1_case1
    # compute_brute_force_result_1_LOF() # for scenario1_case1
    # compute_brute_force_result_1_IF()# for scenario1_case1
    # compute_brute_force_result_1_EE() # for scenario1_case1
    # compute_brute_force_result_1_AE()# for scenario1_case1

    # sort_average_results()


    # scenario1_case2(data_all_users)
    # compute_brute_force_result_2()# for scenario1_case2

    #
    # print("X_tr: ", X_tr.shape)
    # print("X_ts: ", X_ts.shape)
    # print("y_tr: ", y_tr.shape)
    # print("y_ts: ", y_ts.shape)
    #
    #
    # X_tr1, X_ts1 = data_preprocessing(X_tr, X_ts)
    # # classifier(X_tr1, X_ts1, y_tr, y_ts)
    # classifier_2(X_tr1, X_ts1, y_tr, y_ts)






if __name__ == '__main__':
    start()



# def scenario1_case2(data_all_users): # take one user as owner to train the model, and test using data of all other user as attacker data (one vs all)
#     for i in range(len(data_all_users)):# owner user
#         split_ratio = [0.5, 0.6, 0.7, 0.8, 0.9]
#         for k in range(len(split_ratio)):
#             normal_data = data_all_users[i]
#             # print("normal_data: ", normal_data.shape)
#             # splitting dataframe by row index
#             n_train = int(normal_data.shape[0] * split_ratio[k])
#             train_normal = normal_data.iloc[:n_train, :]
#             # print("Train_normal shape: ", train_normal.shape)
#             test_normal = normal_data.iloc[n_train + 1:, :]
#             # print("test_normal shape: ", test_normal.shape)
#             test_attack_data= []
#             for j in range(len(data_all_users)):# attackers
#                 if j != i:# take one user as owner to train the model, and test agnist each other user as attacker data
#                     attack_data_user = data_all_users[j]
#                     n_test = int(len(attack_data_user) * ((1 - split_ratio[k])/(len(data_all_users)-1)))
#                     test_attack_user = attack_data_user.iloc[:n_test, :]
#                     # print("attack_data_one_user: ", test_attack_user.shape)
#                     test_attack_data.append(test_attack_user)
#             test_attack = pd.concat(test_attack_data)
#             # print("test_attack: ", test_attack.shape)
#
#             X_tr_N, X_ts_N, X_ts_A = Data_process3(train_normal, test_normal, test_attack)
#             print("X_tr_N:  ", X_tr_N.shape)
#             print("X_ts_N:  ", X_ts_N.shape)
#             print("X_ts_A:  ", X_ts_A.shape)
#             gamma = ['scale', 'auto', 0.001, 0.01, 0.1]
#             nu = [0.001, 0.01, 0.1, 0.3, 0.5, 0.7]
#             for  g in range (len(gamma)):
#                 for n in range (len(nu)):
#                     Results_path = r'D:\Sejong_Univ\research\Driver_authentication\results\Journal_results\Results_files\Scenario1_case2'
#                     # filename = 'Seat_OCSVM_result_' + str(split_ratio[k]) + '_' + str(gammas[g]) + '_' + str(nu[n]) + '_Tr_user' + str(i) + '_Ts_user' + str(j) + '.csv'
#                     filename = 'Seat_OCSVM_result_ratio_' + str(split_ratio[k]) + '_gamma_' + str(gamma[g]) + '_nu_' + str(nu[n]) + '_Tr_user' + str(i) + '.csv'
#                     run2(X_tr_N, X_ts_N, X_ts_A, split_ratio[k], gamma[g], nu[n], i, Results_path, filename)




