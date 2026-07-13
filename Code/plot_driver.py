# Classification template
# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.model_selection import train_test_split
import os
# # Packages for analysis
# from natsort import natsorted
import numpy as np
# importing Statistics module
import statistics
from collections import Counter
import math
# import metrics
from matplotlib import pyplot as plt
import seaborn as sns
#############################################################################################################################

def plot_OCSVM_mean_F1():# all data sizes (A, B, C, D) and types (BELT, SEAT, FUSION)
    Belt_all_sizes = [88.0, 83.6, 88.1, 83.9]
    Seat_all_sizes = [87.2, 85.6, 85.6, 78.8]

    Fusion_all_sizes = [83.8, 88.0, 86.2, 85.5]
    # set width of bar
    barWidth = 0.12
    fig = plt.subplots(figsize=(9, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(Belt_all_sizes))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    # colors = ['pink', 'lightgreen', 'lightblue','pink', 'lightgreen', 'lightblue']
    plt.bar(br1, Belt_all_sizes, width=barWidth, color='lightblue', edgecolor='black', label='Belt-only Dataset', hatch='\\')
    plt.bar(br2, Seat_all_sizes, width=barWidth, color='orange', edgecolor='black', label='Seat-only Dataset', hatch='o')
    plt.bar(br3, Fusion_all_sizes, width=barWidth, color='pink', edgecolor='black', label='Fusion Dataset', hatch='///')
    # plt.bar(br5, W_s_above_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S > 100)', hatch='\\')
    # plt.bar(br6, W_s_less_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S < 100)',hatch='o')

    plt.grid()

    # Adding Xticks
    plt.xlabel('Dataset sizes', fontsize=22)
    plt.ylabel('Mean F1 scores [%]', fontsize=22)
    plt.xticks([r + barWidth for r in range(4)], ['Size(A)', 'Size(B)', 'Size(C)', 'Size(D)'], fontsize=18)
    plt.yticks(fontsize=16)
    plt.title("OCSVM performance ",fontsize=22)
    # for i, v in enumerate(y):
    #     plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.legend(fontsize=18)
    plt.ylim(60, 100)
    plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_OCSVM_max_F1():# all data sizes (A, B, C, D) and types (BELT, SEAT, FUSION)
    Belt_all_sizes = [92.0, 87.5, 90.9, 87.4]
    Seat_all_sizes = [93.1, 89.4, 93.1, 83.7]
    Fusion_all_sizes = [90.6, 90.1, 92.2, 87.3]
    # set width of bar
    barWidth = 0.12
    fig = plt.subplots(figsize=(9, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(Belt_all_sizes))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    # colors = ['pink', 'lightgreen', 'lightblue','pink', 'lightgreen', 'lightblue']
    plt.bar(br1, Belt_all_sizes, width=barWidth, color='lightblue', edgecolor='black', label='Belt-only Dataset', hatch='\\')
    plt.bar(br2, Seat_all_sizes, width=barWidth, color='orange', edgecolor='black', label='Seat-only Dataset', hatch='o')
    plt.bar(br3, Fusion_all_sizes, width=barWidth, color='pink', edgecolor='black', label='Fusion Dataset', hatch='///')
    # plt.bar(br5, W_s_above_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S > 100)', hatch='\\')
    # plt.bar(br6, W_s_less_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S < 100)',hatch='o')

    plt.grid()

    # Adding Xticks
    plt.xlabel('Dataset sizes', fontsize=22)
    plt.ylabel('Max F1 scores [%]', fontsize=22)
    plt.xticks([r + barWidth for r in range(4)], ['Size(A)', 'Size(B)', 'Size(C)', 'Size(D)'], fontsize=18)
    plt.yticks(fontsize=16)
    plt.title("OCSVM performance ",fontsize=22)
    # for i, v in enumerate(y):
    #     plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.legend(fontsize=18)
    plt.ylim(60, 100)
    plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_LOF_max_F1():# all data sizes (A, B, C, D) and types (BELT, SEAT, FUSION)
    Belt_all_sizes = [82.09, 81.99, 82.55, 87.27]
    Seat_all_sizes = [86.21, 91.83, 84.14, 80.42]

    Fusion_all_sizes = [95.32, 98.01, 98.53, 96.45]
    # set width of bar
    barWidth = 0.12
    fig = plt.subplots(figsize=(9, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(Belt_all_sizes))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    # colors = ['pink', 'lightgreen', 'lightblue','pink', 'lightgreen', 'lightblue']
    plt.bar(br1, Belt_all_sizes, width=barWidth, color='lightblue', edgecolor='black', label='Belt-only Dataset', hatch='\\')
    plt.bar(br2, Seat_all_sizes, width=barWidth, color='orange', edgecolor='black', label='Seat-only Dataset', hatch='o')
    plt.bar(br3, Fusion_all_sizes, width=barWidth, color='pink', edgecolor='black', label='Fusion Dataset', hatch='///')
    # plt.bar(br5, W_s_above_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S > 100)', hatch='\\')
    # plt.bar(br6, W_s_less_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S < 100)',hatch='o')

    plt.grid()

    # Adding Xticks
    plt.xlabel('Dataset sizes', fontsize=22)
    plt.ylabel('Max F1 scores [%]', fontsize=22)
    plt.xticks([r + barWidth for r in range(4)], ['Size(A)', 'Size(B)', 'Size(C)', 'Size(D)'], fontsize=18)
    plt.yticks(fontsize=16)
    plt.title("LOF performance ",fontsize=22)
    # for i, v in enumerate(y):
    #     plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.legend(fontsize=18)
    plt.ylim(60, 100)
    plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_LOF_mean_F1():# all data sizes (A, B, C, D) and types (BELT, SEAT, FUSION)
    Belt_all_sizes = [75.17, 76.09, 77.62, 62.68]
    Seat_all_sizes = [77.67, 81.62, 79.49, 50.57]

    Fusion_all_sizes = [87.57, 93.66, 95.96, 77.29]
    # set width of bar
    barWidth = 0.12
    fig = plt.subplots(figsize=(9, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(Belt_all_sizes))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    # colors = ['pink', 'lightgreen', 'lightblue','pink', 'lightgreen', 'lightblue']
    plt.bar(br1, Belt_all_sizes, width=barWidth, color='lightblue', edgecolor='black', label='Belt-only Dataset', hatch='\\')
    plt.bar(br2, Seat_all_sizes, width=barWidth, color='orange', edgecolor='black', label='Seat-only Dataset', hatch='o')
    plt.bar(br3, Fusion_all_sizes, width=barWidth, color='pink', edgecolor='black', label='Fusion Dataset', hatch='///')
    # plt.bar(br5, W_s_above_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S > 100)', hatch='\\')
    # plt.bar(br6, W_s_less_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S < 100)',hatch='o')

    plt.grid()

    # Adding Xticks
    plt.xlabel('Dataset sizes', fontsize=22)
    plt.ylabel('Mean F1 scores [%]', fontsize=22)
    plt.xticks([r + barWidth for r in range(4)], ['Size(A)', 'Size(B)', 'Size(C)', 'Size(D)'], fontsize=18)
    plt.yticks(fontsize=16)
    plt.title("LOF performance ",fontsize=22)
    # for i, v in enumerate(y):
    #     plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.legend(fontsize=18)
    plt.ylim(60, 100)
    plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



def plot_IF_max_F1():# all data sizes (A, B, C, D) and types (BELT, SEAT, FUSION)
    Belt_all_sizes = [86.06,83.48 , 89.80, 85.80]
    Seat_all_sizes = [89.44,90.83 ,83.68 ,86.11 ]

    Fusion_all_sizes = [91.56,88.19 , 83.75, 88.76]
    # set width of bar
    barWidth = 0.12
    fig = plt.subplots(figsize=(9, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(Belt_all_sizes))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    # colors = ['pink', 'lightgreen', 'lightblue','pink', 'lightgreen', 'lightblue']
    plt.bar(br1, Belt_all_sizes, width=barWidth, color='lightblue', edgecolor='black', label='Belt-only Dataset', hatch='\\')
    plt.bar(br2, Seat_all_sizes, width=barWidth, color='orange', edgecolor='black', label='Seat-only Dataset', hatch='o')
    plt.bar(br3, Fusion_all_sizes, width=barWidth, color='pink', edgecolor='black', label='Fusion Dataset', hatch='///')
    # plt.bar(br5, W_s_above_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S > 100)', hatch='\\')
    # plt.bar(br6, W_s_less_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S < 100)',hatch='o')

    plt.grid()

    # Adding Xticks
    plt.xlabel('Dataset sizes', fontsize=22)
    plt.ylabel('Max F1 scores [%]', fontsize=22)
    plt.xticks([r + barWidth for r in range(4)], ['Size(A)', 'Size(B)', 'Size(C)', 'Size(D)'], fontsize=18)
    plt.yticks(fontsize=16)
    plt.title("IF performance ",fontsize=22)
    # for i, v in enumerate(y):
    #     plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.legend(fontsize=18)
    plt.ylim(60, 100)
    plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_IF_mean_F1():# all data sizes (A, B, C, D) and types (BELT, SEAT, FUSION)
    Belt_all_sizes = [80.94,79.30 , 81.76, 79.61]
    Seat_all_sizes = [82.78, 80.17, 75.93, 72.24]

    Fusion_all_sizes = [82.50, 82.71, 73.63, 78.24]
    # set width of bar
    barWidth = 0.12
    fig = plt.subplots(figsize=(9, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(Belt_all_sizes))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    # colors = ['pink', 'lightgreen', 'lightblue','pink', 'lightgreen', 'lightblue']
    plt.bar(br1, Belt_all_sizes, width=barWidth, color='lightblue', edgecolor='black', label='Belt-only Dataset', hatch='\\')
    plt.bar(br2, Seat_all_sizes, width=barWidth, color='orange', edgecolor='black', label='Seat-only Dataset', hatch='o')
    plt.bar(br3, Fusion_all_sizes, width=barWidth, color='pink', edgecolor='black', label='Fusion Dataset', hatch='///')
    # plt.bar(br5, W_s_above_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S > 100)', hatch='\\')
    # plt.bar(br6, W_s_less_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S < 100)',hatch='o')

    plt.grid()

    # Adding Xticks
    plt.xlabel('Dataset sizes', fontsize=22)
    plt.ylabel('Mean F1 scores [%]', fontsize=22)
    plt.xticks([r + barWidth for r in range(4)], ['Size(A)', 'Size(B)', 'Size(C)', 'Size(D)'], fontsize=18)
    plt.yticks(fontsize=16)
    plt.title("IF performance ",fontsize=22)
    # for i, v in enumerate(y):
    #     plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.legend(fontsize=18)
    plt.ylim(60, 100)
    plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def plot_EE_max_F1():# all data sizes (A, B, C, D) and types (BELT, SEAT, FUSION)
    Belt_all_sizes = [86.06,85.45 ,85.85 ,84.58 ]
    Seat_all_sizes = [90.37, 91.09,90.43 ,82.91 ]

    Fusion_all_sizes = [92.52, 83.65, 88.40, 95.79]
    # set width of bar
    barWidth = 0.12
    fig = plt.subplots(figsize=(9, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(Belt_all_sizes))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    # colors = ['pink', 'lightgreen', 'lightblue','pink', 'lightgreen', 'lightblue']
    plt.bar(br1, Belt_all_sizes, width=barWidth, color='lightblue', edgecolor='black', label='Belt-only Dataset', hatch='\\')
    plt.bar(br2, Seat_all_sizes, width=barWidth, color='orange', edgecolor='black', label='Seat-only Dataset', hatch='o')
    plt.bar(br3, Fusion_all_sizes, width=barWidth, color='pink', edgecolor='black', label='Fusion Dataset', hatch='///')
    # plt.bar(br5, W_s_above_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S > 100)', hatch='\\')
    # plt.bar(br6, W_s_less_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S < 100)',hatch='o')

    plt.grid()

    # Adding Xticks
    plt.xlabel('Dataset sizes', fontsize=22)
    plt.ylabel('Max F1 scores [%]', fontsize=22)
    plt.xticks([r + barWidth for r in range(4)], ['Size(A)', 'Size(B)', 'Size(C)', 'Size(D)'], fontsize=18)
    plt.yticks(fontsize=16)
    plt.title("EE performance ",fontsize=22)
    # for i, v in enumerate(y):
    #     plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.legend(fontsize=18)
    plt.ylim(60, 100)
    plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_EE_mean_F1():# all data sizes (A, B, C, D) and types (BELT, SEAT, FUSION)
    Belt_all_sizes = [80.94,80.88 ,83.02, 80.93]
    Seat_all_sizes = [84.41, 86.71, 84.21,79.80 ]

    Fusion_all_sizes = [84.91, 81.31, 83.27,94.51 ]
    # set width of bar
    barWidth = 0.12
    fig = plt.subplots(figsize=(9, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(Belt_all_sizes))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    # colors = ['pink', 'lightgreen', 'lightblue','pink', 'lightgreen', 'lightblue']
    plt.bar(br1, Belt_all_sizes, width=barWidth, color='lightblue', edgecolor='black', label='Belt-only Dataset', hatch='\\')
    plt.bar(br2, Seat_all_sizes, width=barWidth, color='orange', edgecolor='black', label='Seat-only Dataset', hatch='o')
    plt.bar(br3, Fusion_all_sizes, width=barWidth, color='pink', edgecolor='black', label='Fusion Dataset', hatch='///')
    # plt.bar(br5, W_s_above_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S > 100)', hatch='\\')
    # plt.bar(br6, W_s_less_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S < 100)',hatch='o')

    plt.grid()

    # Adding Xticks
    plt.xlabel('Dataset sizes', fontsize=22)
    plt.ylabel('Mean F1 scores [%]', fontsize=22)
    plt.xticks([r + barWidth for r in range(4)], ['Size(A)', 'Size(B)', 'Size(C)', 'Size(D)'], fontsize=18)
    plt.yticks(fontsize=16)
    plt.title("EE performance ",fontsize=22)
    # for i, v in enumerate(y):
    #     plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.legend(fontsize=18)
    plt.ylim(60, 100)
    plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def plot_bar_belt_seat():
    Belt_Seat_90_10 = [94.96,	95,	95,	95]  # number of wesites that have more than 100 samples
    Belt_Seat_80_20 = [95.8,	96,	96,	96]  # number of wesites that have more than 100 samples

    Belt_Seat_70_30 = [96.12,	96,	96,	96]  # number of wesites that have more than 100 samples
    Belt_Seat_60_40 = [96.08,	96,	96,	96]  # number of wesites that have more than 100 samples

    Accuracy = [94.96, 95.8, 96.12,  96.08]
    Precision =[95,96,96,96]
    Recall = [95,96,96,96]
    F1_score =[95,96,96,96]

    # set width of bar
    barWidth = 0.12
    fig = plt.subplots(figsize=(9, 8))
    # Set position of bar on X axis
    br1 = np.arange(len(Accuracy))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    # colors = ['pink', 'lightgreen', 'lightblue','pink', 'lightgreen', 'lightblue']
    plt.bar(br1, Accuracy, width=barWidth, color='lightblue', edgecolor='black', label='Accuracy', hatch='\\')
    plt.bar(br2, Precision, width=barWidth, color='orange', edgecolor='black', label='Precision' , hatch='o')
    plt.bar(br3, Recall, width=barWidth, color='pink', edgecolor='black', label= 'Recall', hatch='///')
    plt.bar(br4, F1_score, width=barWidth, color='lightgreen', edgecolor='black', label='F1-score', hatch='*')
    # plt.bar(br5, W_s_above_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S > 100)', hatch='\\')
    # plt.bar(br6, W_s_less_100_f3, width=barWidth, color='lightgreen', edgecolor='black', label='F3(S < 100)',hatch='o')

    plt.grid()

    # Adding Xticks
    plt.xlabel('Train/Test split ratios', fontsize=22)
    plt.ylabel('Performance Percentage [%]', fontsize=22)
    plt.xticks([r + barWidth for r in range(4)], [ '90/10', '80/20', '70/30' , '60/40'], fontsize=18)
    plt.yticks(fontsize=16)
    # plt.title("Fusion of Belt and Seat pressure data",fontsize=22)
    # for i, v in enumerate(y):
    #     plt.text(xlocs[i] - 0.25, v + 0.01, str(v))
    plt.legend(fontsize=18)
    plt.ylim(80,100)
    plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Training_testing_times_plot_for Belt and Seat



def plot_authentication_time_Belt():
    Tr_LR = [75.80739999, 79.47,97.00799998, 107.0404]
    Ts_LR = [7.429299993,6.2301,6.175899995, 6.091699994]
    Tr_RF = [4610.4885, 5156.2866, 5682.0133, 6098.119]
    Ts_RF = [536.3063, 512.1129, 476.568, 439.091]
    fig, ax = plt.subplots(figsize=(9, 10))
    Rounds = ['60/40','70/30','80/20','90/10']
    labels_list=['Train using RF', 'Test using RF', 'Train using LR', 'Test using LR']
    color_list= ['red', 'black', 'blue', 'green']
    linestyle_list = ['-','--','-','--','-','--']
    plt.plot(Rounds, Tr_RF, linestyle=linestyle_list[0], marker='^', color=color_list[2], lw=3, label=labels_list[0],clip_on=False)
    plt.plot(Rounds, Ts_RF, linestyle=linestyle_list[1], marker='*', color=color_list[3], lw=3, label=labels_list[1],clip_on=False)
    plt.plot(Rounds, Tr_LR , linestyle=linestyle_list[0], marker='^', color=color_list[0], lw=3,label=labels_list[2] , clip_on=False)
    plt.plot(Rounds, Ts_LR, linestyle=linestyle_list[1], marker='*', color=color_list[1], lw=3,label=labels_list[3] , clip_on=False)

    plt.xlabel('Train/Test split ratios', fontsize=22)
    plt.ylabel('Time in milliseconds (ms)', fontsize=22)
    # plt.title('Time distribution for location identification' , fontsize = 16)
    plt.legend(loc="best", fontsize = 20)
    plt.grid(linestyle='--', linewidth='0.2', color='black')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_yscale('log')
    plt.show()


def plot_authentication_time_Seat():
    Tr_LR = [136.4067, 148.8696,165.3314, 186.3274]
    Ts_LR = [9.076100017,7.714999985,6.890400022,6.3868]
    Tr_RF = [13166.1955,14772.6596,16350.7606, 17717.1758]
    Ts_RF = [1340.454, 1125.1364,1051.7782, 962.6144]
    fig, ax = plt.subplots(figsize=(9, 10))
    Rounds = ['60/40','70/30','80/20','90/10']
    labels_list=['Train using RF', 'Test using RF', 'Train using LR', 'Test using LR']
    color_list= ['red', 'black', 'blue', 'green']
    linestyle_list = ['-','--','-','--','-','--']
    plt.plot(Rounds, Tr_RF, linestyle=linestyle_list[0], marker='^', color=color_list[2], lw=3, label=labels_list[0],clip_on=False)
    plt.plot(Rounds, Ts_RF, linestyle=linestyle_list[1], marker='*', color=color_list[3], lw=3, label=labels_list[1],clip_on=False)
    plt.plot(Rounds, Tr_LR , linestyle=linestyle_list[0], marker='^', color=color_list[0], lw=3,label=labels_list[2] , clip_on=False)
    plt.plot(Rounds, Ts_LR, linestyle=linestyle_list[1], marker='*', color=color_list[1], lw=3,label=labels_list[3] , clip_on=False)

    plt.xlabel('Train/Test split ratios', fontsize=22)
    plt.ylabel('Time in milliseconds (ms)', fontsize=22)
    # plt.title('Time distribution for location identification' , fontsize = 16)
    plt.legend(loc="best", fontsize = 20)
    plt.grid(linestyle='--', linewidth='0.2', color='black')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_yscale('log')
    plt.show()


def plot_authentication_time_Belt_Seat():
    Tr_RF = [19.2383098
,21.0493906
,23.8572451
, 26.0526431
]
    Ts_RF = [2.3278662
,2.2575214
 ,2.1399131
,2.1129995 ]
    fig, ax = plt.subplots(figsize=(9, 10))
    Rounds = ['60/40','70/30','80/20','90/10']
    labels_list=['Train using RF', 'Test using RF']
    color_list= ['red', 'blue']
    linestyle_list = ['-','--','-','--','-','--']
    plt.plot(Rounds, Tr_RF, linestyle=linestyle_list[0], marker='^', color=color_list[0], lw=3, label=labels_list[0],clip_on=False)
    plt.plot(Rounds, Ts_RF, linestyle=linestyle_list[1], marker='*', color=color_list[1], lw=3, label=labels_list[1],clip_on=False)

    plt.xlabel('Train/Test split ratios', fontsize=22)
    plt.ylabel('Time in seconds (s)', fontsize=22)
    # plt.title('Time distribution for location identification' , fontsize = 16)
    plt.legend(loc="best", fontsize = 20)
    plt.grid(linestyle='--', linewidth='0.2', color='black')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # ax.set_yscale('log')
    plt.show()



# def plot_processing_time():# I should plot for 90/10 Tr/ts splitting
#     Tr_Belt_sizeA = [1.177854582,1.671163704,1.282299784,1.240545291,1.42760016,1.246763817,1.305882057,1.295345412,1.202963462,1.342045567,1.341427422,1.558618197]
#     Ts_Belt_sizeA = [0.892008993,1.11459094,0.998736359,0.962263566,1.042436216,1.008890823,0.931136564,0.911381858,0.891872682,1.007009128,1.10764539,1.136236231]
#     Tr_Seat_sizeA = [1.317627051,1.364363611,1.277354546,1.834127459,1.387681643,1.403436319,1.421136528,1.398272982,1.374954527,1.507400196,1.421700147,1.374799928]
#     Ts_Seat_sizeA = [1.076536253,1.05359092,1.174036413,1.328009011,1.153736375,1.139463857,1.277236268,1.121990975,1.083599996,1.120254651,1.22579094,1.13086377]
#     Tr_Fusion_sizeA = [1.688972719,1.781927274,1.462990905,1.606809081,1.618881837,1.460909092,1.484481819,1.559436353,1.536909089,1.593190891,1.516881854,1.608772764]
#     Ts_Fusion_sizeA = [1.206281817,1.351781824,1.279118164,1.212109113,1.172490915,1.175045449,1.180227264,1.181054564,1.147227288,1.192763628,1.200145439,1.286863614]
#     fig, ax = plt.subplots(figsize=(9, 10))
#
#     users = ['1','2','3','4','5','6','7','8','9','10','11','12']
#     labels_list=['Train Belt Data', 'Test Belt Data', 'Train Seat Data', 'Test Seat Data', 'Train Fusion Data','Test Fusion Data']
#     color_list= ['red', 'red','black', 'black','blue', 'blue']
#     linestyle_list = ['-','--','-','--','-','--','-','--']
#     plt.plot(users, Tr_Belt_sizeA, linestyle=linestyle_list[0], marker='^', color=color_list[0], lw=3, label=labels_list[0],clip_on=False)
#     plt.plot(users, Ts_Belt_sizeA, linestyle=linestyle_list[1], marker='*', color=color_list[1], lw=3, label=labels_list[1],clip_on=False)
#     plt.plot(users, Tr_Seat_sizeA , linestyle=linestyle_list[0], marker='^', color=color_list[2], lw=3,label=labels_list[2] , clip_on=False)
#     plt.plot(users, Ts_Seat_sizeA, linestyle=linestyle_list[1], marker='*', color=color_list[3], lw=3,label=labels_list[3] , clip_on=False)
#     plt.plot(users, Tr_Fusion_sizeA, linestyle=linestyle_list[0], marker='^', color=color_list[4], lw=3,label=labels_list[2], clip_on=False)
#     plt.plot(users, Ts_Fusion_sizeA, linestyle=linestyle_list[1], marker='*', color=color_list[5], lw=3,label=labels_list[3], clip_on=False)
#
#     plt.xlabel('User Index', fontsize=22)
#     plt.ylabel('Time in milliseconds (ms)', fontsize=22)
#     # plt.title('Time distribution for location identification' , fontsize = 16)
#     plt.legend(loc="best", fontsize = 20)
#     plt.grid(linestyle='--', linewidth='0.2', color='black')
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
#     ax.set_yscale('log')
#     plt.show()



def plot_processing_Tr_time():# I should plot for 90/10 Tr/ts splitting
    Tr_Belt_sizeA = [1.177854582,1.671163704,1.282299784,1.240545291,1.42760016,1.246763817,1.305882057,1.295345412,1.202963462,1.342045567,1.341427422,1.558618197]
    Tr_Seat_sizeA = [1.317627051,1.364363611,1.277354546,1.834127459,1.387681643,1.403436319,1.421136528,1.398272982,1.374954527,1.507400196,1.421700147,1.374799928]
    Tr_Fusion_sizeA = [1.688972719,1.781927274,1.462990905,1.606809081,1.618881837,1.460909092,1.484481819,1.559436353,1.536909089,1.593190891,1.516881854,1.608772764]
    fig, ax = plt.subplots(figsize=(9, 10))

    users = ['1','2','3','4','5','6','7','8','9','10','11','12']
    labels_list=['Train Belt Data',  'Train Seat Data',  'Train Fusion Data']
    color_list= ['red', 'black', 'blue']
    linestyle_list = ['-','-','-']
    plt.plot(users, Tr_Belt_sizeA, linestyle=linestyle_list[0], marker='^', color=color_list[0], lw=3, label=labels_list[0],clip_on=False)
    plt.plot(users, Tr_Seat_sizeA , linestyle=linestyle_list[0], marker='^', color=color_list[1], lw=3,label=labels_list[1] , clip_on=False)
    plt.plot(users, Tr_Fusion_sizeA, linestyle=linestyle_list[0], marker='^', color=color_list[2], lw=3,label=labels_list[2], clip_on=False)

    plt.xlabel('User Index', fontsize=22)
    plt.ylabel('Time in milliseconds (ms)', fontsize=22)
    # plt.title('Time distribution for location identification' , fontsize = 16)
    plt.legend(loc="best", fontsize = 20)
    plt.grid(linestyle='--', linewidth='0.2', color='black')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_yscale('log')
    plt.show()

def plot_processing_Ts_time():# I should plot for 90/10 Tr/ts splitting
    Ts_Belt_sizeA = [0.892008993,1.11459094,0.998736359,0.962263566,1.042436216,1.008890823,0.931136564,0.911381858,0.891872682,1.007009128,1.10764539,1.136236231]
    Ts_Seat_sizeA = [1.076536253,1.05359092,1.174036413,1.328009011,1.153736375,1.139463857,1.277236268,1.121990975,1.083599996,1.120254651,1.22579094,1.13086377]
    Ts_Fusion_sizeA = [1.206281817,1.351781824,1.279118164,1.212109113,1.172490915,1.175045449,1.180227264,1.181054564,1.147227288,1.192763628,1.200145439,1.286863614]
    fig, ax = plt.subplots(figsize=(9, 10))

    users = ['1','2','3','4','5','6','7','8','9','10','11','12']
    labels_list=[ 'Test Belt Data', 'Test Seat Data', 'Test Fusion Data']
    color_list= ['red', 'black', 'blue']
    linestyle_list = ['--','--','--']
    plt.plot(users, Ts_Belt_sizeA, linestyle=linestyle_list[1], marker='*', color=color_list[0], lw=3, label=labels_list[0],clip_on=False)
    plt.plot(users, Ts_Seat_sizeA, linestyle=linestyle_list[1], marker='*', color=color_list[1], lw=3,label=labels_list[1] , clip_on=False)
    plt.plot(users, Ts_Fusion_sizeA, linestyle=linestyle_list[1], marker='*', color=color_list[2], lw=3,label=labels_list[2], clip_on=False)

    plt.xlabel('User Index', fontsize=22)
    plt.ylabel('Time in milliseconds (ms)', fontsize=22)
    # plt.title('Time distribution for location identification' , fontsize = 16)
    plt.legend(loc="best", fontsize = 20)
    plt.grid(linestyle='--', linewidth='0.2', color='black')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.set_yscale('log')
    plt.show()

if __name__ == '__main__':
    # plot_OCSVM_mean_F1()
    # plot_OCSVM_max_F1()
    # plot_LOF_max_F1()
    # plot_LOF_mean_F1()
    # plot_IF_max_F1()
    # plot_IF_mean_F1()
    # plot_EE_max_F1()
    # plot_EE_mean_F1()




    # plot_bar_belt_seat()
    # plot_authentication_time_Belt()
    # plot_authentication_time_Seat()
    # plot_authentication_time_Belt_Seat()
    # plot_processing_time()
    # plot_processing_Tr_time()
    plot_processing_Ts_time()
