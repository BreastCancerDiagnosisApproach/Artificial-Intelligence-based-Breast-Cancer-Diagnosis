import numpy as np
import pandas as pd
from numpy import matlib
import random
from COA import COA
from GSOA import GSOA
from Global_Vars import Global_Vars
from Model_GRU import Model_GRU
from Model_LSTM import Model_LSTM
from Model_PROPOSED import Model_PROPOSED
from Model_RNN import Model_RNN
from Model_TCN import Model_TCN
from Objective_Function import objfun_feat
from PROPOSED import PROPOSED
from Plot_Results import plot_results_kfold, plot_roc, plot_results_conv, plot_results_Feature_table
from RSA import RSA
from TFMOA import TFMOA

no_of_dataset = 2

# Read Dataset 1
an = 0
if an == 1:
    str_type = ['Day of the week', 'Traffic Situation']
    Directory = './Dataset/Dataset1/Traffic.csv'
    rawData = pd.read_csv(Directory)
    rawData = rawData.drop("Time", axis=1)
    for j in range(len(str_type)):
        if 'Traffic Situation' in str_type[j]:
            uni = np.unique(rawData['Traffic Situation'])
            Target = np.zeros((len(rawData['Traffic Situation']), len(uni))).astype('int')
            for j in range(len(uni)):
                ind = np.where(rawData['Traffic Situation'] == uni[j])
                Target[ind[0], j] = 1
        else:
            arr = np.asarray(rawData[str_type[j]])
            uni = np.unique(rawData[str_type[j]])
            for k in range(len(uni)):
                ind = np.where(arr == uni[k])
                arr[ind[0]] = k + 1
            rawData[str_type[j]] = arr
    np.save('Data_1.npy', np.asarray(rawData)[:, :-1])
    np.save('Target_1.npy', Target)

# Read Dataset 2
an = 0
if an == 1:
    str_type = ['trip_id', 'Degree_of_congestion']
    Directory = './Dataset/Dataset2/GTFS_Data.csv'
    rawData = pd.read_csv(Directory)
    rawData = rawData.drop("arrival_time", axis=1)
    for j in range(len(str_type)):
        if 'Degree_of_congestion' in str_type[j]:
            uni = np.unique(rawData['Degree_of_congestion'])
            Target = np.zeros((len(rawData['Degree_of_congestion']), len(uni))).astype('int')
            for j in range(len(uni)):
                ind = np.where(rawData['Degree_of_congestion'] == uni[j])
                Target[ind[0], j] = 1
        else:
            arr = np.asarray(rawData[str_type[j]])
            uni = np.unique(rawData[str_type[j]])
            for k in range(len(uni)):
                ind = np.where(arr == uni[k])
                arr[ind[0]] = k + 1
            rawData[str_type[j]] = arr
    np.save('Data_2.npy', np.asarray(rawData)[:, :-1])
    np.save('Target_2.npy', Target)

### Optimization for Weighted Features Selection
an = 0
if an == 1:
    for n in range(no_of_dataset):
        data = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        Tar = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Feat = data
        Target = Tar
        Global_Vars.Feat = Feat
        Global_Vars.Target = Target
        Npop = 10
        Chlen = 2 * 7
        xmin = matlib.repmat(np.concatenate(([np.zeros(7), 0.01 * np.ones((1, 7))]), axis=None), Npop, 1)
        xmax = matlib.repmat(np.concatenate(([Feat.shape[1] - 1 * np.ones(7), 0.99 * np.ones((1, 7))]), axis=None),
                             Npop, 1)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
        fname = objfun_feat
        max_iter = 50

        print('RSA....')
        [bestfit1, fitness1, bestsol1, Time1] = RSA(initsol, fname, xmin, xmax, max_iter)

        print('TFMOA....')
        [bestfit2, fitness2, bestsol2, Time2] = TFMOA(initsol, fname, xmin, xmax, max_iter)

        print('COA....')
        [bestfit3, fitness3, bestsol3, Time3] = COA(initsol, fname, xmin, xmax, max_iter)

        print('GSOA....')
        [bestfit4, fitness4, bestsol4, Time4] = GSOA(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

        BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
        fit = [fitness1, fitness2, fitness3, fitness4, fitness5]

        np.save('Fitness.npy', fit)
        np.save('BEST_Sol' + str(n + 1) + '.npy', BestSol)

### Weighted Feature Selection
an = 0
if an == 1:
    for n in range(no_of_dataset):
        Feat = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        sol = np.load('BEST_Sol' + str(n + 1) + '.npy', allow_pickle=True)[4, :]
        weight = sol[7:]  # For Proposed only
        Selected_Feature = Feat[:, np.round(sol[0:7]).astype('int')]
        Weighted_Feature = Selected_Feature * weight
        np.save('Selected_Feature_' + str(n + 1) + '.npy', Weighted_Feature)

# K Fold Classification
an = 0
if an == 1:
    Eval_all = []
    for n in range(no_of_dataset):
        Feature = np.load('Selected_Feature_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        K = 5
        Per = 1 / 5
        Perc = round(Feature.shape[0] * Per)
        eval = []
        for i in range(K):
            Eval = np.zeros((5, 14))
            Feat = Feature
            Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Feat.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Feat[train_index, :]
            Train_Target = Target[train_index, :]
            Eval[0, :], pred1 = Model_GRU(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[1, :], pred2 = Model_RNN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[2, :], pred3 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[3, :], pred4 = Model_TCN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[4, :], pred5 = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target)
            eval.append(Eval)
        Eval_all.append(eval)
    np.save('Eval_Fold.npy', np.asarray(Eval_all))

plot_results_kfold()
plot_roc()
plot_results_conv()
plot_results_Feature_table()
