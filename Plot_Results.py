from sklearn import metrics
import warnings
import seaborn as sn
import pandas as pd
from sklearn.metrics import roc_curve
from itertools import cycle
from prettytable import PrettyTable

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_roc():
    lw = 2
    cls = ['GRU', 'RNN', 'LSTM', 'TCN', 'AGSOA-MRHNet']
    colors = cycle(["m", "b", "r", "lime", "k"])
    Predicted = np.load('roc_score.npy', allow_pickle=True)
    Actual = np.load('roc_act.npy', allow_pickle=True)
    for i in range(len(Actual)):  # For all Datasets
        Dataset = ['Dataset1', 'Dataset2']
        for j, color in zip(range(5), colors):  # For all classifiers

            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[i, 3, j], Predicted[i, 3, j])
            auc = metrics.roc_auc_score(Actual[i, 3, j], Predicted[i, 3, j])
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[j]
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/%s_ROC.png" % (Dataset[i])
        plt.savefig(path)
        plt.show()


def plot_results_conv():
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'RSA-MRHNet', 'TFMOA-MRHNet', 'COA-MRHNet', 'GSOA-MRHNet', 'AGSOA-MRHNet']
    for i in range(Fitness.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Dataset = ['Dataset1', 'Dataset2']
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- ', Dataset[i], 'Statistical Report ',
              '--------------------------------------------------')
        print(Table)
        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='m', linewidth=3, marker='h', markerfacecolor='red', markersize=12,
                 label='RSA-MRHNet')
        plt.plot(length, Conv_Graph[1, :], color='y', linewidth=3, marker='p', markerfacecolor='green',
                 markersize=12,
                 label='TFMOA-MRHNet')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='.', markerfacecolor='cyan',
                 markersize=12,
                 label='COA-MRHNet')
        plt.plot(length, Conv_Graph[3, :], color='lime', linewidth=3, marker='o', markerfacecolor='magenta',
                 markersize=12,
                 label='GSOA-MRHNet')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12,
                 label='AGSOA-MRHNet')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/%s_Conv.png" % (Dataset[i]))
        plt.show()


def plot_results_kfold():
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7]
    Algorithm = ['TERMS', 'RSA-MRHNet', 'TFMOA-MRHNet', 'COA-MRHNet', 'GSOA-MRHNet', 'AGSOA-MRHNet']
    Classifier = ['TERMS', 'GRU', 'RNN', 'LSTM', 'TCN', 'AGSOA-MRHNet']
    Dataset = ['Dataset1', 'Dataset2']
    eval = np.load('Eval_Fold.npy', allow_pickle=True)
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, 4:]
        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[j, :])
        print('-------------------------------------------------- ', Dataset[i], ' - 5 - Fold', 'Classifier Comparison'
                                                                                                '--------------------------------------------------')
        print(Table)
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if Graph_Term[j] == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 0], edgecolor='k', hatch='//', color='r', width=0.10, label="GRU")
            ax.bar(X + 0.10, Graph[:, 1], edgecolor='k', hatch='-', color='m', width=0.10, label="RNN")
            ax.bar(X + 0.20, Graph[:, 2], edgecolor='k', hatch='//', color='lime', width=0.10, label="LSTM")
            ax.bar(X + 0.30, Graph[:, 3], edgecolor='k', hatch='-', color='b', width=0.10, label="TCN")
            ax.bar(X + 0.40, Graph[:, 4], edgecolor='w', hatch='..', color='k', width=0.10, label="AGSOA-MRHNet")
            plt.xticks(X + 0.25, ('1', '2', '3', '4', '5'))
            plt.xlabel('K - Fold')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path = "./Results/%s_%s_bar_2.png" % (Dataset[i], Terms[Graph_Term[j]])
            plt.savefig(path)
            plt.show()


def plot_results_Feature_table():
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Algorithm = ['TERMS', 'RSA-MRHNet', 'TFMOA-MRHNet', 'COA-MRHNet', 'GSOA-MRHNet', 'AGSOA-MRHNet']
    Dataset = ['Dataset1', 'Dataset2']
    eval = np.load('Eval_All.npy', allow_pickle=True)
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('-------------------------------------------------- ', Dataset[i], ' Feature', 'Algorithm Comparison'
                                                                                             '--------------------------------------------------')
        print(Table)


if __name__ == "__main__":
    plot_results_kfold()
    plot_roc()
    plot_results_conv()
    plot_results_Feature_table()
