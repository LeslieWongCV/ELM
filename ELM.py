# -*- coding: utf-8 -*-
# @Time    : 2021/4/7 4:31 下午
# @Author  : Yushuo Wang
# @FileName: ELM.py
# @Software: PyCharm
# @Blog    ：https://lesliewongcv.github.io/
import numpy as np
import os
from scipy.io import loadmat
from sklearn.model_selection import KFold


def sigmoid(a, b, x):

    return 1.0 / (1 + np.exp(-1.0 * (x.dot(a) + b)))


def ELM_prototype(X, T, C, n, L, node_num):
    '''
    Variables：X - input；No. of samples * No. of feature（N*n）
             ：H - H matrix；No. of samples * No. of hidden nodes（N*L）
             ：T - Target；No. of samples * No. of output nodes（N*M）
             ：C - Hyper-parm of the regularization
    '''
    # init the weight of hidden layers randomly
    a = np.random.normal(0, 1, (n, node_num))
    b = np.random.normal(0, 1)
    T = one_hot(T)
    H = sigmoid(a, b, X)
    # calculate the weight of output layers(beta) and output
    HH = H.T.dot(H)
    HT = H.T.dot(T)
    beta = np.linalg.pinv(HH + np.identity(node_num) / C).dot(HT)
    Fl = H.dot(beta)
    return beta, Fl, a, b


def one_hot(l):
    y = np.zeros([len(l), np.max(l)+1])
    for i in range(len(l)):
        y[i, l[i]] = 1
    return y


def predict(X, BETA, a, b):
    H = sigmoid(a, b, X)
    Y = H.dot(BETA)
    Y = Y.argmax(1)
    return Y


def evaluation(y_hat, goundtruth):
    y_hat = y_hat[:, np.newaxis]
    return np.sum(np.equal(y_hat, goundtruth) / len(y_hat))


KFOLD = 4
PATH = '/Users/leslie/Downloads/MatDataset/'  # Path to the dataset
folders = os.listdir(PATH)
res = []
describe = []
C_list = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]  # # hyper-pram 1/C
progress = 0
for folder_name in folders:
    progress += 100/29
    file_name = folder_name
    if folder_name == '.DS_Store':
        continue
    else:
        matfn = PATH + folder_name + '/' + folder_name + '_Train.mat'
        df_data = loadmat(matfn)['Data']
        df_label = loadmat(matfn)['Label']
        kf = KFold(n_splits=4, shuffle=False)

        for C in C_list:
            vali_res = 0
            for train_index, test_index in kf.split(df_label):  # 4-fold
                beta, Fl, A, B = ELM_prototype(df_data[train_index], df_label[train_index],
                                               C=C, n=len(df_data[1]), L=len(train_index), node_num=100)
                y_valid = predict(df_data[test_index], beta, A, B)
                acc_valid = evaluation(y_valid, df_label[test_index])
                vali_res += acc_valid
            res += [folder_name + ':' + str(vali_res/4) + '   C = ' + str(C)]
    print(str(round(progress)) + "%")  # show progress
res = np.array(res)
np.savetxt("ELM_acc_100.txt", res, fmt='%s', delimiter=',')
