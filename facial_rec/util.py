import csv
import os

import numpy as np
import pandas as pd


def init_w_b(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def init_filter(shape, pool_sz):
    w = np.random.randn(*shape) / (np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:]) / np.prod(pool_sz)))
    return w.astype(np.float32)


def relu(x):
    return x * (x > 0)


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def softmax(A):
    exp_a = np.exp(A)
    return exp_a / exp_a.sum(axis=1, keepdims=True)


def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


def cost(T, Y):
    N = len(T)
    return -np.log(Y[np.arange(N), T]).sum()


def error_rate(targets, preds):
    return np.mean(targets != preds)


def indicators(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def get_data(fname, limit=None, balance_ones=True):
    if os.path.exists(fname):
        X = []
        Y = []

        with open(fname) as fin:
            reader = csv.reader(fin, delimiter=',')

            for i, row in enumerate(reader):
                if limit and i > limit:
                    break
                if i:
                    X.append(np.array(row[1].split(' '), dtype=np.int16))
                    Y.append(np.array(row[0], dtype=np.int16))

        # Normalize X, convert to np array
        X, Y = np.array(X) / 255.0, np.array(Y)

        return X, Y

    else:
        raise AttributeError('File does not exist!')


def get_data_image(fname, limit=None, balance_ones=True):
    X, Y = get_data(fname, limit, balance_ones)
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y, d
