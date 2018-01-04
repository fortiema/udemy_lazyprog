import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from hmmd import HMM


def get_data(fname, split_sequences):
    word2id = {}
    pos2id = {}

    X = []
    Y = []

    curr_id = 0
    curr_pos_id = 0
    with open(fname, 'r') as fin:
        x = []
        y = []
        for line in fin:
            line = line.rstrip()
            if not line:
                if x and y:
                    X.append(x[:])
                    Y.append(y[:])
                    x.clear()
                    y.clear()
            else:
                cols = line.split(' ')
                if len(cols) == 3:
                    token, pos, bio = cols
                    if token not in word2id:
                        word2id[token] = curr_id
                        curr_id += 1
                    if pos not in pos2id:
                        pos2id[pos] = curr_pos_id
                        curr_pos_id += 1
                    x.append(word2id[token])
                    y.append(pos2id[pos])
                    # # Debug
                    # x.append(token)
                    # y.append(pos)

    print(len(X), len(Y), curr_id, curr_pos_id)

    print(X[0], Y[0])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)

    print(len(X_train), len(Y_train), len(X_test), len(Y_test))
    return X_train, Y_train, X_test, Y_test, word2id, {v:k for k, v in pos2id.items()}


def accuracy(T, Y):
    n_correct = 0
    n_total = 0
    for t, y in zip(T, Y):
        n_correct += np.sum(t == y)
        n_total += len(y)
    return float(n_correct) / n_total


def total_f1_score(T, Y):
    T = np.concatenate(T)
    Y = np.concatenate(Y)
    return f1_score(T, Y, average='macro')


def main(fname, smoothing=10e-2):
    Xtrain, Ytrain, Xtest, Ytest, word2id, id2pos = get_data(fname, split_sequences=True)
    V = len(word2id) + 1

    M = max(max(y) for y in Ytrain) + 1
    A = np.ones((M, M)) * smoothing
    pi = np.zeros(M)
    for y in Ytrain:
        pi[y[0]] += 1
        for i in range(len(y) - 1):
            A[y[i], y[i+1]] += 1
    A /= A.sum(axis=1, keepdims=True)
    pi /= pi.sum()

    B = np.ones((M, V)) * smoothing
    for x, y in zip(Xtrain, Ytrain):
        for xi, yi in zip(x, y):
            B[yi, xi] += 1    # The state is the target here, needs to be 1st dim!
    B /= B.sum(axis=1, keepdims=True)

    hmm = HMM(M)
    hmm.pi = pi
    hmm.A = A
    hmm.B = B

    Ptrain = []
    for x in Xtrain:
        p = hmm.get_state_sequence(x)
        Ptrain.append(p)
    
    Ptest = []
    for x in Xtest:
        p = hmm.get_state_sequence(x)
        Ptest.append(p)

    print('Train accuracy: {}'.format(accuracy(Ytrain, Ptrain)))
    print('Train f1: {}'.format(total_f1_score(Ytrain, Ptrain)))
    print('Test accuracy: {}'.format(accuracy(Ytest, Ptest)))
    print('Test f1: {}'.format(total_f1_score(Ytest, Ptest)))


if __name__ == '__main__':
    main(sys.argv[1])
