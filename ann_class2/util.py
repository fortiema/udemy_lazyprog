import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


def get_transformed_data(fname):
    print("Reading in and transforming data...")
    df = pd.read_csv(fname)
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)

    # Center the data (have a mean of 0)
    X = data[:, 1:]
    mu = X.mean(axis=0)
    X = X - mu

    # Principal component analysis
    pca = PCA()
    Z = pca.fit_transform(X)
    Y = data[:, 0].astype(np.int32)

    return Z, Y, pca, mu


def get_normalized_data(fname):
    print("Reading in and transforming data...")
    df = pd.read_csv(fname)
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)

    # Normalize data using Z-scores (mean = 0, stdev = 1)
    X = data[:, 1:]
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std

    Y = data[:, 0]

    return X, Y


def plot_cumul_variance(pca):
    P = []
    for p in pca.explained_variante_ratio:
        if len(P) == 0:
            P.append(p)
        else:
            P.append(p + P[-1])
    plt.plot(P)
    plt.show()
    return P


def forward(X, W, b):
    """Softmax Forward Prop
    """
    a = X.dot(W) + b
    expa = np.exp(a)
    y = expa / expa.sum(axis=1, keepdims=True)
    return y


def predict(p_y):
    return np.argmax(p_y, axis=1)


def error_rate(p_y, t):
    pred = predict(p_y)
    return np.mean(pred != t)


def cost(p_y, t):
    return -(t*np.log(p_y)).sum()


def gradW(t, y, X):
    return X.T.dot(t-y)


def gradb(t, y):
    return (t-y).sum(axis=0)


def y2indicator(y):
    N = len(y)
    K = len(set(y))
    y = y.astype(np.int32)
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def benchmark_full():
    X, Y = get_normalized_data('../large_files/train.csv')

    print("Performing logistic regression...")

    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:,]
    Ytest = Y[-1000:]

    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    W = np.random.randn(D, 10) / 28
    b = np.zeros(10)
    LL = []
    LLtest = []
    CRtest = []

    learn_rate = 3*10e-5
    reg = 1*10e-3

    for i in range(500):
        p_y = forward(Xtrain, W, b)
        ll = cost(p_y, Ytrain_ind)
        LL.append(ll)

        p_y_test = forward(Xtest, W, b)
        lltest = cost(p_y_test, Ytest_ind)
        LLtest.append(lltest)

        err = error_rate(p_y_test, Ytest)
        CRtest.append(lltest)

        W += learn_rate*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += learn_rate*(gradb(Ytrain_ind, p_y) - reg*b)
        if i % 10 == 0:
            print("{0: <6}> Cost: {1: .3f}, Error: {2: .3f}".format(i, lltest, err))

    p_y = forward(Xtest, W, b)
    print("Final error: {}".format(error_rate(p_y, Ytest)))
    iters = range(len(LL))
    plt.plot(iters, LL, iters, LLtest)
    plt.show()
    plt.plot(CRtest)
    plt.show()


def benchmark_pca():
    X, Y, _, _ = get_transformed_data('../large_files/train.csv')

    print("Performing logistic regression...")

    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:,]
    Ytest = Y[-1000:]

    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    W = np.random.randn(D, 10) / 28
    b = np.zeros(10)
    LL = []
    LLtest = []
    CRtest = []

    learn_rate = 3*10e-5
    reg = 1*10e-3

    for i in range(800):
        p_y = forward(Xtrain, W, b)
        ll = cost(p_y, Ytrain_ind)
        LL.append(ll)

        p_y_test = forward(Xtest, W, b)
        lltest = cost(p_y_test, Ytest_ind)
        LLtest.append(lltest)

        err = error_rate(p_y_test, Ytest)
        CRtest.append(lltest)

        W += learn_rate*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += learn_rate*(gradb(Ytrain_ind, p_y) - reg*b)
        if i % 10 == 0:
            print("{0: <6}> Cost: {1: .3f}, Error: {2: .3f}".format(i, lltest, err))

    p_y = forward(Xtest, W, b)
    print("Final error: {}".format(error_rate(p_y, Ytest)))
    iters = range(len(LL))
    plt.plot(iters, LL, iters, LLtest)
    plt.show()
    plt.plot(CRtest)
    plt.show()


if __name__ == '__main__':
    # benchmark_full()
    benchmark_pca()
