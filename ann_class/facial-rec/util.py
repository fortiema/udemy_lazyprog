import numpy as np
import pandas as pd


def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def init_filter(shape, pool_size):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(pool_size)))
    return w.astype(np.float32)


def relu(x):
    return x * (x > 0)


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def sigmoid_cost(T, Y):
    """Binary Corss-Entropy
    """
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


def cost(T, Y):
    return -(T*np.log(Y)).sum()


def cost2(T, Y):
    N = len(T)
    return -np.log(Y[np.arange(N), T]).sum()


def error_rate(targets, predictions):
    return np.mean(targets != predictions)


def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def getData(fname, binary=False):
    """Fetches data from source file and format it
    """
    Y = []
    X = []
    first = True
    for line in open(fname):
        if first:
            first = False
        else:
            # Format is '<label>, <pixel values, space-separated>'
            # Adjust code accordingly
            row = line.split(',')
            y = int(row[0])
            if (not binary) or (y in {0, 1}):
                Y.append(y)
                X.append([int(p) for p in row[1].split()])

    # Convert to numpy arrays, normalize X values
    return np.array(X) / 255.0, np.array(Y)


def getImageData(fname):
    X, Y = getData(fname)
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y
