import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from util import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_w2, derivative_w1, derivative_b2, derivative_b1


def main():
    max_iter = 20
    print_period = 10

    X, Y = get_normalized_data('../large_files/train.csv')
    learn_rate = 1*10e-5
    reg = 10e-4

    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:,]
    Ytest = Y[-1000:]
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N, D = Xtrain.shape
    batch_size = 500
    nb_batches = int(N / batch_size)

    M = 300
    K = 10

    W1 = np.random.randn(D, M) / 28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)

    LL_batch = []
    CR_batch = []
    for i in range(max_iter):
        for j in range(nb_batches):
            slice_start = j * batch_size
            slice_end = slice_start + batch_size
            Xbatch = Xtrain[slice_start:slice_end,]
            Ybatch = Ytrain_ind[slice_start:slice_end,]

            p_y_batch, Z = forward(Xbatch, W1, b1, W2, b2)

            W2 -= learn_rate * (derivative_w2(Z, Ybatch, p_y_batch) + reg*W2)
            b2 -= learn_rate * (derivative_b2(Ybatch, p_y_batch) + reg*b2)
            W1 -= learn_rate * (derivative_w1(Xbatch, Z, Ybatch, p_y_batch, W2) + reg*W1)
            b1 -= learn_rate * (derivative_b1(Z, Ybatch, p_y_batch, W2) + reg*b1)

            if j % print_period == 0:
                p_y, _ = forward(Xtest, W1, b1, W2, b2)
                ll = cost(p_y, Ytest_ind)
                LL_batch.append(ll)
                err = error_rate(p_y, Ytest)
                CR_batch.append(err)
                print("{0: <6}-{1: <2}> Cost: {2: .3f}, Error: {3: .3f}".format(i, j, ll, err))
    
    p_y, _ = forward(Xtest, W1, b1, W2, b2)
    print("Final Error Rate: {0: .3f}".format(error_rate(p_y, Ytest)))

    # Momentum

    learn_rate = 1*10e-5
    reg = 10e-4

    W1 = np.random.randn(D, M) / 28
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)

    LL_momentum = []
    CR_momentum = []
    mu = 0.9
    dW2 = 0
    db2 = 0
    dW1 = 0
    db1 = 0
    for i in range(max_iter):
        for j in range(nb_batches):
            slice_start = j * batch_size
            slice_end = slice_start + batch_size
            Xbatch = Xtrain[slice_start:slice_end,]
            Ybatch = Ytrain_ind[slice_start:slice_end,]

            p_y_batch, Z = forward(Xbatch, W1, b1, W2, b2)

            dW2 = mu*dW2 - learn_rate * (derivative_w2(Z, Ybatch, p_y_batch) + reg*W2)
            W2 += dW2
            db2 = mu*db2 - learn_rate * (derivative_b2(Ybatch, p_y_batch) + reg*b2)
            b2 += db2
            dW1 = mu*dW1 - learn_rate * (derivative_w1(Xbatch, Z, Ybatch, p_y_batch, W2) + reg*W1)
            W1 += dW1
            db1 = mu*db1 - learn_rate * (derivative_b1(Z, Ybatch, p_y_batch, W2) + reg*b1)
            b1 += db1

            if j % print_period == 0:
                p_y, _ = forward(Xtest, W1, b1, W2, b2)
                ll = cost(p_y, Ytest_ind)
                LL_momentum.append(ll)
                err = error_rate(p_y, Ytest)
                CR_momentum.append(err)
                print("{0: <6}-{1: <2}> Cost: {2: .3f}, Error: {3: .3f}".format(i, j, ll, err))
    
    p_y, _ = forward(Xtest, W1, b1, W2, b2)
    print("Final Error Rate: {0: .3f}".format(error_rate(p_y, Ytest)))

    plt.plot(LL_batch, label='batch')
    plt.plot(LL_momentum, label='momentum')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()