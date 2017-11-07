import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime

from util import get_transformed_data, forward, error_rate, cost, gradW, gradb, y2indicator

def main():
    X, Y, _, _ = get_transformed_data('../large_files/train.csv')
    X = X[:, :300]

    mu = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mu) / std

    print("Performing logistic regression...")
    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:,]
    Ytest = Y[-1000:]

    N, D = Xtrain.shape
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    # 1) Full GD
    W = np.random.randn(D, 10) / 28
    b = np.zeros(10)

    learn_rate = 10e-5
    reg = 10e-3
    LL = []

    t0 = datetime.now()
    for i in range(200):
        p_y = forward(Xtrain, W, b)
        
        W += learn_rate * (gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
        b += learn_rate * (gradb(Ytrain_ind, p_y) - reg*b)

        p_y_test = forward(Xtest, W, b)
        ll = cost(p_y_test, Ytest_ind)
        LL.append(ll)

        if i % 10 == 0:
            err = error_rate(p_y_test, Ytest)
            print("{0: <6}> Cost: {1: .3f}, Error: {2: .3f}".format(i,ll,err))
    
    p_y = forward(Xtest, W, b)
    print("Final error rate: {0: .3f}".format(error_rate(p_y, Ytest)))
    print("Elapsed Time - Full GD: {}".format(datetime.now() - t0))

    # 2) Stochastic GD
    W = np.random.randn(D, 10) / 28
    b = np.zeros(10)

    learn_rate = 10e-5
    reg = 10e-3
    LL_stochastic = []

    t0 = datetime.now()
    for i in range(1):
        tmp_x, tmp_y = shuffle(Xtrain, Ytrain_ind)
        for n in range(min(N, 500)):
            x = tmp_x[n,:].reshape(1,D)
            y = tmp_y[n,:].reshape(1,10)
            p_y = forward(x, W, b)

            W += learn_rate * (gradW(y, p_y, x) - reg*W)
            b += learn_rate * (gradb(y, p_y) - reg*b)

            p_y_test = forward(Xtest, W, b)
            ll = cost(p_y_test, Ytest_ind)
            LL_stochastic.append(ll)

            if n % (N/2) == 0:
                err = error_rate(p_y_test, Ytest)
                print("{0: <6}> Cost: {1: .3f}, Error: {2: .3f}".format(i,ll,err))
    
    p_y = forward(Xtest, W, b)
    print("Final error rate: {0: .3f}".format(error_rate(p_y, Ytest)))
    print("Elapsed Time - Stochastic GD: {}".format(datetime.now() - t0))

    # 3) Batch GD
    W = np.random.randn(D, 10) / 28
    b = np.zeros(10)

    learn_rate = 10e-5
    reg = 10e-3
    LL_batch = []
    batch_size = 500
    nb_batches = int(N / batch_size)

    t0 = datetime.now()
    for i in range(50):
        tmp_x, tmp_y = shuffle(Xtrain, Ytrain_ind)
        for j in range(nb_batches):
            x = tmp_x[j*batch_size:(j*batch_size + batch_size),:]
            y = tmp_y[j*batch_size:(j*batch_size + batch_size),:]
            p_y = forward(x, W, b)

            W += learn_rate * (gradW(y, p_y, x) - reg*W)
            b += learn_rate * (gradb(y, p_y) - reg*b)

            p_y_test = forward(Xtest, W, b)
            ll = cost(p_y_test, Ytest_ind)
            LL_batch.append(ll)

            if j % int(nb_batches/2) == 0:
                err = error_rate(p_y_test, Ytest)
                print("{0: <6}> Cost: {1: .3f}, Error: {2: .3f}".format(i,ll,err))
    
    p_y = forward(Xtest, W, b)
    print("Final error rate: {0: .3f}".format(error_rate(p_y, Ytest)))
    print("Elapsed Time - Batch GD: {}".format(datetime.now() - t0))


    # Plotting...
    x1 = np.linspace(0, 1, len(LL))
    plt.plot(x1, LL, label="full")
    x2 = np.linspace(0, 1, len(LL_stochastic))
    plt.plot(x2, LL_stochastic, label="stochastic")
    x3 = np.linspace(0, 1, len(LL_batch))
    plt.plot(x3, LL_batch, label="batch")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
