import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

from util import get_normalized_data, y2indicator


def error_rate(p, t):
    return np.mean(p != t)


def main():
    X, Y = get_normalized_data('../large_files/train.csv')

    max_iter = 20
    print_period = 10

    learn_rate = 1*10e-5
    reg = 10e-4

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:,]
    Ytest = Y[-1000:]
    Ytrain_ind = y2indicator(Ytrain).astype(np.float32)
    Ytest_ind = y2indicator(Ytest).astype(np.float32)

    N, D = Xtrain.shape
    batch_size = 500
    nb_batches = int(N / batch_size)

    M = 300
    K = 10

    W1_init = (np.random.randn(D, M) / 28).astype(np.float32)
    b1_init = np.zeros(M).astype(np.float32)
    W2_init = (np.random.randn(M, K) / np.sqrt(M)).astype(np.float32)
    b2_init = np.zeros(K).astype(np.float32)

    t_X = T.matrix('X')
    t_T = T.matrix('K')
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')

    t_Z = T.nnet.relu(t_X.dot(W1) + b1)
    t_Y = T.nnet.softmax(t_Z.dot(W2) + b2)

    cost = -(t_T * T.log(t_Y)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())
    pred = T.argmax(t_Y, axis=1)

    upd_W1 = W1 - learn_rate*T.grad(cost, W1)
    upd_b1 = b1 - learn_rate*T.grad(cost, b1)
    upd_W2 = W2 - learn_rate*T.grad(cost, W2)
    upd_b2 = b2 - learn_rate*T.grad(cost, b2)

    train = theano.function(
        inputs=[t_X, t_T],
        updates=[(W1, upd_W1), (b1, upd_b1), (W2, upd_W2), (b2, upd_b2)]
    )

    get_pred = theano.function(
        inputs=[t_X, t_T],
        outputs=[cost, pred]
    )

    LL = []
    for i in range(max_iter):
        for j in range(nb_batches):
            slice_start = j * batch_size
            slice_end = slice_start + batch_size
            Xbatch = Xtrain[slice_start:slice_end,]
            Ybatch = Ytrain_ind[slice_start:slice_end,]

            train(Xbatch, Ybatch)

            if j % print_period == 0:
                cost_val, pred_val = get_pred(Xtest, Ytest_ind)
                LL.append(cost_val)
                err = error_rate(pred_val, Ytest)
                print("{0: <6}-{1: <2}> Cost: {2: .3f}, Error: {3: .3f}".format(i, j, float(cost_val.flat[0]), err))

    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    main()