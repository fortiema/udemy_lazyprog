import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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

    M1 = 300
    M2 = 100
    K = 10

    W1_init = (np.random.randn(D, M1) / 28).astype(np.float32)
    b1_init = np.zeros(M1).astype(np.float32)
    W2_init = (np.random.randn(M1, M2) / np.sqrt(M1)).astype(np.float32)
    b2_init = np.zeros(M2).astype(np.float32)
    W3_init = (np.random.randn(M2, K) / np.sqrt(M2)).astype(np.float32)
    b3_init = np.zeros(K).astype(np.float32)

    X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')

    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    Z1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
    YZ = tf.matmul(Z2, W3) + b3

    cost_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=YZ, labels=T))
    train_op = tf.train.RMSPropOptimizer(learn_rate, decay=0.99, momentum=0.9).minimize(cost_op)
    predict_op = tf.argmax(YZ, 1)

    LL = []
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(max_iter):
            for j in range(nb_batches):
                slice_start = j * batch_size
                slice_end = slice_start + batch_size
                Xbatch = Xtrain[slice_start:slice_end,]
                Ybatch = Ytrain_ind[slice_start:slice_end,]

                sess.run(train_op, feed_dict={X: Xbatch, T: Ybatch})

                if j % print_period == 0:
                    cost = sess.run(cost_op, feed_dict={X: Xtest, T: Ytest_ind})
                    LL.append(cost)
                    pred = sess.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(pred, Ytest)
                    print("{0: <6}-{1: <2}> Cost: {2: .3f}, Error: {3: .3f}".format(i, j, cost, err))

    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    main()
