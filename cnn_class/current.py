import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.utils import shuffle


def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def error_rate(p, t):
    return np.mean(p != t)


def flatten(X):
    """Flattens MATLAB image file data to 1-D tensors

    MATLAB image format stores row number at the last index
    """
    N = X.shape[-1]
    flat = np.zeros((N, 3072))
    for idx, i in enumerate(range(N)):
        # if not idx:
            # print(X[:,:,:,i].reshape(3072))
        flat[i] = X[:,:,:,i].reshape(3072)
    return flat


def main():
    train = loadmat('/data/datasets/image/svhn/train_32x32.mat')
    test = loadmat('/data/datasets/image/svhn/test_32x32.mat')

    X = flatten(train['X'].astype(np.float32) / 255.0)
    Y = train['y'].flatten() - 1  # MATLAB indexes at 1
    X, Y = shuffle(X, Y)
    Y_ind = y2indicator(Y)

    Xval = flatten(test['X'].astype(np.float32) / 255.0)
    Yval = test['y'].flatten() - 1 # MATLAB indexes at 1
    Yval_ind = y2indicator(Yval)

    max_iter = 20
    print_per = 10

    N, D = X.shape
    batch_size = 100
    nb_batches = N // batch_size

    M1 = 1000
    M2 = 500
    K = 10

    W1_init = np.random.randn(D, M1) / np.sqrt(D + M1)
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2 + K)
    b3_init = np.zeros(K)

    tf_X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.placeholder(tf.float32, shape=(None, K), name='T')

    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    Z1 = tf.nn.relu(tf.matmul(tf_X, W1) + b1)
    Z2 = tf.nn.relu(tf.matmul(Z1, W2) + b2)
    YZ = tf.matmul(Z2, W3) + b3

    cost_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=YZ, labels=T))
    train_op = tf.train.AdamOptimizer(0.0001).minimize(cost_op)
    predict_op = tf.argmax(YZ, axis=1)

    LL = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(max_iter):
            for j in range(nb_batches):
                split = j*batch_size
                X_batch = X[split:(split+batch_size),]
                Y_batch = Y_ind[split:(split+batch_size),]

                sess.run(train_op, feed_dict={tf_X: X_batch, T: Y_batch})

                if j % print_per == 0:
                    cost = sess.run(cost_op, feed_dict={tf_X: Xval, T: Yval_ind})
                    LL.append(cost)
                    pred = sess.run(predict_op, feed_dict={tf_X: Xval})
                    print(pred[0], Yval[0], pred[-1], Yval[-1])
                    err = error_rate(pred, Yval)
                    print("[{0:3}] {1:5}/{2:} - C: {3:.3f} | E: {4:.3f}".format(i, j, nb_batches, cost, err))

    plt.plot(LL)
    plt.plot(err)
    plt.show()


if __name__ == '__main__':
    main()
