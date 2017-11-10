from datetime import datetime
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.utils import shuffle

LOGS_PATH='logs/cnn_class/tf_conv'


def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def error_rate(p, t):
    return np.mean(p != t)


def init_filter(shape, pool_size):
    # Fan in -> nb_feat_in * width * height
    # Fan out -> nb_feat_out * width * height / pooling downsample 
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(pool_size)))
    return w.astype(np.float32)


def convpool(X, W, b):
    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    conv_out = tf.nn.bias_add(conv_out, b)
    pool_out = tf.nn.max_pool(
        conv_out,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME'
    )
    return pool_out


def format_data(X):
    """Converts MATLAB image data to a more firendly representation"""
    N = X.shape[-1]
    out = np.zeros((N, 32, 32, 3), dtype=np.float32)
    for i in range(N):
        for j in range(3):
            out[i, :, :, j] = X[:, :, j, i]
    return out / 255.0
    

def main():
    train = loadmat('/data/datasets/image/svhn/train_32x32.mat')
    test = loadmat('/data/datasets/image/svhn/test_32x32.mat')

    X = format_data(train['X'])
    Y = train['y'].flatten() - 1  # MATLAB indexes at 1
    del train
    X, Y = shuffle(X, Y)
    Y_ind = y2indicator(Y)

    Xval = format_data(test['X'])
    Yval = test['y'].flatten() - 1 # MATLAB indexes at 1
    del test
    Yval_ind = y2indicator(Yval)

    max_iter = 50
    print_per = 25

    N = X.shape[0]
    W, H, D = X.shape[1:]
    batch_size = 64
    nb_batches = N // batch_size

    M = 500
    K = 10

    # If not enough RAM to handle variable input size, make it constant
    # X = X[:73000,]
    # X = Y[:73000]
    # Xval = Xval[:26000,]
    # Yval = Yval[:26000]
    # Yval_ind = Yval_ind[:26000,]

    W1_shape = (5, 5, 3, 20)
    W1_init = init_filter(W1_shape, (2, 2))
    b1_init = np.zeros(W1_shape[-1], dtype=np.float32)

    W2_shape = (5, 5, 20, 50)
    W2_init = init_filter(W2_shape, (2, 2))
    b2_init = np.zeros(W2_shape[-1], dtype=np.float32)

    W3_init = np.random.randn(W2_shape[-1]*8*8, M) / np.sqrt(W2_shape[-1]*8*8 + M)
    b3_init = np.zeros(M, dtype=np.float32)
    W4_init = np.random.randn(M, K) / np.sqrt(M + K)
    b4_init = np.zeros(K, dtype=np.float32)

    tf_X = tf.placeholder(tf.float32, shape=(batch_size, W, H, D), name='X')
    T = tf.placeholder(tf.float32, shape=(batch_size, K), name='T')

    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))
    W4 = tf.Variable(W4_init.astype(np.float32))
    b4 = tf.Variable(b4_init.astype(np.float32))

    Z1 = convpool(tf_X, W1, b1)
    Z2 = convpool(Z1, W2, b2)

    Z2_shape = Z2.get_shape().as_list()
    Z2r = tf.reshape(Z2, shape=[Z2_shape[0], np.prod(Z2_shape[1:])])
    Z3 = tf.nn.relu(tf.matmul(Z2r, W3) + b3)
    YZ = tf.matmul(Z3, W4) + b4

    cost_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=YZ, labels=T))
    train_op = tf.train.AdamOptimizer(0.0001).minimize(cost_op)
    predict_op = tf.argmax(YZ, axis=1)

    with tf.name_scope('Accuracy'):
        # Accuracy
        correct_prediction = tf.equal(tf.argmax(YZ,1), tf.argmax(T,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create a summary for our cost and accuracy
    tf.summary.scalar("cost", cost_op)
    tf.summary.scalar("accuracy", accuracy)

    summary_op = tf.summary.merge_all()

    start = datetime.now()
    LL = []
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        writer = tf.summary.FileWriter(
            os.path.join(LOGS_PATH, str(int(time.time()))),
            graph=tf.get_default_graph()
        )

        for i in range(max_iter):
            for j in range(nb_batches):
                split = j*batch_size
                X_batch = X[split:(split+batch_size),]
                Y_batch = Y_ind[split:(split+batch_size),]

                # Last batch is maybe not full and would crash the network!
                if len(X_batch) == batch_size:
                    _, summary = sess.run([train_op, summary_op], feed_dict={tf_X: X_batch, T: Y_batch})

                    writer.add_summary(summary, i * nb_batches + j)

                    if j % print_per == 0:
                        cost = 0
                        pred = np.zeros(len(Xval))

                        for k in range(len(Xval) // batch_size):
                            Xval_batch = Xval[k*batch_size:(k*batch_size + batch_size), ]
                            Yval_batch = Yval_ind[k*batch_size:(k*batch_size + batch_size), ]

                            cost += sess.run(cost_op, feed_dict={tf_X: Xval_batch, T: Yval_batch})
                            pred[k*batch_size:(k*batch_size + batch_size)] = sess.run(predict_op, feed_dict={tf_X: Xval_batch})

                        LL.append(cost)
                        print(pred[0], Yval[0], pred[-1], Yval_batch[-1])
                        err = error_rate(pred, Yval)
                        print("[{0:3}] {1:5}/{2:} - {3:} - C: {4:.3f} | E: {5:.3f}".format(i, j, nb_batches, datetime.now()-start, cost, err))

            writer.flush()

    print('Done! Total training time: {}s'.format(datetime.now()-start))

    plt.plot(LL)
    plt.plot(err)
    plt.show()


if __name__ == '__main__':
    main()
