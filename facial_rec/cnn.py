import os
import sys
import time
import uuid

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf

from layers import ConvMaxPool2D, Dense, Flatten
from util import error_rate, get_data_image, indicators, init_w_b


LOGS_PATH='logs/facial_rec/cnn'


class CNN:
    def __init__(self, input_size, output_size, conv_layers, dense_layers):
        """Constructs network per specification

        Arguments:
            input_size (Tuple): Pixel size (2D) of raw images entering network
            output_size (int): Number of classes to predict
            conv_layers (list[Tuple]): Configuration of each convolution layer
            dense_layers (list[int]): Configuration of each fully connected layer

        """
        self.input = input_size
        self.K = output_size
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers

        self.layers = []

        for idx, W in enumerate(self.conv_layers):
            h = ConvMaxPool2D(W, (2,2), layer_id='conv{}'.format(idx))
            self.layers.append(h)

        self.layers.append(Flatten())

        M1 = self.conv_layers[-1][-1] * np.prod(np.divide(self.input, len(conv_layers) * 2).astype(np.int32))
        for idx, M2 in enumerate(self.dense_layers):
            h = Dense(M1, M2, tf.nn.relu, layer_id='dense{}'.format(idx))
            self.layers.append(h)
            M1 = M2

        # Setup final layer
        out = Dense(M1, self.K, tf.nn.softmax, 'out')
        self.layers.append(out)

    def set_session(self, session):
        self.session = session

    def fit(self, X, Y, Xval, Yval, epochs=100, batch_size=128, show_fig=False):
        N = X.shape[0]
        W, H, D = X.shape[1:]
        Yval_flat = np.argmax(Yval, axis=1)

        tf_X = tf.placeholder(tf.float32, shape=(None, W, H, D), name='X')
        tf_Y = tf.placeholder(tf.int32, shape=(None, self.K), name='Y')
        YZ = self.forward(tf_X, train=True)

        # reg_cost = lr * sum([sum(tf.nn.l2_loss(p)) for p in self.params])
        cost_op = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=YZ,
                labels=tf_Y
            )
        )
        pred_op = tf.argmax(YZ, 1)
        train_op = tf.train.AdamOptimizer(10e-5).minimize(cost_op)

        with tf.name_scope('Accuracy'):
            # Accuracy
            correct_prediction = tf.equal(tf.argmax(YZ,1), tf.argmax(tf_Y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create a summary for our cost and accuracy
        train_summary = tf.summary.merge([
            tf.summary.scalar("train_cost", cost_op),
            tf.summary.scalar("train_accuracy", accuracy)
        ])
        test_summary = tf.summary.merge([
            tf.summary.scalar("test_cost", cost_op),
            tf.summary.scalar("test_accuracy", accuracy)
        ])
        writer = tf.summary.FileWriter(
            os.path.join(LOGS_PATH, str(int(time.time()))),
            graph=tf.get_default_graph()
        )

        self.session.run(tf.global_variables_initializer())
        n_batches = N // batch_size
        costs = []
        for i in range(epochs):
            for j in range(n_batches):
                split = j*batch_size
                X_batch = X[split:(split+batch_size),]
                Y_batch = Y[split:(split+batch_size),]

                _, summary = self.session.run([train_op, train_summary], feed_dict={tf_X: X_batch, tf_Y: Y_batch})
                writer.add_summary(summary, i * n_batches + j)

                if (j+1) % 100 == 0:
                    c, p, summary = self.session.run([cost_op, pred_op, test_summary], feed_dict={tf_X: Xval, tf_Y: Yval})
                    writer.add_summary(summary, i * n_batches + j)
                    costs.append(c)
                    e = error_rate(Yval_flat, p)
                    print(p[0], Yval_flat[0], p[-1], Yval_flat[-1])
                    print("[{0:3}] {1:6}/{2:} - C: {3:.3f} | E: {4:.3f}".format(i+1, j+1, n_batches, c, e))

        if show_fig:
            plt.plot(costs)
            plt.show()

    def forward(self, X, train=False):
        out = X
        for h in self.layers:
            out = h.forward(out, train)
        return out

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)

    def predict(self, X):
        act = self.forward(X)
        return tf.argmax(act, axis=1)


def main(fname):
    X, Y, D = get_data_image(fname)

    model = CNN(
        input_size=(D, D),
        output_size=7,
        conv_layers=[(5, 5, X.shape[-1], 20), (5, 5, 20, 50)],
        dense_layers=[2000, 1000, 500]
    )
    session = tf.InteractiveSession()
    model.set_session(session)

    X = X.astype(np.float32)
    Y = indicators(Y).astype(np.float32)
    Xval, Yval = X[-1000:], Y[-1000:]
    X, Y = X[:-1000], Y[:-1000]
    model.fit(X, Y, Xval, Yval, show_fig=True)


if __name__ == '__main__':
    main(sys.argv[1])
