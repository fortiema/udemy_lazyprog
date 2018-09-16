from datetime import datetime
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf

from util import init_weight, all_parity_pairs


LOGS_PATH = "logs/rnn_class/mlp_parity"


class HiddenLayer:
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W = init_weight(M1, M2)
        b = np.zeros(M2)
        self.W = tf.Variable(W.astype(np.float32), name=f"W_{self.id}")
        self.b = tf.Variable(b.astype(np.float32), name=f"b_{self.id}")

        self.params = [self.W, self.b]

    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)


class ANN:
    def __init__(self, hid_layer_sizes):
        self.hid_layer_sizes = hid_layer_sizes

    def fit(self, X, Y, lr=10e-3, reg=10e-12, epochs=400, batch_size=20, print_per=1, plot_cost=False):
        Y = Y.astype(np.int32)
        # Yind = y2indicator(Y)

        N, D = X.shape
        K = len(set(Y))
        self.hid_layers = []
        M1 = D
        count = 0

        for M2 in self.hid_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hid_layers.append(h)
            M1 = M2
            count += 1

        W = init_weight(M1, K)
        b = np.zeros(K)
        self.W = tf.Variable(W.astype(np.float32), name="W_logreg")
        self.b = tf.Variable(b.astype(np.float32), name="b_logreg")

        self.params = [self.W, self.b]
        for h in self.hid_layers:
            self.params += h.params

        tf_X = tf.placeholder(tf.float32, shape=(batch_size, D), name="X")
        tf_Y = tf.placeholder(tf.int64, shape=(batch_size,), name="Y")
        pY = self.forward(tf_X)

        cost_op = (
            tf.reduce_sum(tf.losses.sparse_softmax_cross_entropy(logits=pY, labels=tf_Y))
            + tf.losses.get_regularization_loss()
        )
        train_op = tf.train.AdamOptimizer(lr).minimize(cost_op)
        predict_op = self.predict(tf_X)

        with tf.name_scope("Accuracy"):
            # Accuracy
            correct_prediction = tf.equal(tf.argmax(pY, 1), tf_Y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # create a summary for our cost and accuracy
        tf.summary.scalar("cost", cost_op)
        tf.summary.scalar("accuracy", accuracy)

        summary_op = tf.summary.merge_all()

        n_batches = N // batch_size
        costs = []

        start = datetime.now()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            writer = tf.summary.FileWriter(
                os.path.join(LOGS_PATH, str(int(time.time()))), graph=tf.get_default_graph()
            )

            for i in range(epochs):
                X, Y = shuffle(X, Y)
                for j in range(n_batches):
                    Xbatch = X[j * batch_size:(j * batch_size + batch_size)]
                    Ybatch = Y[j * batch_size:(j * batch_size + batch_size)]

                    # if not j:
                    #     print(Xbatch)
                    #     print(Ybatch)

                    _, summary = sess.run([train_op, summary_op], feed_dict={tf_X: Xbatch, tf_Y: Ybatch})
                    writer.add_summary(summary, i * n_batches + j)

                    if j % print_per == 0:
                        c = 0
                        pred = np.zeros(len(Xbatch))

                        c += sess.run(cost_op, feed_dict={tf_X: Xbatch, tf_Y: Ybatch})
                        pred = sess.run(predict_op, feed_dict={tf_X: Xbatch})

                        costs.append(c)
                        e = np.mean(Ybatch != pred)
                        print(f"i: {i}; nb: {n_batches}; cost: {c}; error_rate: {e}")

                writer.flush()

        print(f"Done! Total training time: {datetime.now()-start}s")

        if plot_cost:
            plt.plot(costs)
            plt.show()

    def forward(self, X):
        Z = X
        for h in self.hid_layers:
            Z = h.forward(Z)
        return tf.nn.softmax(tf.matmul(Z, self.W) + self.b)

    def predict(self, X):
        pY = self.forward(X)
        return tf.argmax(pY, axis=1)


def wide():
    X, Y = all_parity_pairs(12)
    model = ANN([2048])
    model.fit(X, Y, lr=10e-5, epochs=300, print_per=10, plot_cost=True)


def deep():
    X, Y = all_parity_pairs(12)
    model = ANN([1024] * 2)
    model.fit(X, Y, lr=10e-4, epochs=100, print_per=10, plot_cost=True)


if __name__ == "__main__":
    deep()
