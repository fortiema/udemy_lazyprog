import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class HMM:
    def __init__(self, M, V, pi=None, a=None, b=None):
        self.M = M

        pi = pi or np.zeros(self.M).astype(np.float32)
        a = a or np.random.randn(self.M, self.M).astype(np.float32)
        b = b or np.random.randn(self.M, V).astype(np.float32)

        self.pre_softmax_pi = tf.Variable(pi)
        self.pre_softmax_a = tf.Variable(a)
        self.pre_softmax_b = tf.Variable(b)

        # Applying softmax ensures params are probability dists!
        pi = tf.nn.softmax(self.pre_softmax_pi)
        a = tf.nn.softmax(self.pre_softmax_a)
        b = tf.nn.softmax(self.pre_softmax_b)

        self.tfx = tf.placeholder(tf.int32, shape=(None,), name='x')
        def recurrence(old_a_old_s, x_t):
            nonlocal a
            # Reshape alpha into 2D to be able to perform tf.matmul
            old_a = tf.reshape(old_a_old_s[0], (1, M))
            a = tf.matmul(old_a, a) * b[:, x_t]
            a = tf.reshape(a, (M,))
            s = tf.reduce_sum(a)
            return (a / s), s

        alpha, scale = tf.scan(
            fn=recurrence,
            elems=self.tfx[1:],
            initializer=(pi*b[:, self.tfx[0]], np.float32(1.0))
        )

        self.cost = -tf.reduce_sum(tf.log(scale))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.cost)

    def set_session(self, session):
        self.session = session

    def fit(self, X, lr=0.001, epochs=10, pper=1):
        N = len(X)

        costs = []
        for e in range(epochs):
            if e % pper == 0:
                print('Iteration {}:'.format(e))

            for n in range(N):
                c = self.get_cost_multi(X).sum()
                costs.append(c)
                self.session.run(self.train_op, feed_dict={self.tfx: X[n]})

        plt.plot(costs)
        plt.show()

    def get_cost(self, x):
        return self.session.run(self.cost, feed_dict={self.tfx: x})

    def log_likelihood(self, x):
        """Negative LL"""
        return -self.session.run(self.cost, feed_dict={self.tfx: x})

    def get_cost_multi(self, X):
        P = np.random.random(len(X))
        return np.array([self.get_cost(x) for x, p in zip(X, P)])

    def set_param(self, pi, a, b):
        op1 = self.pre_softmax_pi.assign(pi)
        op2 = self.pre_softmax_a.assign(a)
        op3 = self.pre_softmax_b.assign(b)
        self.session.run([op1, op2, op3])


def fit_coin():
    X = []
    for line in open('large_files/coin_data.txt'):
        x = [1 if e == 'H' else 0 for e in line.rstrip()]
        X.append(x)

    # Let the model learn what the most likely params are
    hmm = HMM(2, 2)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        hmm.set_session(sess)
        hmm.fit(X, epochs=5)
        L = hmm.get_cost_multi(X).sum()
        print('Log-L /w fit: {}'.format(L))

    # Try passing the known (true) parameters
    hmm = HMM(2, 2)
    pi = np.log(np.array([.5, .5])).astype(np.float32)
    a = np.log(np.array([[.1, .9], [.8, .2]])).astype(np.float32)
    b = np.log(np.array([[.6, .4], [.3, .7]])).astype(np.float32)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        hmm.set_session(sess)
        hmm.set_param(pi, a, b)
        L = hmm.get_cost_multi(X).sum()
        print('Log-L /w true: {}'.format(L))


if __name__ == '__main__':
    fit_coin()
