import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import theano
import theano.tensor as T

from util import init_weight, all_parity_pairs


class SimpleRNN:
    def __init__(self, M):
        self.M = M

    def fit(self, X, Y, lr=10-1, mu=0.99, reg=10e-5, activation=T.tanh, epochs=100, show_fig=False):
        D = X[0].shape[1]
        K = len(set(Y.flatten()))
        N = len(Y)
        M = self.M
        self.act = activation

        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wo = init_weight(M, K)
        bo = np.zeros(K)

        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.fmatrix('X')
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            """Returns h(t), y(t)"""
            h_t = self.act(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=thX,
            n_steps=thX.shape[0]
        )

        py_x = y[:, 0, :]
        pred = T.argmax(py_x, axis=1)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value()*0) for p in self.params]

        updates = [
            (p, p + mu*dp - lr*g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu*dp - lr*g) for dp, g in zip(dparams, grads)
        ]

        self.predict_op = theano.function(inputs=[thX], outputs=pred)
        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, pred, y],
            updates=updates
        )

        costs = []
        for i in range(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0

            for j in range(N):
                c, p, r_out = self.train_op(X[j], Y[j])
                cost += c
                if p[-1] == Y[j, -1]:
                    n_correct += 1
            
            print(f'Y.shape: {r_out.shape}')
            print(f'i: {i}; cost: {cost}; classif rate: {float(n_correct) / N}')

            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()


def parity(B=12, lr=10e-5, epochs=200):
    X, Y = all_parity_pairs(B)
    N, t = X.shape

    Y_t = np.zeros(X.shape, dtype=np.int32)
    for n in range(N):
        ones_count = 0
        for i in range(t):
            if X[n,i] == 1:
                ones_count += 1
            if ones_count %2 == 1:
                Y_t[n,i] = 1

    X = X.reshape(N, t, 1).astype(np.float32)

    rnn = SimpleRNN(4)
    rnn.fit(X, Y_t, lr=lr, epochs=epochs, activation=T.nnet.sigmoid, show_fig=True)


if __name__ == '__main__':
    parity()