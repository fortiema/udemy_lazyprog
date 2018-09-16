import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import theano
import theano.tensor as T

from util import init_weight, get_robert_frost


class SimpleLMRRNN:
    """A simple Language Model using SRN"""

    def __init__(self, D, M, V):
        self.D = D
        self.M = M
        self.V = V

    def fit(self, X, epochs=200, lr=10e-1, mu=0.99, reg=1.0, show_fig=False):
        N = len(X)
        D = self.D
        M = self.M
        V = self.V

        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)
        Wxz = init_weight(D, M)
        Whz = init_weight(M, M)
        bz = np.ones(M)
        Wo = init_weight(M, V)
        bo = np.zeros(V)

        thX, thY, py_x, pred = self.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value() * 0) for p in self.params]

        updates = [
            (p, p + mu * dp - lr * g) for p, dp, g in zip(self.params, dparams, grads)
        ] + [
            (dp, mu * dp - lr * g) for dp, g in zip(dparams, grads)
        ]

        self.pred_op = theano.function(inputs=[thX], outputs=pred, allow_input_downcast=True)
        self.train_op = theano.function(inputs=[thX, thY], outputs=[cost, pred], updates=updates)

        costs = []
        n_total = sum((len(seq) + 1) for seq in X)
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            n_total = 0
            cost = 0

            for j in range(N):
                # Only go all the way to EOS 10% of the time
                if np.random.random() < 0.1:
                    input_seq = [0] + X[j]
                    output_seq = X[j] + [1]
                else:
                    input_seq = [0] + X[j][:-1]
                    output_seq = X[j]
                n_total += len(output_seq)

                c, p = self.train_op(input_seq, output_seq)
                cost += c
                for pj, xj in zip(p, output_seq):
                    if pj == xj:
                        n_correct += 1
            print(f"i: {i}; cost: {cost}; accuracy: {float(n_correct) / n_total}")
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])

    @staticmethod
    def load(filename):
        npz = np.load(filename)
        We = npz["arr_0"]
        Wx = npz["arr_1"]
        Wh = npz["arr_2"]
        bh = npz["arr_3"]
        h0 = npz["arr_4"]
        Wxz = npz["arr_5"]
        Whz = npz["arr_6"]
        bz = npz["arr_7"]
        Wo = npz["arr_8"]
        bo = npz["arr_9"]

        V, D = We.shape
        _, M = Wx.shape

        model = SimpleLMRRNN(D, M, V)
        model.set(We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo)

        return model

    def set(self, We, Wx, Wh, bh, h0, Wxz, Whz, bz, Wo, bo):
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wxz = theano.shared(Wxz)
        self.Whz = theano.shared(Whz)
        self.bz = theano.shared(bz)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wxz, self.Whz, self.bz, self.Wo, self.bo]

        thX = T.ivector("X")
        Ei = self.We[thX]
        thY = T.ivector("Y")

        def recurrence(x_t, h_t1):
            hhat_t = T.nnet.relu(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            z_t = T.nnet.sigmoid(x_t.dot(self.Wxz) + h_t1.dot(self.Whz) + self.bz)
            h_t = (1 - z_t) * h_t1 + z_t * hhat_t
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=Ei,
            n_steps=Ei.shape[0]
        )

        py_x = y[:, 0, :]
        pred = T.argmax(py_x, axis=1)
        self.pred_op = theano.function(
            inputs=[thX],
            outputs=[py_x, pred],
            allow_input_downcast=True
        )

        return thX, thY, py_x, pred

    def generate(self, word2idx):
        idx2word = {v: k for k, v in word2idx.items()}
        V = len(word2idx)
        n_lines = 0

        X = [0]
        while n_lines < 4:
            # Get distribution over next , sample from it
            PY_X, _ = self.pred_op(X)
            PY_X = PY_X[-1].flatten()
            P = [np.random.choice(V, pi=P)]
            # Append choice to sequence
            X = np.concatenate([X, P])

            if P > 1:
                word = idx2word[P]
                print(word + "\n")
            elif P == 1:
                n_lines += 1
                X = [0]
                print("\n")


def train_poetry():
    sents, w2i = get_robert_frost()
    model = SimpleLMRRNN(50, 50, len(w2i))
    model.fit(sents, lr=10e-5, show_fig=True, epochs=500)
    model.save("RRNN_D50_M50_epochs2000_relu.npz")


def generate_poetry():
    sents, w2i = get_robert_frost()
    model = SimpleLMRRNN.load("RRNN_D50_M50_epochs2000_relu.npz")
    model.generate(w2i)


if __name__ == "__main__":
    train_poetry()
    generate_poetry()
