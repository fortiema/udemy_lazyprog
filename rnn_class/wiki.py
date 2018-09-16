from datetime import datetime
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
import theano
import theano.tensor as T

from gru import GRU
from lstm import LSTM
from util import init_weight, get_wikipedia_data


class RNN:
    def __init__(self, D, hidden_layer_sizes, V):
        self.hls = hidden_layer_sizes
        self.D = D
        self.V = V

    def fit(
        self, X, lr=10e-5, mu=0.99, epochs=10, show_fig=True, activation=T.nnet.relu, unit=GRU, normalize_embed=True
    ):
        D = self.D
        V = self.V
        N = len(X)

        We = init_weight(V, D)
        self.hl = []
        Mi = D
        for Mo in self.hls:
            ru = unit(Mi, Mo, activation)
            self.hl.append(ru)
            Mi = Mo

        Wo = init_weight(Mi, V)
        bo = np.zeros(V)

        self.We = theano.shared(We)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.Wo, self.bo]
        for ru in self.hl:
            self.params += ru.params

        thX = T.ivector("X")
        thY = T.ivector("Y")

        Z = self.We[thX]
        for ru in self.hl:
            Z = ru.output(Z)
        py_x = T.nnet.softmax(Z.dot(self.Wo) + self.bo)

        pred = T.argmax(py_x, axis=1)
        self.predict_op = theano.function(inputs=[thX], outputs=[py_x, pred], allow_input_downcast=True)

        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY]))
        grads = T.grad(cost, self.params)
        dparams = [theano.shared(p.get_value() * 0) for p in self.params]

        dWe = theano.shared(self.We.get_value() * 0)
        gWe = T.grad(cost, self.We)
        dWe_update = mu * dWe - lr * gWe
        We_update = self.We + dWe_update
        if normalize_embed:
            We_update /= We_update.norm(2)

        updates = (
            [(p, p + mu * dp - lr * g) for p, dp, g in zip(self.params, dparams, grads)]
            + [(dp, mu * dp - lr * g) for dp, g in zip(dparams, grads)]
            + [(self.We, We_update), (dWe, dWe_update)]
        )

        self.train_op = theano.function(inputs=[thX, thY], outputs=[cost, pred], updates=updates)

        costs = []
        for i in range(epochs):
            t0 = datetime.now()
            X = shuffle(X)
            n_correct, n_total, cost = 0, 0, 0

            for j in range(N):
                if np.random.random() < 0.01 or len(X[j]) <= 1:
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

                if j % 200 == 0:
                    sys.stdout.write("j/N: %d/%d correct rate so far: %f\r" % (j, N, float(n_correct) / n_total))
                    sys.stdout.flush()

            print(f"i: {i}; cost: {cost}; accuracy: {float(n_correct)/n_total}")

        if show_fig:
            plt.plot(costs)
            plt.show()


def train_wiki(we_file="embeddings.npy", w2i_file="dict.json", rnn_uni=GRU):
    print("Preprocessing...")
    sent, w2i, i2w = get_wikipedia_data(n_entries=1000, n_vocab=2000)
    print(f"Done. Vocab Size: {len(w2i)}")

    model = RNN(50, [50], len(w2i))
    model.fit(list(sent), lr=10e-6, epochs=10)

    np.save(we_file, model.We.get_value())
    with open(w2i_file, "w") as f:
        json.dump(w2i, f)


if __name__ == "__main__":
    train_wiki()
