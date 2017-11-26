"""Hidden Markov Model for discrete classification

Learns the writing style from 2 different writers (Robert Frost & Edgar Allan Poe).
Model is then able to predict who is more likely to have written new sentences.
"""
import string

from cytoolz import itertoolz
import matplotlib.pyplot as plt
import numpy as np
from pomegranate import *
from sklearn.utils import shuffle
import spacy


nlp = spacy.load('en_core_web_sm', ner=False)


class HMMClassifier():

    def __init__(self):
        self.models = []

    def fit(self, X, Y):
        X = np.array(X)
        Y = np.array(Y, dtype=np.int32)

        self.models.append(HiddenMarkovModel.from_samples(
            DiscreteDistribution,
            n_components=2,
            X=X[np.where(Y == [0])[0]],
            name='robert',
            stop_threshold=1e-8,
            max_iterations=30,
            n_jobs=8
        ))

        self.models.append(HiddenMarkovModel.from_samples(
            DiscreteDistribution,
            n_components=2,
            X=X[np.where(Y == [1])[0]],
            name='edgar',
            stop_threshold=1e-8,
            max_iterations=30,
            n_jobs=8
        ))

    def most_likely_author(self, seq):
        return np.argmax([m.log_probability(seq) for m in self.models])

    def eval(self, X, Y):
        return np.average([1 if (self.most_likely_author(x) == y[0]) else 0 for x, y in zip(X, Y)])


def get_lines(fname):
    count = 0
    with open(fname, 'r') as fin:
        for line in fin:
            if line:
                count += 1
                yield line
    print('Read {} lines from {}'.format(count, fname))


def get_data():
    vocab = {}
    current_id = 0
    X = []
    Y = []

    for fname, label in zip(('large_files/robert_frost.txt',
                             'large_files/edgar_allan_poe.txt'), (0, 1)):
        count = 0
        for batch in itertoolz.partition_all(1000, get_lines(fname)):
            for doc in nlp.pipe(batch, batch_size=1000, n_threads=8):
                tags = [token.tag_ for token in doc if token]
                if tags:
                    for tag in tags:
                        if tag and tag not in vocab:
                            vocab[tag] = current_id
                            current_id += 1

                    sequence = np.array([vocab[t] for t in tags if t], dtype=np.int32)
                    if np.any(sequence):
                        X.append(sequence)
                        Y.append([label])
                        count += 1

    print('Processed {} sentences'.format(count))
    print('Vocabulary Size: {} tokens'.format(current_id))
    return X, Y, current_id


def main():
    X, Y, V = get_data()

    X, Y = shuffle(X, Y)
    N = 20

    X_train, Y_train = X[:-N], Y[:-N]
    X_test, Y_test = X[-N:], Y[-N:]

    model = HMMClassifier()
    model.fit(X_train, Y_train)
    print('Accuracy: {}'.format(model.eval(X_test, Y_test)))

if __name__ == '__main__':
    main()
