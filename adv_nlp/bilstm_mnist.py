""" bilstm_mnist

Example application of RNN networks to image classification problems.
"""

import os

import keras.backend as K
from keras.layers import Bidirectional, Concatenate, Dense, GlobalMaxPooling1D, Input, Lambda, LSTM
from keras.models import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_mnist(limit=None):
    if not os.path.exists("./large_files/mnist_train.csv"):
        print("Unable to find data source!")

    print("Loading data...")
    df = pd.read_csv("./large_files/mnist_train.csv", engine="c")
    data = df.values()
    np.random.shuffle(data)
    X = data[:, 1:].reshape(-1, 28, 28) / 255.0
    Y = data[:, 0]

    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


X, Y = get_mnist()

D = 28
M = 15

print("Building model...")
_in = Input(shape=(D, D))
rnn1 = Bidirectional(LSTM(M, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(_in)
rnn1 = GlobalMaxPooling1D()(rnn1)
rnn2 = Lambda(lambda t: K.permute_dimensions(t, pattern=(0, 2, 1)))(_in)
rnn2 = Bidirectional(LSTM(M, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(rnn2)
rnn2 = GlobalMaxPooling1D()(rnn2)
x = Concatenate(axis=1)([rnn1, rnn2])
_out = Dense(10, activation="softmax")(x)
model = Model(_in, _out)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])

print("Training model...")
r = model.fit(X, Y, batch_size=32, epochs=10, validation_split=0.25)

plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

plt.plot(r.history["acc"], label="acc")
plt.plot(r.history["val_acc"], label="val_acc")
plt.legend()
plt.show()
