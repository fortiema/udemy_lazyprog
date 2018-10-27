import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Activation, BatchNormalization
from keras.layers import Dense, Dropout, GlobalMaxPooling1D, Input
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score


# Basic Constraints
MAX_SEQ_LEN = 100
MAX_VOCAB = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10


print("Loading pretrained vectors...")
word2vec = {}
with open(f"/data/datasets/text/glove.6B/glove.6B.{EMBEDDING_DIM}d.txt", "r") as f_in:
    for line in f_in:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype="float32")
        word2vec[word] = vec
print(f"Loaded {len(word2vec)} vectors.")


DATA_ROOT = "/data/datasets/kaggle/jigsaw-toxic-comment-classification-challenge"

print("Loading data...")
train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"), engine="c")
sentences = train["comment_text"].fillna("DUMMY_VALUE").values
categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targets = train[categories].values

print("Tokenizing...")
tokenizer = Tokenizer(num_words=MAX_VOCAB)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
w2i = tokenizer.word_index

print("Padding...")
data = pad_sequences(sequences, maxlen=MAX_SEQ_LEN)

print("Constructing embeddings...")
num_words = min(MAX_VOCAB, len(w2i) + 1)
embeddings = np.zeros((num_words, EMBEDDING_DIM))
for word, i in w2i.items():
    if i < MAX_VOCAB:
        vector = word2vec.get(word)
        if vector is not None:
            embeddings[i] = vector

embedding = Embedding(
    embeddings.shape[0],
    embeddings.shape[1],
    weights=[embeddings],
    input_length=MAX_SEQ_LEN,
    trainable=False
)

print("Building model...")
_input = Input(shape=(MAX_SEQ_LEN,))
x = embedding(_input)
x = Conv1D(128, 3, activation="relu")(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.25)(x)
x = Conv1D(128, 3, activation="relu")(x)
x = MaxPooling1D(3)(x)
x = Dropout(0.25)(x)
x = Conv1D(128, 3, activation="relu")(x)
x = GlobalMaxPooling1D()(x)
x = Dropout(0.25)(x)
x = Dense(128)(x)
x = BatchNormalization()(x)
x = Activation("relu")(x)
x = Dropout(0.4)(x)
_output = Dense(len(categories), activation="sigmoid")(x)

model = Model(_input, _output)
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

print("Training model...")
r = model.fit(
    data,
    targets,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
)

plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

plt.plot(r.history["acc"], label="acc")
plt.plot(r.history["val_acc"], label="val_acc")
plt.legend()
plt.show()

print("Computing ROC AUC...")
p = model.predict(data)
aucs = []
for j in range(6):
    auc = roc_auc_score(targets[:,j], p[:,j])
    aucs.append(auc)
print(np.mean(aucs))
