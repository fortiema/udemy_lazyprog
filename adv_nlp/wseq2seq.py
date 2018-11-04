""" poetry

RNN Seq2Seq Poetry Generator
"""
import os

from keras.layers import Dense, Embedding, Input, LSTM
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import numpy as np

# Basic Constraints
MAX_SEQ_LEN = 100
MAX_VOCAB = 3000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 24
NUM_SAMPLES = 10000

MODEL_FNAME = "s2s.h5"

input_texts = []
target_texts = []
target_texts_inputs = []

print("Loading data...")
t = 0
for line in open("/data/datasets/text/nmt/spa-eng/spa.txt"):
    t += 1
    if t > NUM_SAMPLES:
        break

    line = line.rstrip()

    if "\t" not in line:
        continue

    input_text, translation = line.split("\t")

    target_text_input = "<bos> " + translation
    target_line = translation + " <eos>"

    input_texts.append(input_text)
    target_texts_inputs.append(target_text_input)
    target_texts.append(target_line)


print("Numericalizing Source...")
tok_src = Tokenizer(num_words=MAX_VOCAB)
tok_src.fit_on_texts(input_texts)
input_sequences = tok_src.texts_to_sequences(input_texts)

print("Building Source vocab...")
word2idx_src = tok_src.word_index
print(f"Unique tokens (Source): {len(word2idx_src)}")

print("Numericalizing Target...")
tok_tgt = Tokenizer(num_words=MAX_VOCAB, filters="")
tok_tgt.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tok_tgt.texts_to_sequences(target_texts)
target_sequences_inputs = tok_tgt.texts_to_sequences(target_texts_inputs)

print("Building Target vocab...")
word2idx_tgt = tok_tgt.word_index
num_words_tgt = len(word2idx_tgt) + 1
print(f"Unique tokens (Target): {len(word2idx_tgt)}")

print("Padding...")
max_len_src = max(len(s) for s in input_sequences)
print(f"Max sequence length (Source): {max_len_src}")
max_len_tgt = max(len(s) for s in target_sequences)
print(f"Max sequence length (Target): {max_len_tgt}")
encoder_in_seq = pad_sequences(input_sequences, maxlen=max_len_src)
decoder_in_seq = pad_sequences(target_sequences_inputs, maxlen=max_len_tgt, padding="post")
decoder_tgt_seq = pad_sequences(target_sequences, maxlen=max_len_tgt, padding="post")
print(f"Shape of data tensor: {encoder_in_seq.shape}")

if not os.path.exists(MODEL_FNAME):
    print("Loading pretrained vectors...")
    word2vec = {}
    with open(f"/data/datasets/text/glove.6B/glove.6B.{EMBEDDING_DIM}d.txt", "r") as f_in:
        for line in f_in:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype="float32")
            word2vec[word] = vec
    print(f"Loaded {len(word2vec)} vectors.")

    print("Constructing embeddings...")
    num_words = min(MAX_VOCAB, len(word2idx_src) + 1)
    embeddings = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word2idx_src.items():
        if i < MAX_VOCAB:
            vector = word2vec.get(word)
            if vector is not None:
                embeddings[i] = vector

    encoder_embed = Embedding(
        embeddings.shape[0],
        embeddings.shape[1],
        weights=[embeddings],
        input_length=None,
        name="encoder_embedding",
        # trainable=False
    )

    print("Building model...")
    encoder_input = Input(shape=(max_len_src,), name="encoder_input")
    x = encoder_embed(encoder_input)
    encoder = LSTM(LATENT_DIM, dropout=0.3, recurrent_dropout=0.3, return_state=True, name="encoder_lstm")
    encoder_output, h, c = encoder(x)
    encoder_states = [h, c]

    decoder_input = Input(shape=(max_len_tgt,), name="decoder_input")
    decoder_embed = Embedding(num_words_tgt, LATENT_DIM, name="decoder_embedding")
    decoder = LSTM(
        LATENT_DIM, dropout=0.3, recurrent_dropout=0.3, return_sequences=True, return_state=True, name="decoder_lstm"
    )
    decoder_output, _, _ = decoder(decoder_embed(decoder_input), initial_state=encoder_states)

    decoder_dense = Dense(num_words_tgt, activation="softmax", name="decoder_dense")
    decoder_out = decoder_dense(decoder_output)

    model = Model([encoder_input, decoder_input], decoder_out)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(lr=0.01))

    print("Training model...")
    r = model.fit(
        [encoder_in_seq, decoder_in_seq],
        np.expand_dims(decoder_tgt_seq, -1),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
    )

    plt.plot(r.history["loss"], label="loss")
    plt.plot(r.history["val_loss"], label="val_loss")
    plt.legend()
    plt.show()

    print("Saving model...")
    model.save(MODEL_FNAME)
else:
    model = load_model(MODEL_FNAME)

print("Building generator...")
encoder_input = Input(shape=(max_len_src,))
encoder_emb = model.get_layer("encoder_embedding")(encoder_input)
_, h, c = model.get_layer("encoder_lstm")(encoder_emb)
encoder_states = [h, c]
encoder = Model(encoder_input, encoder_states)

decoder_in_h = Input(shape=(LATENT_DIM,))
decoder_in_c = Input(shape=(LATENT_DIM,))
decoder_states_input = [decoder_in_h, decoder_in_c]

decoder_input = Input(shape=(1,))
decoder_emb = model.get_layer("decoder_embedding")(decoder_input)
decoder_output, h, c = model.get_layer("decoder_lstm")(decoder_emb, initial_state=decoder_states_input)
decoder_states = [h, c]
decoder_out = model.get_layer("decoder_dense")(decoder_output)

decoder = Model([decoder_input] + decoder_states_input, [decoder_out] + decoder_states)

idx2word_src = {v: k for k, v in word2idx_src.items()}
idx2word_tgt = {v: k for k, v in word2idx_tgt.items()}


def translate(input_seq):
    states = encoder.predict(input_seq)

    tgt_seq = np.zeros((1, 1))
    tgt_seq[0, 0] = word2idx_tgt.get("<bos>")
    eos = word2idx_tgt.get("<eos>")

    output_sentence = []
    for _ in range(max_len_tgt):
        o, h, c = decoder.predict([tgt_seq] + states)

        idx = np.argmax(o[0, 0, :])
        if idx == eos:
            break

        if idx > 0:
            output_sentence.append(idx2word_tgt.get(idx))

        tgt_seq[0, 0] = idx
        states = [h, c]

    return " ".join(output_sentence)


while True:
    i = np.random.choice(len(input_texts))
    input_seq = encoder_in_seq[i : i + 1]
    translation = translate(input_seq)
    print("---")
    print(f"Input: {input_texts[i]}")
    print(f"Translation: {translation}")

    ans = input("Continue? [Y/n]")
    if ans and ans.lower().startswith("n"):
        break
