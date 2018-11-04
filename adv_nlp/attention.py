""" attention

Seq2seq with Attention Decoding
"""
import os

import keras.backend as K
from keras.layers import (
    Bidirectional,
    Concatenate,
    Dense,
    Dot,
    Embedding,
    Input,
    Lambda,
    LSTM,
    RepeatVector,
    Softmax,
)
from keras.models import load_model, Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import matplotlib.pyplot as plt
import numpy as np


def softmax_over_time(x):
    assert K.ndim(x) > 2
    e = K.exp(x - K.max(x, axis=1, keepdims=True))
    s = K.sum(e, axis=1, keepdims=True)
    return e / s


# Basic Constraints
MAX_SEQ_LEN = 100
MAX_VOCAB = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 192
LATENT_DIM_DECODER = 192
NUM_SAMPLES = 10000

MODEL_FNAME = "s2s-att.h5"

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
    encoder = Bidirectional(
        LSTM(LATENT_DIM, dropout=0.3, recurrent_dropout=0.3, return_sequences=True), name="encoder_lstm"
    )
    encoder_output = encoder(x)

    decoder_input = Input(shape=(max_len_tgt,), name="decoder_input")
    decoder_embedding = Embedding(num_words_tgt, EMBEDDING_DIM, name="decoder_embedding")
    decoder_embed = decoder_embedding(decoder_input)

    attn_repeat = RepeatVector(max_len_src, name="attn_repeat")
    attn_concat = Concatenate(axis=-1, name="attn_concat")
    attn_dense1 = Dense(10, activation="tanh", name="attn_dense_1")
    attn_dense2 = Dense(1, name="attn_dense_2")
    attn_softmax = Softmax(axis=1, name="attn_softmax")
    attn_dot = Dot(axes=1, name="attn_dot")

    def one_step_attention(h, st_1):
        # h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2)
        # st_1 = s(t-1), shape = (LATENT_DIM_DECODER)

        # Copy s(t-1) Tx times
        st_1 = attn_repeat(st_1)

        x = attn_concat([h, st_1])
        x = attn_dense1(x)
        x = attn_dense2(x)
        alphas = attn_softmax(x)
        context = attn_dot([alphas, h])

        return context

    decoder_lstm = LSTM(LATENT_DIM_DECODER, dropout=0.3, recurrent_dropout=0.3, return_state=True, name="decoder_lstm")
    decoder_dense = Dense(num_words_tgt, activation="softmax", name="decoder_dense")

    decoder_in_s = Input(shape=(LATENT_DIM_DECODER,), name="s0")
    decoder_in_c = Input(shape=(LATENT_DIM_DECODER,), name="c0")
    context_concat = Concatenate(axis=2, name="context_concat")

    s = decoder_in_s
    c = decoder_in_c

    outputs = []
    for t in range(max_len_tgt):
        # Compute attention for each timestep, concat with target, and pass to decoder
        context = one_step_attention(encoder_output, s)

        # Fetch correct time step of target to be able to perform teacher forcing
        selector = Lambda(lambda x: x[:, t : t + 1])
        xt = selector(decoder_embed)
        decoder_lstm_input = context_concat([context, xt])

        o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])
        decoder_output = decoder_dense(o)
        outputs.append(decoder_output)

    def stack_and_transpose(x):
        # T x N x D -> N x T x D
        x = K.stack(x)
        x = K.permute_dimensions(x, pattern=(1, 0, 2))
        return x

    stacker = Lambda(stack_and_transpose)
    outputs = stacker(outputs)

    model = Model([encoder_input, decoder_input, decoder_in_s, decoder_in_c], outputs)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(lr=0.01))

    print("Training model...")
    z = np.zeros((NUM_SAMPLES, LATENT_DIM_DECODER))
    r = model.fit(
        [encoder_in_seq, decoder_in_seq, z, z],
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
    # TODO (mfortier) - Loading back model does not quite work because of for loop in output. To be fixed...
    model = load_model(MODEL_FNAME)
    model.summary()

print("Building generator...")
encoder_input = Input(shape=(max_len_src,))
encoder_emb = model.get_layer("encoder_embedding")(encoder_input)
encoder_output = model.get_layer("encoder_lstm")(encoder_emb)
encoder = Model(encoder_input, encoder_output)

encoder_output_input = Input(shape=(max_len_src, LATENT_DIM * 2))

decoder_input = Input(shape=(1,))
decoder_emb = model.get_layer("decoder_embedding")(decoder_input)


def one_step_attention(h, st_1):
    # h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2)
    # st_1 = s(t-1), shape = (LATENT_DIM_DECODER)

    # Copy s(t-1) Tx times
    st_1 = model.get_layer("attn_repeat")(st_1)

    x = model.get_layer("attn_concat")([h, st_1])
    x = model.get_layer("attn_dense_1")(x)
    x = model.get_layer("attn_dense_2")(x)
    alphas = model.get_layer("attn_softmax")(x)
    context = model.get_layer("attn_dot")([alphas, h])

    return context


decoder_in_s = Input(shape=(LATENT_DIM_DECODER,), name="s0")
decoder_in_c = Input(shape=(LATENT_DIM_DECODER,), name="c0")

context = one_step_attention(encoder_output_input, decoder_in_s)
decoder_lstm_input = model.get_layer("context_concat")([context, decoder_emb])


o, s, c = model.get_layer("decoder_lstm")(decoder_lstm_input, initial_state=[decoder_in_s, decoder_in_c])
decoder_output = model.get_layer("decoder_dense")(o)

decoder = Model(
    inputs=[decoder_input, encoder_output_input, decoder_in_s, decoder_in_c], outputs=[decoder_output, s, c]
)

idx2word_src = {v: k for k, v in word2idx_src.items()}
idx2word_tgt = {v: k for k, v in word2idx_tgt.items()}


def translate(input_seq):
    states = encoder.predict(input_seq)

    tgt_seq = np.zeros((1, 1))
    tgt_seq[0, 0] = word2idx_tgt.get("<bos>")
    eos = word2idx_tgt.get("<eos>")

    s = np.zeros((1, LATENT_DIM_DECODER))
    c = np.zeros((1, LATENT_DIM_DECODER))

    output_sentence = []
    for _ in range(max_len_tgt):
        o, s, c = decoder.predict([tgt_seq, states, s, c])

        idx = np.argmax(o.flatten())
        if idx == eos:
            break

        if idx > 0:
            output_sentence.append(idx2word_tgt.get(idx))

        tgt_seq[0, 0] = idx

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
