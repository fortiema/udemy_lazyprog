import numpy as np

from datasets import Wikipedia
from text import numericalize_tok, remove_punct, Tokenizer


def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


def all_parity_pairs(nbit):
    N = 2**nbit
    remainder = 100 - (N % 100)
    Ntotal = N + remainder
    X = np.zeros((Ntotal, nbit))
    Y = np.zeros(Ntotal)

    for ii in range(Ntotal):
        i = ii % N
        for j in range(nbit):
            if i % (2**(j + 1)) != 0:
                i -= 2**j
                X[ii, j] = 1
        Y[ii] = X[ii].sum() % 2
    return X, Y


def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def get_robert_frost():
    word2idx = {'BOS': 0, 'EOS': 1}
    current_idx = 2

    sents = []
    for line in open('large_files/robert_frost.txt'):
        line = line.strip()
        if line:
            tokens = remove_punct(line.lower()).split()
            sent = []
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    current_idx += 1
                idx = word2idx[t]
                sent.append(idx)
            sents.append(sent)
    return sents, word2idx


def get_wikipedia_data(n_entries, n_vocab):
    """Loads up Data From Wikipedia Dump Files

    Args:
        n_entries (int): Limit data to N entries
        n_vocab (int): Limit total vocabulary to top N tokens (/freq)
    """
    wiki_data = Wikipedia('/data/datasets/text/wikidump/20180901/processed/AA')

    tokenizer = Tokenizer()

    def get_data():
        for entry_count, entry in enumerate(wiki_data):
            if entry_count >= n_entries:
                break
            yield tokenizer.proc_text(entry.get('text', ''))

    int2tok, tok2int = numericalize_tok(list(get_data()), max_vocab=n_vocab)

    sents = []
    for ent in get_data():
        for sent in ent:
            sents.append([tok2int[tok] for tok in sent])

    return sents, tok2int, int2tok
