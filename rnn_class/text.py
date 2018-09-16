from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import re
import string

import numpy as np
import spacy
from spacy.attrs import ORTH


def remove_punct(s):
    return s.translate(str.maketrans("", "", string.punctuation))


def tokenize_basic(text):
    return text.lower().split()


def numericalize_tok(
    tokens, max_vocab=50000, min_freq=0, unk_tok="<unk>", pad_tok="<pad>", bos_tok="<bos>", eos_tok="<eos>"
):
    """Takes in text tokens and returns int2tok and tok2int converters

    Args:
        tokens(list): List of tokens. Can be a list of strings, or a list of lists of strings.
        max_vocab(int): Number of tokens to return in the vocab (sorted by frequency)
        min_freq(int): Minimum number of instances a token must be present in order to be preserved.
        unk_tok(str): Token to use when unknown tokens are encountered in the source text.
        pad_tok(str): Token to use when padding sequences.
    """
    if isinstance(tokens, str):
        raise ValueError("Expected to receive a list of tokens. Received a string instead")
    if isinstance(tokens[0], list):
        tokens = [p for o in tokens for p in o]
    freq = Counter(tokens)
    int2tok = [o for o, c in freq.most_common(max_vocab) if c > min_freq]
    unk_id = 3
    int2tok.insert(0, bos_tok)
    int2tok.insert(1, pad_tok)
    int2tok.insert(2, eos_tok)
    int2tok.insert(unk_id, unk_tok)
    tok2int = defaultdict(lambda: unk_id, {v: k for k, v in enumerate(int2tok)})
    return int2tok, tok2int


class Tokenizer:
    def __init__(self, lang="en"):
        self.re_br = re.compile(r"<\s*br\s*/?>", re.IGNORECASE)
        self.tok = spacy.load(lang)
        for w in ("<eos>", "<bos>", "<unk>"):
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])

    def sub_br(self, x):
        return self.re_br.sub("\n", x)

    def spacy_tok(self, x):
        return [t.text for t in self.tok.tokenizer(self.sub_br(x))]

    re_rep = re.compile(r"(\S)(\1{3,})")
    re_word_rep = re.compile(r"(\b\w+\W+)(\1{3,})")

    @staticmethod
    def replace_rep(m):
        TK_REP = "tk_rep"
        c, cc = m.groups()
        return f" {TK_REP} {len(cc)+1} {c} "

    @staticmethod
    def replace_wrep(m):
        TK_WREP = "tk_wrep"
        c, cc = m.groups()
        return f" {TK_WREP} {len(cc.split())+1} {c} "

    @staticmethod
    def do_caps(ss):
        TOK_UP, TOK_SENT, TOK_MIX = " t_up ", " t_st ", " t_mx "
        res = []
        # prev = "."
        # re_word = re.compile("\w")
        # re_nonsp = re.compile("\S")
        for s in re.findall(r"\w+|\W+", ss):
            res += [TOK_UP, s.lower()] if (s.isupper() and (len(s) > 2)) else [s.lower()]
        return "".join(res)

    def proc_text(self, s):
        s = self.re_rep.sub(Tokenizer.replace_rep, s)
        s = self.re_word_rep.sub(Tokenizer.replace_wrep, s)
        s = Tokenizer.do_caps(s)
        s = re.sub(r"([/#])", r" \1 ", s)
        s = re.sub(" {2,}", " ", s)
        return self.spacy_tok(s)

    @staticmethod
    def proc_all(ss, lang):
        tok = Tokenizer(lang)
        return [tok.proc_text(s) for s in ss]

    @staticmethod
    def proc_all_mp(ss, lang="en", ncpus=2):
        ncpus = ncpus
        with ProcessPoolExecutor(ncpus) as e:
            return sum(e.map(Tokenizer.proc_all, ss, [lang] * len(ss)), [])


class LanguageModelLoader:
    """ Returns a language model iterator that iterates through batches of text data

    Batches are of length N(bptt,5)
    The first batch returned is always bptt+25; the max possible width.  This is done because of they way that pytorch
    allocates cuda memory in order to prevent multiple buffers from being created as the batch width grows.
    """

    def __init__(self, nums, bs, bptt, backwards=False):
        self.bs, self.bptt, self.backwards = bs, bptt, backwards
        self.data = self.batchify(nums)
        self.i, self.iter = 0, 0
        self.n = len(self.data)

    def __iter__(self):
        self.i, self.iter = 0, 0
        while self.i < self.n - 1 and self.iter < len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res

    def __len__(self):
        return self.n // self.bptt - 1

    def batchify(self, data):
        nb = data.shape[0] // self.bs
        data = np.array(data[: nb * self.bs])
        data = data.reshape(self.bs, -1).T
        if self.backwards:
            data = data[::-1]
        return data

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i : i + seq_len], source[i + 1 : i + 1 + seq_len].view(-1)
