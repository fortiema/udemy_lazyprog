from collections import defaultdict
import string
import sys

import numpy as np


initial = {}
second_word = defaultdict(list)
transitions = defaultdict(list)


def remove_punct(s):
    """Naively remove all punctuation from string"""
    return s.translate({ord(k): None for k in set(string.punctuation)})


def list2pdict(ts):
    d = {}
    n = len(ts)
    for t in ts:
        d[t] = d.get(t, 0.0) + 1
    for t, c in d.items():
        d[t] = c / n
    return d


def sample_word(d):
    p0 = np.random.random()
    cumul = 0
    for t, p in d.items():
        cumul += p
        if p0 < cumul:
            return t
    assert False


def make_model():
    for line in open('large_files/robert_frost.txt', 'r'):
        tokens = remove_punct(line.rstrip().lower()).split()

        T = len(tokens)
        for i in range(T):
            t = tokens[i]
            if i == 0:
                initial[t] = initial.get(t, 0.1) + 1
            else:
                t_1 = tokens[i-1]
                if i == T - 1:
                    # last word
                    transitions[(t_1, t)].append('END')
                if i == 1:
                    # second word
                    second_word[(t_1)].append(t)
                else:
                    t_2 = tokens[i-2]
                    transitions[(t_2, t_1)].append(t)

    # Normalize dists.
    initial_total = sum(initial.values())
    for t, c in initial.items():
        initial[t] = c / initial_total

    for t_1, ts in second_word.items():
        second_word[t_1] = list2pdict(ts)

    for k, ts in transitions.items():
        transitions[k] = list2pdict(ts)


def generate(lines):
    for i in range(lines):
        sent = []

        w0 = sample_word(initial)
        sent.append(w0)

        w1 = sample_word(second_word[w0])
        sent.append(w1)

        while True:
            w2 = sample_word(transitions[(w0, w1)])
            if w2 == 'END':
                break
            sent.append(w2)
            w0 = w1
            w1 = w2
        print(' '.join(sent))


if __name__ == '__main__':
    make_model()
    generate(int(sys.argv[1]))
