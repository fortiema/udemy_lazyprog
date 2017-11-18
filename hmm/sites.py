from cytoolz.dicttoolz import keyfilter
import numpy as np

transitions = {}
row_sums = {}

# Collect all counts
for line in open('large_files/site_data.csv'):
    s, e = line.rstrip().split(',')
    transitions[(s, e)] = transitions.get((s,e), 0.0) + 1
    row_sums[s] = row_sums.get(s, 0.) + 1

# Normalize
for k, v in transitions.items():
    s, e = k
    transitions[k] = v / row_sums[s]

# Initial State dist.
print('Init. State Distrib.')
for k, v in keyfilter(lambda x: x[0] == '-1', transitions).items():
    print(k[1], v)

# Bounce dist.
print('Bounce Distrib.')
for k, v in keyfilter(lambda x: x[1] == 'B', transitions).items():
    print(k[0], v)
