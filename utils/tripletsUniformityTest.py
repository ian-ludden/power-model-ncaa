from __future__ import print_function

__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import json
import numpy as np
from scipy.stats import chisquare


def load_brackets(fmt='TTT'):
    with open('allBrackets{}.json'.format(fmt)) as f:
        brackets = json.load(f)['brackets']
        return brackets


def test_uniformity(triplets):
    triplets = np.apply_along_axis(''.join, 1, triplets.astype(str))
    triplets, counts = np.unique(triplets, return_counts=True)
    for t in ['000', '001', '010', '011', '100', '101', '110', '111']:
        if t not in triplets:
            triplets = np.append(triplets, t)
            counts = np.append(counts, 0.)
    return chisquare(counts, 136/8, ddof=3)


def all_triplets(vectors):
    triplets = [
        [12, 13, 14],
        [8, 9, 12],
        [10, 11, 13],
        [0, 1, 8],
        [2, 3, 9],
        [4, 5, 10],
        [6, 7, 11]
    ]
    r1 = vectors[:, :60].reshape(-1, 15)
    return [r1[triplet] for triplet in triplets]


if __name__ == '__main__':
    import sys

    fmt = sys.argv[1]
    output = sys.argv[2]

    brackets = load_brackets(fmt)
    vectors = np.array([list(bracket['bracket']['fullvector'])
                        for bracket in brackets])
    triplets_split = all_triplets(vectors)
    e8, f4_1, f4_2, r2_1, r2_2, r2_3, r2_4 = triplets_split
    names = ['E8-F4', 'S16-E8_1', 'S16_E8_2', 'R2_1', 'R2_2', 'R2_3', 'R2_4']
    for triplets, name in zip(triplets_split, names):
        _, p_value = test_uniformity(triplets)
        print(name, 'p-value', p_value)
