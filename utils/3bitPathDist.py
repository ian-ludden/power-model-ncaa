from __future__ import print_function

__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2
import rpy2.robjects as robjects
import seaborn as sns
from scipy.stats import chisquare


paths = {
    'P_S1': [0, 8, 12],
    'P_S8': [1, 8, 12],
    'P_S5': [2, 9, 12],
    'P_S4': [3, 9, 12],
    'P_S6': [4, 10, 13],
    'P_S3': [5, 10, 13],
    'P_S7': [6, 11, 13],
    'P_S2': [7, 11, 13],
    'P_R2_1': [8, 12, 14],
    'P_R2_2': [9, 12, 14],
    'P_R2_3': [10, 13, 14],
    'P_R2_4': [11, 13, 14]
}


plt.style.use('seaborn-white')
sns.set_palette('colorblind')


def load_brackets(fmt='TTT'):
    with open('allBrackets{}.json'.format(fmt)) as f:
        brackets = json.load(f)['brackets']
        return brackets


def observed_dist(brackets, year, bits):
    vectors = [list(bracket['bracket']['fullvector'])
               for bracket in brackets
               if int(bracket['bracket']['year']) < year]
    vectors = np.array(vectors, dtype=int)
    vectors = vectors[:, :60].reshape(-1, 15)
    triplets = vectors[:, bits]
    triplets, counts = np.unique(triplets, axis=0, return_counts=True)
    triplet_labels = np.apply_along_axis(''.join, 1, triplets.astype(str))
    for t in ['000', '001', '010', '011', '100', '101', '110', '111']:
        if t not in triplet_labels:
            triplet_labels = np.append(triplet_labels, t)
            counts = np.append(counts, 0)
    return {l: c for l, c in zip(triplet_labels, counts)}


def expected_dist(brackets, year, bits):
    vectors = [list(bracket['bracket']['fullvector'])
               for bracket in brackets
               if int(bracket['bracket']['year']) < year]
    vectors = np.array(vectors, dtype=int)
    vectors = vectors[:, :60].reshape(-1, 15)
    triplets = vectors[:, bits]
    p_1 = np.mean(triplets, axis=0)
    p_0 = 1 - p_1
    p = [p_0, p_1]
    result = {}
    for t in ['000', '001', '010', '011', '100', '101', '110', '111']:
        values = [int(x) for x in list(t)]
        triplet_p = np.prod([p[values[i]][i] for i in range(3)])
        result[t] = (year - 1985) * 4 * triplet_p
    return result


def uniformity_check(observed, expected):
    chi, p = chisquare(observed, expected)
    print('Uniformity chi-square test p-value', p)


def plot_dist(brackets, year, bits, name):
    observed = observed_dist(brackets, year, bits)
    expected = expected_dist(brackets, year, bits)
    data = {'Observed': observed, 'Expected (ind)': expected}
    df = pd.DataFrame.from_dict(data)
    df.plot.bar(rot=0)
    # plt.show()
    plt.title('3-bit path value distribution - {}'.format(name))
    plt.savefig('DistPlots/TTT/3bit_path-{}.png'.format(name))
    plt.cla()
    plt.clf()
    values = list(observed.values())
    keys = list(observed.keys())
    arr = 'array(c{}, dim=c(2, 2, 2))'.format(
        tuple(np.array(values)[np.argsort(keys)].astype(int).tolist()))
    res = robjects.r('library(hypergea); hypergeom.test(' + arr + ")['p.value']")
    p_value = np.array(res[0])[0]
    print('Independence Fisher exact test p-value', p_value)

    uniformity_check(list(observed.values()), np.repeat((year - 1985) * 4 / 8, 8))
    print()
    # print('m = array(c{}, dim=c(2, 2, 2))'.format(tuple(np.array(list(observed.values()))[np.argsort(observed.keys())].astype(int).tolist())))


if __name__ == '__main__':
    brackets = load_brackets()

    for name, bits in paths.items():
        print('path {}'.format(name))
        plot_dist(brackets, 2019, bits, name)
