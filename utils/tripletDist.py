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
import seaborn as sns
from scipy.stats import power_divergence, chi2_contingency
from sklearn.model_selection import train_test_split


plt.style.use('seaborn-white')
sns.set_palette('dark')


def load_brackets(fmt='TTT'):
    with open('allBrackets{}.json'.format(fmt)) as f:
        brackets = json.load(f)['brackets']
        return brackets


def triplets_f4(brackets, year):
    vectors = [list(bracket['bracket']['fullvector'])
               for bracket in brackets
               if int(bracket['bracket']['year']) < year]
    vectors = np.array(vectors, dtype=int)
    vectors = vectors[:, :60].reshape(-1, 15)
    return vectors[:, -3:]


def triplets_e8(brackets, year):
    vectors = [list(bracket['bracket']['fullvector'])
               for bracket in brackets
               if int(bracket['bracket']['year']) < year]
    vectors = np.array(vectors, dtype=int)
    vectors = vectors[:, :60].reshape(-1, 15)
    return vectors[:, [8, 9, 12]], vectors[:, [10, 11, 13]]


def triplets_r2(brackets, year):
    vectors = [list(bracket['bracket']['fullvector'])
               for bracket in brackets
               if int(bracket['bracket']['year']) < year]
    vectors = np.array(vectors, dtype=int)
    vectors = vectors[:, :60].reshape(-1, 15)
    return vectors[:, [0, 1, 8]], vectors[:, [2, 3, 9]], vectors[:, [4, 5, 10]], vectors[:, [6, 7, 11]]


def all_years_f4_triplet_plot(brackets, output_dir):
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for n, year in enumerate(range(2014, 2020)):
        triplets = triplets_f4(brackets, year)
        triplets = np.apply_along_axis(''.join, 1, triplets.astype(str))
        triplets, counts = np.unique(triplets, return_counts=True)

        i = n // 3
        j = n % 3
        sns.barplot(triplets, counts, color='black', ax=ax[i, j],
                    order=triplets[np.argsort(counts)])
        ax[i, j].set_title('1985 - {}'.format(year - 1))
    fig.suptitle('E8-F4 triplet dist', fontsize=16)
    # plt.show()
    plt.savefig(output_dir + '/f4_triplet_all.png')
    plt.cla()
    plt.clf()


def per_year_f4_triplet_plot(brackets, output_dir):
    for year in range(2013, 2020):
        triplets = triplets_f4(brackets, year)
        triplets = np.apply_along_axis(''.join, 1, triplets.astype(str))
        triplets, counts = np.unique(triplets, return_counts=True)
        sns.barplot(triplets, counts, color='black', order=triplets[np.argsort(counts)])
        plt.title('E8-F4 triplet distribution (1985 - {})'.format(year - 1))
        # plt.xlabel('Triplet (TTT format)')
        # plt.show()
        plt.savefig(output_dir + '/f4_triplet_{}.png'.format(year))
        plt.cla()
        plt.clf()


def check_triplet_dependency_all_data(raw_triplets):
    triplets_for_estimation = raw_triplets
    triplets_for_independence = raw_triplets
    return _check_triplet_dependency(triplets_for_estimation, triplets_for_independence)


def check_triplet_dependency_50_split(raw_triplets):
    triplets_for_estimation = raw_triplets[:len(raw_triplets) / 2, :]
    triplets_for_independence = raw_triplets[len(raw_triplets) / 2:, :]
    return _check_triplet_dependency(triplets_for_estimation, triplets_for_independence)


def check_triplet_dependency_random_split(raw_triplets):
    values = []
    for i in range(1000):
        triplets_for_estimation, triplets_for_independence = train_test_split(raw_triplets, train_size=0.5, test_size=0.5)
        p_value = _check_triplet_dependency(triplets_for_estimation, triplets_for_independence)
        values.append(p_value)
    p_value = 1000 - np.count_nonzero(np.greater(values, 0.05))
    print('# rejects', p_value)
    return p_value


def _check_triplet_dependency(triplets_for_estimation, triplets_for_independence):
    n = triplets_for_estimation.shape[0]
    triplets = np.apply_along_axis(''.join, 1,
                                   triplets_for_estimation.astype(str))
    triplets, counts = np.unique(triplets, return_counts=True)
    triplet_probs = 1. * counts / counts.sum()

    bits, counts_0 = np.unique(triplets_for_independence[:, 0], return_counts=True)
    if bits[0] != 0:
        bits = np.insert(bits, 0, 0)
        counts_0 = np.insert(counts_0, 0, 0)
    if bits[1] != 1:
        counts_0 = np.insert(counts_0, 1, 0)
    probs_0 = 1. * counts_0 / counts_0.sum()

    bits, counts_1 = np.unique(triplets_for_independence[:, 1], return_counts=True)
    if bits[0] != 0:
        bits = np.insert(bits, 0, 0)
        counts_1 = np.insert(counts_1, 0, 0)
    if bits[1] != 1:
        counts_1 = np.insert(counts_1, 1, 0)
    probs_1 = 1. * counts_1 / counts_1.sum()

    bits, counts_2 = np.unique(triplets_for_independence[:, 2], return_counts=True)
    if bits[0] != 0:
        bits = np.insert(bits, 0, 0)
        counts_2 = np.insert(counts_2, 0, 0)
    if bits[1] != 1:
        counts_2 = np.insert(counts_2, 1, 0)
    probs_2 = 1. * counts_2 / counts_2.sum()

    observed = []
    expected = []
    for t in ['000', '001', '010', '011', '100', '101', '110', '111']:
        if t not in triplets:
            triplets = np.append(triplets, t)
            triplet_probs = np.append(triplet_probs, 0.)
            counts = np.append(counts, 0.)

    # sns.barplot(triplets, counts, color='black')
    # plt.xlabel('Triplet')
    # plt.ylabel('Count')
    # plt.show()

    for i in range(8):
        triplet = np.array(list(triplets[i]), dtype=int)
        n_triplet = counts[i]
        # p_triplet = triplet_probs[i]
        p_ind = probs_0[triplet[0]] * probs_1[triplet[1]] * probs_2[triplet[2]]
        n_ind = n * p_ind
        # print('Triplet: {0}, p(i,j,k) = {1:5f}, p(i)p(j)p(k) = {2:5f}. diff = {3:5f}'.format(triplets[i], p_triplet, p_ind, p_triplet - p_ind))
        if p_ind == 0:
            continue
        observed.append(n_triplet)
        expected.append(n_ind)
        # expected.append(np.ceil(n_ind).astype(int))

    # df = pd.DataFrame.from_records(np.vstack((observed, expected)).T, index=triplets, columns=['Observed', 'Expected']).astype(int)
    # print(df)

    print('m = array(c{}, dim=c(2, 2, 2))'.format(tuple(np.array(observed)[np.argsort(triplets)].astype(int).tolist())))

    chi2, p_value = power_divergence(observed, expected, ddof=3, lambda_='pearson')
    # print('Chi-square p-value', p_value, '=>',
    #       ('Indepedent' if p_value > 0.05 else 'There is dependency'))
    # print()
    return p_value


def chi_square_test(triplets):
    for i in range(3):
        l = list(range(3))
        l.pop(i)
        data = triplets[:, l]
        _, c00 = np.unique(triplets[(data[:, 0] == 0) & (data[:, 1] == 0), i],
                           return_counts=True)
        _, c01 = np.unique(triplets[(data[:, 0] == 0) & (data[:, 1] == 1), i],
                           return_counts=True)
        _, c10 = np.unique(triplets[(data[:, 0] == 1) & (data[:, 1] == 0), i],
                           return_counts=True)
        _, c11 = np.unique(triplets[(data[:, 0] == 1) & (data[:, 1] == 1), i],
                           return_counts=True)
        table = {
            '00': {'0': c00[0], '1': c00[1]},
            '01': {'0': c01[0], '1': c01[1]},
            '10': {'0': c10[0], '1': c10[1]},
            '11': {'0': c11[0], '1': c11[1]}
        }
        table = pd.DataFrame.from_dict(table)
        chi2, p, dof, expected = chi2_contingency(table)
        print(table)
        print('p-value', p)


def check_all_triplet_dependency(brackets):
    independence_check = check_triplet_dependency_all_data
    for year in range(2019, 2020):
        print('===== YEAR = {} ====='.format(year))
        print('E8-F4 triplet')
        raw_triplets = triplets_f4(brackets, year)
        independence_check(raw_triplets)
        # chi_square_test(raw_triplets)

        print('Top S16-E8 triplet')
        top, bottom = triplets_e8(brackets, year)
        independence_check(top)
        # chi_square_test(top)

        print('Bottom S16-E8 triplet')
        independence_check(bottom)
        # chi_square_test(bottom)

        t1, t2, t3, t4 = triplets_r2(brackets, year)
        print('R1-R2 triplet 1')
        independence_check(t1)

        print('R1-R2 triplet 2')
        independence_check(t2)

        print('R1-R2 triplet 3')
        independence_check(t3)

        print('R1-R2 triplet 4')
        independence_check(t4)


if __name__ == '__main__':
    import os
    import sys
    fmt = sys.argv[1]
    output = sys.argv[2]

    if not os.path.exists(output):
        os.makedirs(output)

    brackets = load_brackets(fmt)
    check_all_triplet_dependency(brackets)
    # all_years_f4_triplet_plot(brackets, output)
    # per_year_f4_triplet_plot(brackets, output)