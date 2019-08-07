__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"

import json
# import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb

dirname = os.path.dirname(__file__) or '.'

pd.set_option('display.width', 200)

# plt.switch_backend('MacOSX')
# plt.rcParams["figure.figsize"] = (15, 8)

formats = ['TTT', 'FFF', 'TTF', 'TFT', 'TFF', 'FTT', 'FTF', 'FFT']

triplets = [
    [0, 1, 8],
    [2, 3, 9],
    [4, 5, 10],
    [6, 7, 11],
    [12, 13, 14]
]

default_triplet_series = pd.Series({
    '000': 0,
    '001': 0,
    '010': 0,
    '011': 0,
    '100': 0,
    '101': 0,
    '110': 0,
    '111': 0
})


default_triplet_df = pd.DataFrame(data=np.array([
    [0, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 0],
    [1, 0, 1, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 0]
]), columns=[0, 1, 2, 'count'])
default_triplet_df = pd.concat([default_triplet_df[[0, 1, 2]].astype(str).apply(''.join, axis=1), default_triplet_df['count']], axis=1)
default_triplet_df = default_triplet_df.set_index(0)


default_single_bit_series = pd.Series({
    '0': 0,
    '1': 0
})

UNPOOLED = 0
POOLED = 1


data_cache = {}


def read_data(fmt, limit=0, start=1985):
    key = hash((fmt, limit, start))
    if key not in data_cache:
        with open(dirname + '/../../generators/allBrackets%s.json' % fmt) as f:
            vectors = [a['bracket']['fullvector']
                       for a in json.load(f)['brackets']
                       if (limit == 0 or int(a['bracket']['year']) < limit) and
                       int(a['bracket']['year']) >= start]

        unpooled = pd.DataFrame([list(a) for a in vectors])

        pooled = pd.DataFrame()
        pooled = pooled.append([list(a)[:15] for a in vectors])
        pooled = pooled.append([list(a)[15:30] for a in vectors])
        pooled = pooled.append([list(a)[30:45] for a in vectors])
        pooled = pooled.append([list(a)[45:60] for a in vectors])
        data_cache[key] = (unpooled, pooled)
    return data_cache[key]


def pooled_single_bit_distributions():
    for bits in np.arange(0, 15)[:, np.newaxis].tolist():
        title = 'Single bit distribution for bit %d' % tuple(bits)
        plot_path = ('plots/single_bit/pooled_%d' % tuple(bits)) + '.png'
        csv_path = ('csv/single_bit/pooled_%d' % tuple(bits)) + '.csv'
        _dist_helper(bits, POOLED, default_single_bit_series,
                     title, plot_path, csv_path)


def unpooled_single_bit_distributions():
    for region in [0, 1, 2, 3]:
        _unpooled_single_bit_distributions(region=region)
    final_3_single_bit_distributions()


def pooled_triplet_distributions():
    for triplet in triplets:
        title = 'Triplet distribution for bits %d, %d and %d' % tuple(triplet)
        plot_path = ('plots/triplets/pooled_%d_%d_%d' % tuple(triplet)) + '.png'
        csv_path = ('csv/triplets/pooled-%d-%d-%d' % tuple(triplet)) + '.csv'
        _dist_helper(triplet, POOLED, default_triplet_series, title, plot_path, csv_path)


def unpooled_triplet_distributions():
    for region in [0, 1, 2, 3]:
        _unpooled_triplet_distributions(region=region)
    final_3_triplet_distributions()


def final_3_single_bit_distributions():
    bits = [[60], [61], [62]]
    for bit in bits:
        title = 'Single bit distribution for bit %d' % tuple(bit)
        plot_path = ('plots/single_bit/unpooled_%d' % tuple(bit)) + '.png'
        csv_path = ('csv/single_bit/unpooled-%d' % tuple(bit)) + '.csv'
        _dist_helper(bit, UNPOOLED, default_single_bit_series, title, plot_path, csv_path)


def final_3_triplet_distributions():
    triplet = [60, 61, 62]
    title = 'Triplet distribution for bits %d, %d and %d' % tuple(triplet)
    plot_path = ('plots/triplets/unpooled_%d_%d_%d' % tuple(triplet)) + '.png'
    csv_path = ('csv/triplets/unpooled-%d-%d-%d' % tuple(triplet)) + '.csv'
    _dist_helper(triplet, UNPOOLED, default_triplet_series, title, plot_path, csv_path)


def _unpooled_single_bit_distributions(region=0):
    for bits in np.arange(region * 15, (region + 1) * 15)[:, np.newaxis].tolist():
        title = 'Region: %d. Single bit distribution for bit %d' % tuple([region] + bits)
        plot_path = ('plots/single_bit/region_%d-unpooled_%d' % tuple([region] + bits)) + '.png'
        csv_path = ('csv/single_bit/region_%d-unpooled_%d' % tuple([region] + bits)) + '.csv'
        _dist_helper(bits, UNPOOLED, default_single_bit_series,
                     title, plot_path, csv_path)


def _unpooled_triplet_distributions(region=0):
    for triplet in (np.array(triplets) + (region * 15)).tolist():
        title = 'Region: %d. Triplet distribution for bits %d, %d and %d' % tuple([region] + triplet)
        plot_path = ('plots/triplets/region_%d-unpooled_%d_%d_%d' % tuple([region] + triplet)) + '.png'
        csv_path = ('csv/triplets/region_%d-unpooled_%d_%d_%d' % tuple([region] + triplet)) + '.csv'
        _dist_helper(triplet, UNPOOLED, default_triplet_series,
                     title, plot_path, csv_path)


def _dist_helper(cols, data_selector, zeros, title, plot_file_path, csv_file_path):
    series = {}
    for fmt in formats:
        data = read_data(fmt)[data_selector]
        dist = data[cols].groupby(by=cols).size().reset_index(name='count')
        dist = pd.concat([
            dist[cols].apply(lambda a: ''.join(a), axis=1),
            dist[['count']]],
            axis=1
        ).set_index(0)
        dist = dist / dist['count'].sum()
        series[fmt] = zeros.add(dist['count'], fill_value=0)
    all_formats_df = pd.DataFrame.from_dict(series)
    all_formats_df.plot.bar(rot=0, grid=True, title=title)
    plt.xlabel('Combination')
    plt.ylabel('Probability')
    plt.savefig(plot_file_path)
    plt.close()

    print(title)
    print(all_formats_df)
    print('#' * 120)

    all_formats_df.to_csv(csv_file_path, sep=',',
                          index_label='combination', float_format='%.3f')


if __name__ == '__main__':
    for result_dir in ['csv', 'plots']:
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
            os.mkdir(result_dir + '/single_bit')
            os.mkdir(result_dir + '/triplets')
        else:
            if not os.path.exists(result_dir + '/single_bit'):
                os.mkdir(result_dir + '/single_bit')
            if not os.path.exists(result_dir + '/triplets'):
                os.mkdir(result_dir + '/triplets')

    pooled_single_bit_distributions()
    unpooled_single_bit_distributions()

    pooled_triplet_distributions()
    unpooled_triplet_distributions()
