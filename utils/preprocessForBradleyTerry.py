__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"

import json
import numpy as np
import os
import pandas as pd

dirname = os.path.dirname(__file__) or '.'


def preprocess(year):
    """
    Creates counts of s1 vs s2 winning/lossing over all years until the given year - 1.
    i.e. year=2018 uses data up until 2017 for its counts
    :param year: limit year
    :return: a count distribution of s1 winning/lossing over s2
    """
    unpooled, pooled = read_data('TTT', year)
    unpooled = unpooled.values.astype(int)
    data = pooled.values.astype(int)

    records = []
    counter = 0
    f4Seeds = []

    for row_i in range(data.shape[0]):
        row = data[row_i, :]
        seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
        winning_seed = -1
        for i, bit in enumerate(row):
            matchup = seeds[i * 2], seeds[i * 2 + 1]
            winning_seed = matchup[1 - bit]
            seeds.append(winning_seed)
            records.append([
                's{}'.format(min(matchup)),
                's{}'.format(max(matchup)),
                'W{}'.format(1 if winning_seed == min(matchup) else 2)])

        f4Seeds.append(winning_seed)
        if len(f4Seeds) == 4:
            year_index = counter // 4
            winning_seed_1 = f4Seeds[unpooled[year_index, 60]]
            if f4Seeds[0] == f4Seeds[1]:
                winner = unpooled[year_index, 60] + 1
            else:
                winner = 1 if winning_seed_1 == min(f4Seeds[0],
                                                    f4Seeds[1]) else 2
            records.append([
                's{}'.format(min(f4Seeds[0], f4Seeds[1])),
                's{}'.format(max(f4Seeds[0], f4Seeds[1])),
                'W{}'.format(winner)
            ])

            winning_seed_2 = f4Seeds[2:][unpooled[year_index, 61]]
            if f4Seeds[2] == f4Seeds[3]:
                winner = unpooled[year_index, 61] + 1
            else:
                winner = 1 if winning_seed_2 == min(f4Seeds[2],
                                                    f4Seeds[3]) else 2
            records.append([
                's{}'.format(min(f4Seeds[2], f4Seeds[3])),
                's{}'.format(max(f4Seeds[2], f4Seeds[3])),
                'W{}'.format(winner)
            ])

            champ = [winning_seed_1, winning_seed_2][unpooled[year_index, 62]]
            if winning_seed_1 == winning_seed_2:
                winner = unpooled[year_index, 62] + 1
            else:
                winner = 1 if champ == min(winning_seed_1,
                                           winning_seed_2) else 2
            records.append([
                's{}'.format(min(winning_seed_1, winning_seed_2)),
                's{}'.format(max(winning_seed_1, winning_seed_2)),
                'W{}'.format(winner)
            ])
        counter += 1

    df = pd.DataFrame(records, columns=['player1', 'player2', 'outcome'])
    df.to_csv('forBT-raw.csv', index=False)

    groups = df.groupby(
        by=['player1', 'player2', 'outcome']).size().reset_index(name='count')
    result = []
    for key, data in groups.groupby(by=['player1', 'player2']):
        tmp = data[data['outcome'] == 'W1']['count']
        if tmp.size > 0:
            w1 = tmp.values[0]
        else:
            w1 = 0

        tmp = data[data['outcome'] == 'W2']['count']
        if tmp.size > 0:
            w2 = tmp.values[0]
        else:
            w2 = 0
        result.append([key[0], key[1], w1, w2])

    df = pd.DataFrame(result,
                      columns=['player1', 'player2', 'item1wins', 'item2wins'])
    return df


if __name__ == '__main__':
    for year in range(2013, 2020):
        df = preprocess(year)
        df.to_csv('bradleyTerry/counts-{}.csv'.format(year), index=False)


data_cache = {}

def read_data(fmt, limit=0, start=1985):
    key = hash((fmt, limit, start))
    if key not in data_cache:
        # allBracketsTTT.json is stored one level above
        with open(dirname + '/../allBrackets%s.json' % fmt) as f:
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