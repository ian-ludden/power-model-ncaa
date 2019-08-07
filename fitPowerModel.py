__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import json
import numpy as np
from collections import defaultdict
from pprint import pprint

CAPPED_ALPHA = 2.


def calculateAlpha(s1, s2, s1Wins, s2Wins, perturb=False):
    if s1Wins + s2Wins == 0:
        return 0.

    if s1Wins == 0:
        return -CAPPED_ALPHA
    if s2Wins == 0:
        return CAPPED_ALPHA
    if s1Wins == s2Wins:
        return 0.
    if s1 == s2:
        return 0.

    p_j = 1. * s1Wins / (s1Wins + s2Wins)
    if perturb:
        p_j = np.clip(p_j + np.random.uniform(0, p_j * 0.1), 0, 1)
    try:
        val = np.log(p_j / (1. - p_j)) / np.log(1. * s2 / s1)
        return val
    except:
        return CAPPED_ALPHA


def bracket_to_seeds(bracket):
    f4_seeds = []
    ru_seeds = []

    per_round_seeds = defaultdict(list)

    for region in range(4):
        round_num = 1
        region_vector = bracket[region * 15: (region + 1) * 15]
        seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
        per_round_seeds[0] += seeds
        while len(seeds) > 1:
            n_games = len(seeds) // 2
            new_seeds = []
            for game in range(n_games):
                s1, s2 = seeds[game * 2], seeds[game * 2 + 1]
                if region_vector[game] == 1:
                    new_seeds.append(s1)
                else:
                    new_seeds.append(s2)
            region_vector = region_vector[n_games:]
            per_round_seeds[round_num] += new_seeds
            seeds = new_seeds
            round_num += 1
        f4_seeds.append(seeds[0])
    f4_games = bracket[-3:]
    if f4_games[0] == 1:
        ru_seeds.append(f4_seeds[0])
    else:
        ru_seeds.append(f4_seeds[1])

    if f4_games[1] == 1:
        ru_seeds.append(f4_seeds[2])
    else:
        ru_seeds.append(f4_seeds[3])

    if f4_games[2] == 1:
        nc_seed = ru_seeds[0]
    else:
        nc_seed = ru_seeds[1]

    per_round_seeds[5] = ru_seeds
    per_round_seeds[6] = [nc_seed]

    return per_round_seeds


def aggregate(brackets):
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for bracket in brackets:
        seeds = bracket_to_seeds(bracket)
        for round in range(1, 7):
            round_seeds = seeds[round]
            for game_id, winner in enumerate(round_seeds):
                s1 = min(seeds[round - 1][game_id * 2], seeds[round - 1][game_id * 2 + 1])
                s2 = max(seeds[round - 1][game_id * 2], seeds[round - 1][game_id * 2 + 1])
                matchup = (s1, s2)
                counts[round][matchup][winner] += 1
    return counts


def load_ref_brackets():
    # with open("allBracketsSince2002TTT.json") as f:
    with open("allBracketsTTT.json") as f:
        data = json.load(f)
        vectors = {
            int(bracket['bracket']['year']):
                np.array(list(bracket['bracket']['fullvector']), dtype=int)
            for bracket in data['brackets']}
    return vectors


def compute_all_alphas(brackets):
    counts = aggregate(brackets)
    result = {}
    for round in range(1, 7):
        result[round] = {}
        alphas = []
        weights = []
        for seedPair, seedGames in counts[round].items():
            s1 = min(seedPair[0], seedPair[1])
            s2 = max(seedPair[0], seedPair[1])

            if s1 == s2:
                continue

            s1Wins = seedGames[s1]
            s2Wins = seedGames[s2]
            alpha = calculateAlpha(s1, s2, s1Wins, s2Wins)
            # print(seedPair, alpha)
            if round == 1:
                alpha = np.sign(alpha) * min(abs(alpha), CAPPED_ALPHA)
                result[round][s1] = {s2: alpha}
                print('{0},{1}'.format(s1, s2))
                print('{0},{1},{2}'.format(s1Wins, s2Wins, s1Wins + s2Wins))
            else:
                alphas.append(alpha)
                weights.append(s1Wins + s2Wins)
        if round > 1:
            alpha = np.average(alphas, weights=weights)
            alpha = np.sign(alpha) * min(abs(alpha), CAPPED_ALPHA)
            result[round] = alpha

    print()

    return result


if __name__ == '__main__':
    from collections import defaultdict
    all_results = defaultdict(list)
    for year in range(2013, 2021):
        print(year)
        result = compute_all_alphas([b for x, b in load_ref_brackets().items() if x < year])
        for r in [2, 3, 4, 5, 6]:
            all_results[r].append(result[r])
    for r, data in all_results.items():
        print(r, data)