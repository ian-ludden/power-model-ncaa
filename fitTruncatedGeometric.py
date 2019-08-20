import json
import numpy as np
from collections import defaultdict
from math import log
import numpy as np
import sys
from utils.bracketManipulations import aggregate
from fitPowerModel import load_ref_brackets

######################################################################
# Authors:  Ian Ludden
# Date:     20 August 2019
# 
# fitTruncatedGeometric.py
# 
# Fits modified truncated geometric distributions to 
# historical tournament data and outputs optimized parameters.
# 
######################################################################

def computeAllRoundCounts(brackets):
    # roundCounts(i, j) = the number of times seed j reached round i
    # (0 indices are unused)
    roundCounts = np.zeros((8, 17))

    counts = aggregate(brackets)
    for r in range(1, 7):
        for seedPair, seedGames in counts[r].items():
            s1 = min(seedPair[0], seedPair[1])
            s2 = max(seedPair[0], seedPair[1])
            totalGames = seedGames[s1] + seedGames[s2]
            if s1 == s2:
                totalGames = totalGames // 2
            roundCounts[r, s1] += totalGames
            roundCounts[r, s2] += totalGames

    return roundCounts


def compute_all_alphas(brackets):
    """Computes all alpha-values using data from 
       the given historical tournament brackets.

       Parameters
       ----------
       brackets : list of lists
           A list of 63-integer lists of 0s and 1s representing brackets

       Returns
       -------
       counts : dictionary of dictionaries of integers
           Win counts keyed by round, matchup, and winner
    """
    counts = bm.aggregate(brackets)
    result = {}
    for r in range(1, 7):
        result[r] = {}
        alphas = []
        weights = []
        for seedPair, seedGames in counts[r].items():
            s1 = min(seedPair[0], seedPair[1])
            s2 = max(seedPair[0], seedPair[1])

            if s1 == s2: # Omit match-ups between identical seeds from weighted averages 
                continue

            s1Wins = seedGames[s1]
            s2Wins = seedGames[s2]
            alpha = calculateAlpha(s1, s2, s1Wins, s2Wins)
            # print(seedPair, alpha)
            if r == 1:
                alpha = np.sign(alpha) * min(abs(alpha), CAPPED_ALPHA)
                result[r][s1] = {s2: alpha}
                print('{0},{1}'.format(s1, s2))
                print('{0},{1},{2},{3:.4f}'.format(s1Wins, s2Wins, s1Wins + s2Wins, alpha))
            else:
                alphas.append(alpha)
                weights.append(s1Wins + s2Wins)
        
        if r > 1:
            alpha = np.average(alphas, weights=weights)
            alpha = np.sign(alpha) * min(abs(alpha), CAPPED_ALPHA)
            result[r] = alpha

    print()

    return result


if __name__ == '__main__':
    all_results = defaultdict(list)

    # # Print Round 1 results
    # for year in range(2013, 2021):
    #     print(year)
    #     result = compute_all_alphas([b for x, b in load_ref_brackets().items() if x < year])
    #     for r in range(2, 7):
    #         all_results[r].append(result[r])

    # # Print Rounds 2-6 results
    # sys.stdout.write('Round,')
    # for year in range(2013, 2021):
    #     sys.stdout.write('{0},'.format(year))
    # sys.stdout.write('\n')

    # for r, data in all_results.items():
    #     sys.stdout.write('{0},'.format(r))
    #     for element in data:
    #         sys.stdout.write('{0:.4f},'.format(element))
    #     sys.stdout.write('\n')