import json
import numpy as np
from collections import defaultdict
from math import log
import sys
import utils.bracketManipulations as bm

######################################################################
# Authors:  Nestor Bermudez, Ian Ludden
# Date:     08 August 2019
# 
# fitPowerModel.py
# 
# Fits the power model to historical tournament data and outputs 
# the parameters to be used to estimate win probabilities. 
# 
######################################################################

# Maximum absolute value of alpha-value
CAPPED_ALPHA = 2.

def calculateAlpha(seed1, seed2, seed1Wins, seed2Wins):
    """Computes the alpha-value for the power model 
       given the seeds and how many games each has won in
       their previous match-ups in some round. As described in pdf.

       Parameters
       ----------
       seed1 : the first seed (int between 1 and 16, inclusive)
       seed2 : the second seed (int between 1 and 16, inclusive)
       seed1Wins : the number of past games won by seed1
       seed2Wins : the number of past games won by seed2

       Returns
       -------
       alpha : float 
           The alpha value for seed1 playing seed2 in 
           the round from which the win counts are tallied. 
    """
    # Handle special cases
    if seed1Wins + seed2Wins == 0:
        return 0.
    if seed1Wins == 0:
        return -CAPPED_ALPHA
    if seed2Wins == 0:
        return CAPPED_ALPHA
    if seed1Wins == seed2Wins:
        return 0.
    if seed1 == seed2:
        return 0.

    historicalProportion = 1. * seed1Wins / (seed1Wins + seed2Wins)
    try:
        numerator = log(historicalProportion / (1. - historicalProportion))
        denominator = log(1. * seed2 / seed1)
        return numerator / denominator
    except: # Perhaps a divide by zero error
        return CAPPED_ALPHA


def load_ref_brackets(inputFilename="allBracketsTTT.json"):
    """Given the name of a JSON file containing 
       the actual tournament brackets in some subset of 
       tournament years, this function returns a 
       dictionary of the bracket vectors with years as keys.
       Each bracket vector is a list of 63 0s and/or 1s.

       This is called in the main function. So the historical results
       are converted from json to the 63 bit string representation
    """
    with open(inputFilename, 'r') as f:
        data = json.load(f)
        vectors = {
            int(bracket['bracket']['year']):
                np.array(list(bracket['bracket']['fullvector']), dtype=int)
            for bracket in data['brackets']}
    return vectors


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

                # output looks like this
                print('{0},{1}'.format(s1, s2))
                print('{0},{1},{2},{3:.4f}'.format(s1Wins, s2Wins, s1Wins + s2Wins, alpha))
            else:
                alphas.append(alpha)
                weights.append(s1Wins + s2Wins)
        
        if r > 1:
            # any games after round of 64 are "weighted" by number of occurences
            
            alpha = np.average(alphas, weights=weights)
            alpha = np.sign(alpha) * min(abs(alpha), CAPPED_ALPHA)
            result[r] = alpha

    print()

    return result


if __name__ == '__main__':
    all_results = defaultdict(list)

    # Print Round 1 results
    for year in range(2013, 2021):
        print(year)

        # x corresponds to year, b the bit string result
        result = compute_all_alphas([b for x, b in load_ref_brackets().items() if x < year])
        for r in range(2, 7):
            all_results[r].append(result[r])

    # Print Rounds 2-6 results
    sys.stdout.write('Round,')
    for year in range(2013, 2021):
        sys.stdout.write('{0},'.format(year))
    sys.stdout.write('\n')

    for r, data in all_results.items():
        sys.stdout.write('{0},'.format(r))
        for element in data:
            sys.stdout.write('{0:.4f},'.format(element))
        sys.stdout.write('\n')
