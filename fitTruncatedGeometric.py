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

def calcAllRoundCounts(brackets):
    # TODO: document
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


def optimizeE8(roundCounts):
    """Chooses the optimal adjustment to the counts for 
       1-seeds and 11-seeds (and the corresponding additional 
       probability of choosing each) to best fit 
       two-stage truncated geometric sampling methods 
       for the top and bottom E8 slots to 
       the historical E8 distributions.

       Parameters
       ----------
       roundCounts : 2-D array
           See calcAllRoundCounts for definition

       Returns
       -------
       freq1 : int
           The modified observed frequency of the 1-seeds
       p1 : float
           The probability with which to choose a 1-seed 
           (versus sampling from the top truncated geometric)
       chiSq1 : float
           The chi-squared goodness-of-fit test statistic 
           for the top E8 sampling distribution against actualFreqs
       freq11 : int
           The modified observed frequency of the 11-seeds
       p11 : float
           The probability with which to choose an 11-seed
           (versus sampling from the bottom truncated geometric)
       chiSq11 : float
           The chi-squared goodness-of-fit test statistic 
           for the bottom E8 sampling distribution against actualFreqs
    """
    # TODO
    actualFreqs = np.copy(roundCounts[4, :])
    topSeeds = [1, 4, 5, 8, 9, 12, 13, 16]
    bottomSeeds = [2, 3, 6, 7, 10, 11, 14, 15]
    topFreqs = [0] + [int(actualFreqs[i]) for i in topSeeds]
    bottomFreqs = [0] + [int(actualFreqs[i]) for i in bottomSeeds]

    freq1, pChoose1, chiSq1 = optimizeModFreq(actualFreqs=topFreqs, modifiedSeed=1, maxVal=8)
    freq11, pChoose11, chiSq11 = optimizeModFreq(actualFreqs=bottomFreqs, modifiedSeed=6, maxVal=8)
    return freq1, pChoose1, chiSq1, freq11, pChoose11, chiSq11


def optimizeF4(roundCounts):
    """Chooses the optimal adjustment to the counts for 
       11-seeds (and the additional probability of 
       choosing an 11-seed first) to best fit 
       a two-stage truncated geometric sampling method 
       to the historical F4 distribution.

       Parameters
       ----------
       roundCounts : 2-D array
           See calcAllRoundCounts for definition

       Returns
       -------
       freq11 : int
           The modified observed frequency of the 11-seeds
       pChoose11 : float
           The probability with which to choose an 11-seed
           (versus sampling from the bottom truncated geometric)
       chiSq : float
           The chi-squared goodness-of-fit test statistic 
           for the sampling distribution against actualFreqs
    """
    freq11, pChoose11, chiSq = optimizeModFreq(actualFreqs=np.copy(roundCounts[5, :]), modifiedSeed=11, maxVal=16)
    return freq11, pChoose11, chiSq


def fitTruncGeomF4top6(roundCounts):
    """Fits the truncated geometric distribution to 
       the historical distribution of seeds 1 to 6 
       in the Final Four. Used by the F4_B model. 

       Parameters
       ----------
       roundCounts : 2-D array
           See calcAllRoundCounts for definition

       Returns
       -------
       p : float
           The truncated geometric parameter (probability of success)
       pSum : float
           The sum of the geometric probabilities for the top 6 seeds 
           (needed to sample from the truncated geometric distribution 
           by inversion) 
    """
    p, pSum, pdf = getTruncatedGeometricPdf(actualCounts=np.copy(roundCounts[5, :7]), maxVal=6)
    return p, pSum, pdf


def optimizeNCG(roundCounts):
    """Chooses the optimal adjustment to the counts for 
       8-seeds (and the additional probability of 
       choosing an 8-seed first) to best fit 
       a two-stage truncated geometric sampling method 
       to the historical NCG distribution.

       Parameters
       ----------
       roundCounts : 2-D array
           See calcAllRoundCounts for definition

       Returns
       -------
       freq8 : int
           The modified observed frequency of the 8-seeds
       p8 : float
           The probability with which to choose an 8-seed
           (versus sampling from the bottom truncated geometric)
       chiSq : float
           The chi-squared goodness-of-fit test statistic 
           for the sampling distribution against actualFreqs
    """
    freq8, pChoose8, chiSq = optimizeModFreq(actualFreqs=roundCounts[6, :9], modifiedSeed=8, maxVal=8)
    return freq8, pChoose8, chiSq


def chiSquared(actualCounts, expCounts):
    """Returns the chi-squared test statistic for the given 
       observed (actual) and expected frequencies.
    """
    return np.sum(np.power(np.subtract(actualCounts, expCounts), 2.) / expCounts)


def calcAdjustedChiSquared(modFreqs, actualFreqs, maxVal, modifiedSeed):
    """Computes the chi-squared value for the 
       expected counts from the two-stage truncated geometric
       sampling procedure. 
       The probability of choosing the modified seed is chosen 
       to match the overall expected count to the actual observed count 
       for that seed. 

       Parameters
       ----------
       modFreqs : 1-D array
           Modified frequencies (same as actualFreqs except at modifiedSeed)
       actualFreqs : 1-D array
           The actual observed frequencies
       maxVal : int
           The maximum value taken by the truncated geometric random variable
       modifiedSeed : int
           The seed whose frequency has been modified in modFreqs

       Returns
       -------
       adjustedChiSquared : float
           The chi-squared value for the overall expected counts from
           the two-stage sampling procedure against actualFreqs
    """
    totalCountActual = np.sum(actualFreqs)
    p, pSum, pdf = getTruncatedGeometricPdf(modFreqs, maxVal=maxVal)
    pChooseModSeed = (actualFreqs[modifiedSeed] - pdf[modifiedSeed] * totalCountActual) / (totalCountActual * (1 - pdf[modifiedSeed]))
    pChooseModSeed = max(0, pChooseModSeed)
    newPdf = pdf * (1 - pChooseModSeed)
    newPdf[modifiedSeed] += pChooseModSeed
    newExpCounts = newPdf * totalCountActual
    return chiSquared(actualFreqs[1:], newExpCounts[1:])


def getTruncatedGeometricPdf(actualCounts, maxVal):
    """Computes the truncated geometric pdf given 
       the observed frequencies.
       
       Parameters
       ----------
       actualCounts : 1-D array
           The observed seed frequencies; actualCounts[1] is # 1-seeds
       maxVal : int
           The maximum value of the truncated geometric random variable

       Returns
       -------
       p : float
           The truncated geometric parameter
       pSum : float
           The sum of the geometric distribution probabilities with 
           parameter p from 1 through maxVal
       pdf : 1-D array
           The truncated geometric pdf (same length as actualCounts)
    """
    totalCount = np.sum(actualCounts)
    weightedAvg = np.sum([i * actualCounts[i] for i in range(maxVal + 1)]) / totalCount
    p = 1. / weightedAvg
    geomPdf = [0.] + [(1-p)**(i-1) * p for i in range(1, maxVal + 1)]
    pSum = np.sum(geomPdf)
    return p, pSum, geomPdf / pSum


def optimizeModFreq(actualFreqs, modifiedSeed, maxVal):
    """Optimizes the modified frequency of modifiedSeed 
       for a two-stage truncated geometric sampling procedure 
       in which the modified seed is selected with some fixed 
       probability, and a truncated geometric sample is generated 
       otherwise.

       Parameters
       ----------
       actualFreqs : 1-D array
           The actual observed frequencies
       modifiedSeed : int
           The seed whose frequency will be modified
       maxVal : int
           The maximum value of the truncated geometric random variable

       Returns
       -------
       modFreq : int
           The modified frequency of modifiedSeed
       pChooseModSeed : float
           The fixed probability of sampling modifiedSeed in the first stage
       chiSq : float
           The chi-squared goodness-of-fit test statistic 
           for the sampling distribution against actualFreqs
    """
    totalCountActual = np.sum(actualFreqs)
    freqs = np.copy(actualFreqs)
    freqLB = 0
    freqUB = int(actualFreqs[modifiedSeed])

    # Find modified frequency that minimizes chi-squared
    bestChiSq = 1e6
    bestFreq = 0
    for newFreq in range(freqLB, freqUB + 1):
        freqs[modifiedSeed] = newFreq
        newChiSq = calcAdjustedChiSquared(modFreqs=freqs, actualFreqs=actualFreqs, maxVal=maxVal, modifiedSeed=modifiedSeed)
        if newChiSq < bestChiSq:
            bestChiSq = newChiSq
            bestFreq = newFreq

    freqs[modifiedSeed] = bestFreq
    p, pSum, pdf = getTruncatedGeometricPdf(freqs, maxVal=maxVal)
    pChooseModSeed = (actualFreqs[modifiedSeed] - pdf[modifiedSeed] * totalCountActual) / (totalCountActual * (1 - pdf[modifiedSeed]))
    chiSq = calcAdjustedChiSquared(modFreqs=freqs, actualFreqs=actualFreqs, maxVal=maxVal, modifiedSeed=modifiedSeed)

    return bestFreq, pChooseModSeed, chiSq


if __name__ == '__main__':
    minYear = 2013
    maxYear = 2020
    nYears = maxYear - minYear + 1

    bracketsAll = load_ref_brackets()
    roundCounts = np.zeros((nYears, 8, 17))
    for year in range(minYear, maxYear + 1):
        bracketsBeforeYear = [b for x, b in bracketsAll.items() if x < year]
        roundCounts[year - 2013, :, :] = calcAllRoundCounts(bracketsBeforeYear)

    # Print E8 results
    print('Elite Eight,Modified 1 and 11')
    for year in range(minYear, maxYear + 1):
        print(year)
        roundCountsYear = roundCounts[year - 2013, :, :]
        freqs = roundCountsYear[4, :]
        topSeeds = [1, 4, 5, 8, 9, 12, 13, 16]
        bottomSeeds = [2, 3, 6, 7, 10, 11, 14, 15]
        topFreqs = [0] + [int(freqs[i]) for i in topSeeds]
        bottomFreqs = [0] + [int(freqs[i]) for i in bottomSeeds]
        sumTopFreqs = np.sum(topFreqs)
        sumBottomFreqs = np.sum(bottomFreqs)

        freq1_E8, pChoose1_E8, chiSq1_E8, freq11_E8, pChoose11_E8, chiSq11_E8 = optimizeE8(roundCountsYear)
        
        # Print 'top' tables
        print('Top')
        modTopFreqs = np.copy(topFreqs)
        modTopFreqs[1] = freq1_E8
        print('Seed,Count,TG pdf,Exp. Count,Chi-Squared,,' * 2)
        p, pSum, pdf = getTruncatedGeometricPdf(actualCounts=topFreqs, maxVal=8)
        pMod, pSumMod, pdfMod = getTruncatedGeometricPdf(actualCounts=modTopFreqs, maxVal=8)
        expCounts = pdf * sumTopFreqs
        newPdf = pdf * (1 - pChoose1_E8)
        newPdf[1] += pChoose1_E8
        newExpCounts = newPdf * sumTopFreqs

        for index in range(1, 9):
            chiSq = (topFreqs[index] - expCounts[index]) ** 2 / expCounts[index]
            chiSqMod = (topFreqs[index] - newExpCounts[index]) ** 2 / newExpCounts[index]
            sys.stdout.write('{0},{1},{2:.4f},{3:.4f},{4:.4f},,'.format(topSeeds[index - 1], topFreqs[index], pdf[index], expCounts[index], chiSq))
            sys.stdout.write('{0},{1},{2:.4f},{3:.4f},{4:.4f},,'.format(topSeeds[index - 1], modTopFreqs[index], newPdf[index], newExpCounts[index], chiSqMod))
            sys.stdout.write('\n')
        
        print(',p,pSum,,chi-sq. sum,,,p,pSum,pChoose1,chi-sq. sum')
        sys.stdout.write(',{0:.4f},{1:.4f},,{2:.4f},,'.format(p, pSum, chiSquared(topFreqs[1:], expCounts[1:])))
        sys.stdout.write(',{0:.4f},{1:.4f},{3:.4f},{2:.4f},,\n'.format(pMod, pSumMod, chiSq1_E8, pChoose1_E8))
        print()

        # Print 'bottom' tables
        print('Bottom')
        modBottomFreqs = np.copy(bottomFreqs)
        modBottomFreqs[6] = freq11_E8 # 6 is index of 11 in bottomSeeds
        print('Seed,Count,TG pdf,Exp. Count,Chi-Squared,,' * 2)
        p, pSum, pdf = getTruncatedGeometricPdf(actualCounts=bottomFreqs, maxVal=8)
        pMod, pSumMod, pdfMod = getTruncatedGeometricPdf(actualCounts=modBottomFreqs, maxVal=8)
        expCounts = pdf * sumBottomFreqs
        newPdf = pdf * (1 - pChoose11_E8)
        newPdf[6] += pChoose11_E8
        newExpCounts = newPdf * sumBottomFreqs

        for index in range(1, 9):
            chiSq = (bottomFreqs[index] - expCounts[index]) ** 2 / expCounts[index]
            chiSqMod = (bottomFreqs[index] - newExpCounts[index]) ** 2 / newExpCounts[index]
            sys.stdout.write('{0},{1},{2:.4f},{3:.4f},{4:.4f},,'.format(bottomSeeds[index - 1], bottomFreqs[index], pdf[index], expCounts[index], chiSq))
            sys.stdout.write('{0},{1},{2:.4f},{3:.4f},{4:.4f},,'.format(bottomSeeds[index - 1], modBottomFreqs[index], newPdf[index], newExpCounts[index], chiSqMod))
            sys.stdout.write('\n')
        
        print(',p,pSum,,chi-sq. sum,,,p,pSum,pChoose11,chi-sq. sum')
        sys.stdout.write(',{0:.4f},{1:.4f},,{2:.4f},,'.format(p, pSum, chiSquared(bottomFreqs[1:], expCounts[1:])))
        sys.stdout.write(',{0:.4f},{1:.4f},{3:.4f},{2:.4f},,\n'.format(pMod, pSumMod, chiSq11_E8, pChoose11_E8))
        print()

    # Print F4_A results
    print('Final Four A,Modified 11')
    for year in range(minYear, maxYear + 1):
        print(year)
        roundCountsYear = roundCounts[year - 2013, :, :]
        freqs = roundCountsYear[5, :]
        sumFreqs = np.sum(freqs)
        freq11_F4, pChoose11_F4, chiSq11_F4 = optimizeF4(roundCountsYear)
        modFreqs = np.copy(freqs)
        modFreqs[11] = freq11_F4
        print('Seed,Count,TG pdf,Exp. Count,Chi-Squared,,' * 2)
        p, pSum, pdf = getTruncatedGeometricPdf(actualCounts=freqs, maxVal=16)
        pMod, pSumMod, pdfMod = getTruncatedGeometricPdf(actualCounts=modFreqs, maxVal=16)
        expCounts = pdf * sumFreqs
        newPdf = pdf * (1 - pChoose11_F4)
        newPdf[11] += pChoose11_F4
        newExpCounts = newPdf * sumFreqs

        for seed in range(1, 17):
            chiSq = (freqs[seed] - expCounts[seed]) ** 2 / expCounts[seed]
            chiSqMod = (freqs[seed] - newExpCounts[seed]) ** 2 / newExpCounts[seed]
            sys.stdout.write('{0},{1},{2:.4f},{3:.4f},{4:.4f},,'.format(seed, freqs[seed], pdf[seed], expCounts[seed], chiSq))
            sys.stdout.write('{0},{1},{2:.4f},{3:.4f},{4:.4f},,'.format(seed, modFreqs[seed], newPdf[seed], newExpCounts[seed], chiSqMod))
            sys.stdout.write('\n')
        
        print(',p,pSum,,chi-sq. sum,,,p,pSum,pChoose11,chi-sq. sum')
        sys.stdout.write(',{0:.4f},{1:.4f},,{2:.4f},,'.format(p, pSum, chiSquared(freqs[1:], expCounts[1:])))
        sys.stdout.write(',{0:.4f},{1:.4f},{3:.4f},{2:.4f},,\n'.format(pMod, pSumMod, chiSq11_F4, pChoose11_F4))
        print()

    # Print F4_B results
    print('Final Four B,Trunc. Geom. for 1-6,two-value distr. for 7-12,13-16 ignored')
    for year in range(minYear, maxYear + 1):
        print(year)
        roundCountsYear = roundCounts[year - 2013, :, :]
        freqsTop6 = roundCountsYear[5, :7]
        sumFreqsTop6 = np.sum(freqsTop6)
        p, pSum, pdfTop6 = fitTruncGeomF4top6(roundCountsYear)
        expCountsTop6 = pdfTop6 * sumFreqsTop6

        freqsBottom6 = roundCountsYear[5, 7:13]
        sumFreqsBottom6 = np.sum(freqsBottom6)
        pdfBottom6 = np.array([2./9, 2./9, 1./9, 1./9, 2./9, 1./9]) # index 0 is for 7-seed
        expCountsBottom6 = pdfBottom6 * sumFreqsBottom6

        proportionTop6 = sumFreqsTop6 * 1. / (sumFreqsTop6 + sumFreqsBottom6)
        print('Pr(sample 1-6),{0:.4f},Pr(sample 7-12),{1:.4f}'.format(proportionTop6, 1 - proportionTop6))
        print('Seed,Count,TG pdf,Exp. Count,Chi-Squared,,Seed,Count,Custom pdf,Exp. Count,Chi-Squared,,')
        for index in range(1, 7):
            chiSqTop6 = (freqsTop6[index] - expCountsTop6[index]) ** 2 / expCountsTop6[index]
            chiSqBottom6 = (freqsBottom6[index - 1] - expCountsBottom6[index - 1]) ** 2 / expCountsBottom6[index - 1]
            sys.stdout.write('{0},{1},{2:.4f},{3:.4f},{4:.4f},,'.format(index, freqsTop6[index], pdfTop6[index], expCountsTop6[index], chiSqTop6))
            sys.stdout.write('{0},{1},{2:.4f},{3:.4f},{4:.4f},,'.format(index + 6, freqsBottom6[index - 1], pdfBottom6[index - 1], expCountsBottom6[index - 1], chiSqBottom6))
            sys.stdout.write('\n')
        
        print(',p,pSum,,chi-sq. sum,,,,,,chi-sq. sum')
        sys.stdout.write(',{0:.4f},{1:.4f},,{2:.4f},,'.format(p, pSum, chiSquared(freqsTop6[1:], expCountsTop6[1:])))
        sys.stdout.write(',,,,{0:.4f},,\n'.format(chiSquared(freqsBottom6, expCountsBottom6)))
        print()

    # Print NCG results
    print('National Championship Game,Modified 8')
    for year in range(minYear, maxYear + 1):
        print(year)
        roundCountsYear = roundCounts[year - 2013, :, :]
        freqs = roundCountsYear[6, :9]
        sumFreqs = np.sum(freqs)
        freq8_NCG, pChoose8_NCG, chiSq8_NCG = optimizeNCG(roundCountsYear)
        modFreqs = np.copy(freqs)
        modFreqs[8] = freq8_NCG
        print('Seed,Count,TG pdf,Exp. Count,Chi-Squared,,' * 2)
        p, pSum, pdf = getTruncatedGeometricPdf(actualCounts=freqs, maxVal=8)
        pMod, pSumMod, pdfMod = getTruncatedGeometricPdf(actualCounts=modFreqs, maxVal=8)
        expCounts = pdf * sumFreqs
        newPdf = pdf * (1 - pChoose8_NCG)
        newPdf[8] += pChoose8_NCG
        newExpCounts = newPdf * sumFreqs

        for seed in range(1, 9):
            chiSq = (freqs[seed] - expCounts[seed]) ** 2 / expCounts[seed]
            chiSqMod = (freqs[seed] - newExpCounts[seed]) ** 2 / newExpCounts[seed]
            sys.stdout.write('{0},{1},{2:.4f},{3:.4f},{4:.4f},,'.format(seed, freqs[seed], pdf[seed], expCounts[seed], chiSq))
            sys.stdout.write('{0},{1},{2:.4f},{3:.4f},{4:.4f},,'.format(seed, modFreqs[seed], newPdf[seed], newExpCounts[seed], chiSqMod))
            sys.stdout.write('\n')
        
        print(',p,pSum,,chi-sq. sum,,,p,pSum,pChoose8,chi-sq. sum')
        sys.stdout.write(',{0:.4f},{1:.4f},,{2:.4f},,'.format(p, pSum, chiSquared(freqs[1:], expCounts[1:])))
        sys.stdout.write(',{0:.4f},{1:.4f},{3:.4f},{2:.4f},,\n'.format(pMod, pSumMod, chiSq8_NCG, pChoose8_NCG))
        print()