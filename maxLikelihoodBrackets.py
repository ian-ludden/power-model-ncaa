import csv
from itertools import permutations
import json
import math
# import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint
import random

# from utils.scoringFunctions import scoreBracket

######################################################################
# Author: 	Ian Ludden
# Date: 	19 July 2019
# 
# maxLikelihoodBrackets.py
# 
# Given win probabilities for all pairs of teams, 
# computes the log-likelihood of brackets and sorts them from 
# most likely to least likely. 
# 
# Given a max pool size M, can then return the M most likely brackets.  
# 
######################################################################
NUM_REGION_VECTORS = 32768

# Historical brackets
historicalBrackets = {
	'2019': '100111011000100100111011000110100011011100111101101111100001010', 
	'2018': '001001110111110101111110001110101011011100111111101111110101010', 
	'2017': '111101110001000111101111010111101101111001110110111011000101000', 
	'2016': '101001111100100110101011000100101110111111111100101001011100111', 
	'2015': '111111111101111111110011010101111101110001000111100111110101000', 
	'2014': '100101011011111110111111001000110111111010100111100110010101001', 
	'2013': '110111111100101100010010010101111101101000011100111111000011111'
}


# Strings for certain brackets
specialBrackets = {
	# pick favorite
	'bracket1111': '111111111000101111111111000101111111111000101111111111000101111',
	
	# One 2-seed (region 0) wins in E8, then loses in F4. 
	'bracket1112': '111111111000100111111111000101111111111000101111111111000101011',

	# One 3-seed (region 0) beats 2 and 1, then loses in F4. 
	'bracket1113': '111111111000110111111111000101111111111000101111111111000101011',

	# Two 2-seeds (regions 0, 2) win in E8, then lose in F4. 
	'bracket1212': '111111111000100111111111000101111111111000100111111111000101001',

	# Two 2-seeds (regions 0, 1) win in E8, then play each other in F4. Winner loses NCG. 
	'bracket1122': '111111111000100111111111000100111111111000101111111111000101111'
}

# Dictionary of most likely region vector 
# (hex code and log-likelihood) given winner of region
mostLikelyRegions = {
	'1': ['5fc5', -5.3414], 
	'2': ['5fc4', -5.4454], 
	'3': ['5fc6', -5.8265], 
	'4': ['5fc1', -6.6376], 
	'5': ['5fe1', -7.0378], 
	'6': ['5fd6', -6.9956], 
	'7': ['5fcc', -7.4400], 
	'8': ['7f85', -8.4459]
}

# Predicted probability seed i defeats seed 17 - i in round 1, i = 1 to 8. 
# These are for predicting the 2020 tournament.
R1_PROBS = [None, 0.99, 0.94, 0.85, 0.79, 0.64, 0.63, 0.6, 0.5]

# Generic alpha values to use for each round
# (unused for round 1, since R1_PROBS is used)
# These are for predicting the 2020 tournament.
ALPHA_VALS = [None, 0.96, 1.09, 0.85, 0.11, 0.62, 1.23]

def applyRoundResults(seeds, results):
	"""Takes in a list of seeds that competed
	   in a round within a region, listed from top to bottom
	   in the official bracket format. It also takes a list
	   of results, where a 1 (0) indicates the top (bottom) 
	   team won. It outputs a list of the seeds in the next
	   round, i.e., the winners of the given round.
	"""
	nGames = len(results)
	return [seeds[2*i] * results[i] + seeds[2*i+1] * (1 - results[i]) for i in range(nGames)]


def regionVectorFromHex(regionHex):
	"""Convert a region vector's 4-digit hexadecimal representation
	   (with a leading 0) into a list of 15 0's or 1's. 
	"""
	bitString = bin(int(regionHex, 16))[2:].zfill(15)
	return [int(bitString[i]) for i in range(len(bitString))]


def prettifyRegionVector(regionHex):
	"""Returns a more descriptive string for the 
	   given 4-digit hex representation of a region vector. 
	"""
	regionVector = regionVectorFromHex(regionHex)
	seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
	r1Winners = applyRoundResults(seeds, regionVector[:8])
	r2Winners = applyRoundResults(r1Winners, regionVector[8:12])
	r3Winners = applyRoundResults(r2Winners, regionVector[12:14])
	r4Winner = applyRoundResults(r3Winners, regionVector[14:])
	return '{0} {1} {2} {3}'.format(r1Winners, r2Winners, r3Winners, r4Winner)


def getWinProbability(team1, team2, r):
	"""Returns the predicted probability that team1 defeats team2 
	   in the given round r. 
	   Can be modified to use different models. 

	   Arguments:
	   team1 - the "upper" team in the standard bracket representation
			   dict with seed and region
	   team2 - the "lower" team
			   dict with seed and region
	   r	 - the round number
			   integer from 1 to 6
	"""
	# Currently using Power Model
	s1 = team1['seed']
	s2 = team2['seed']

	# Use R1_PROBS for round 1
	if r == 1:
		if not (s1 + s2 == 17):
			exit('Invalid round 1 matchup: seeds {0} vs. {1}.'.format(s1, s2))
		return R1_PROBS[s1] if s1 < s2 else R1_PROBS[s2]
	
	# Use ALPHA_VALS for other rounds (unless seeds are same)
	if s1 == s2:
		return 0.5

	alpha = ALPHA_VALS[r]
	s1a = (s1 * 1.0) ** alpha
	s2a = (s2 * 1.0) ** alpha
	return s2a / (s1a + s2a)


def logLikelihood(bracket):
	"""Returns the log-likelihood of the given bracket using the 
	   win probabilities of the default model. 

	   Pr(bracket) = Pr(R6 | R5) * Pr(R5 | R4) * ... * Pr(R2 | R1) * Pr(R1)
	   
	   log(Pr(bracket)) = log(Pr(R1)) + sum_{i=2}^{6} log(Pr(R_i | R_{i-1}))

	   Could be updated to allow user to specify model 
	   to use in getWinProbability(). 

	   Also handles a single region vector, if provided

	   Arguments:
	   bracket - a list of 63 'bits' (0 or 1) in 'TTT' format, 
				 i.e., 1 if the 'upper' team wins
	"""
	totalLogProb = 0

	isSingleRegion = len(bracket) == 15

	# Rounds 1 through 4
	regionWinners = []
	for region in range(1 if isSingleRegion else 4):
		start = 15 * region # Offset for index of first game in region
		end = start + 8 	# Round 1 has 8 games per region
		regionVec = bracket[start:end]
		seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

		# Percolate teams through the first four rounds
		for r in range(1, 4 + 1):
			nGames = int(len(seeds) / 2)
			for game in range(nGames):
				team1 = {'seed': seeds[2 * game], 'region': region}
				team2 = {'seed': seeds[2 * game + 1], 'region': region}
				winProb = getWinProbability(team1, team2, r=r)
				if regionVec[game] == 1:
					totalLogProb += math.log(winProb)
					# if r == 4:
					# 	print('Adding {0:.3f} for {1} defeating {2} in round {3}.'.format(math.log(winProb), team1['seed'], team2['seed'], r))
				else:
					totalLogProb += math.log(1 - winProb)
					# if r == 4:
					# 	print('Adding {0:.3f} for {1} losing to {2} in round {3}.'.format(math.log(1 - winProb), team1['seed'], team2['seed'], r))

			# Update seeds for next round
			seeds = applyRoundResults(seeds, regionVec)
			start = end
			end += int(len(seeds) / 2)
			regionVec = bracket[start:end]

			# print('\nAt end of round {0} for region {2}, subtotal log-likelihood is {1:.3f}.\n'.format(r, totalLogProb, region))

		# Only remaining team in the region after four rounds 
		# is the region winner
		regionWinners.append(seeds[0])

	# If only a single region, then haven't reached Rounds 5 and 6
	if isSingleRegion:
		return totalLogProb

	# Round 5
	f4Game1 = bracket[-3]
	team1 = {'seed': regionWinners[0], 'region': 0}
	team2 = {'seed': regionWinners[1], 'region': 1}
	winProb = getWinProbability(team1, team2, r=5)
	if f4Game1 == 1:
		ncgTeam1 = team1
		totalLogProb += math.log(winProb)
		# print('Adding {0:.3f} for {1} defeating {2} in round {3}.'.format(math.log(winProb), team1['seed'], team2['seed'], 5))
	else:
		ncgTeam1 = team2
		totalLogProb += math.log(1 - winProb)
		# print('Adding {0:.3f} for {1} losing to {2} in round {3}.'.format(math.log(1 - winProb), team1['seed'], team2['seed'], 5))

	f4Game2 = bracket[-2]
	team1 = {'seed': regionWinners[2], 'region': 2}
	team2 = {'seed': regionWinners[3], 'region': 3}
	winProb = getWinProbability(team1, team2, r=5)
	if f4Game2 == 1:
		ncgTeam2 = team1
		totalLogProb += math.log(winProb)
		# print('Adding {0:.3f} for {1} defeating {2} in round {3}.'.format(math.log(winProb), team1['seed'], team2['seed'], 5))
	else:
		ncgTeam2 = team2
		totalLogProb += math.log(1 - winProb)
		# print('Adding {0:.3f} for {1} losing to {2} in round {3}.'.format(math.log(1 - winProb), team1['seed'], team2['seed'], 5))

	# Round 6
	ncg = bracket[-1]
	winProb = getWinProbability(ncgTeam1, ncgTeam2, r=6)
	if ncg == 1:
		totalLogProb += math.log(winProb)
		# print('Adding {0:.3f} for {1} defeating {2} in round {3}.'.format(math.log(winProb), ncgTeam1['seed'], ncgTeam2['seed'], 6))
	else:
		totalLogProb += math.log(1 - winProb)
		# print('Adding {0:.3f} for {1} losing to {2} in round {3}.'.format(math.log(1 - winProb), ncgTeam1['seed'], ncgTeam2['seed'], 6))

	return totalLogProb


def generatePossibleBrackets(M):
	"""Generates a set of brackets that is expected to contain 
	   the M most likely brackets. 
	   Not a very scientific approach; 
	   simply assumes the top four seeds all make it 
	   to the Sweet 16, and then considers what happens from there. 

	   Arguments: 
	   M - the number of most likely brackets we want to 'catch'
		   (currently unused)
	"""
	# TODO: I bet it's actually more likely that some 8 vs 9 games flip 
	# than the F4 seeds change. (Confirmed by region vector experiments)

	# Strings for regions, indexed by seed winning region. 
	# All other games are pick favorite
	regionStrings = [None, 
		'111111111000101', # seed 1 wins (over seed 2)
		'111111111000100', # 2 beats 1
		'111111111000110', # 3 beats 1
		'111111111000001'] # 4 beats 2

	# Final Four seed options to test
	finalFourPossibilities = ['1111', 
		'1112', '1121', '1211', '2111', 
		'1113', '1131', '1311', '3111', 
		'1122', '1212', '1221', '2112', '2121', '2211', 
		'1114', '1141', '1411', '4111', 
		'1123', '1132', '1312', '1213', 
		'1231', '1321', '2113', '3112', 
		'2131', '3121', '2311', '3211'
		]

	brackets = []
	for finalFourSeeds in finalFourPossibilities:
		f4seeds = [int(finalFourSeeds[i]) for i in range(4)]
		bracket = ''
		for region in range(4):
			bracket = bracket + regionStrings[f4seeds[region]]

		# Assume favorites win in rounds 5 and 6
		# f4Game1 = '1' if f4seeds[0] <= f4seeds[1] else '0'
		# bracket.append(f4Game1)
		# f4Game2 = '1' if f4seeds[2] <= f4seeds[3] else '0'
		# bracket.append(f4Game2)
		# ncg = '1' if np.min([f4seeds[0], f4seeds[1]]) <= np.min([f4seeds[2], f4seeds[3]]) else '0'
		# bracket.append(ncg)

		# Try all possible final 3 bits
		for i in range(8):
			last3String = '{0:03b}'.format(i)
			fullBracket = bracket + last3String
			brackets.append(fullBracket)

	return brackets


def generateAllRegionVectors():
	"""Returns a list of all possible 15-bit region vectors 
	   in TTT format as 4-digit hexadecimal strings with a leading 0. 
	"""
	regionVectors = []
	for i in range(NUM_REGION_VECTORS):
		regionVectors.append('{0:04x}'.format(i))
	return regionVectors


def evaluateAndSortBrackets(brackets):
	"""Computes the log-likelihood of each of the given brackets 
	   and sorts them in decreasing order of likelihood. 
	   Returns both the brackets and their log-likelihoods 
	   in a structured numpy array. 

	   Arguments:
	   brackets - A list of 63-bit lists in 'TTT' format
	"""
	logLhoods = []
	for bracketString in brackets:
		bracket = [int(bracketString[i]) for i in range(63)]
		logLhoods.append(logLikelihood(bracket))

	# Create array for bracket strings and log-likelihoods
	dt = np.dtype([('bracketString', np.unicode_, 63), ('log-lhood', np.float64, 1)])
	bracketsLogLhoods = np.array([(brackets[i], logLhoods[i]) for i in range(len(brackets))], dtype=dt)
	return np.sort(bracketsLogLhoods, order=['log-lhood'], axis=0)[::-1]


def evaluateAndSortRegionBrackets(regionStrings):
	"""Computes the log-likelihood of each of the given region 
	   strings and sorts them in decreasing order of likelihood. 
	   Returns both the region strings and their log-likelihoods 
	   in a structured numpy array. 

	   Arguments:
	   regionStrings - A list of 4-digit hexadecimal region strings 
	   				   in 'TTT' format with a leading zero
	"""
	logLhoods = []
	for regionString in regionStrings:
		regionVector = regionVectorFromHex(regionString)
		logLhoods.append(logLikelihood(regionVector))

	# Create array for bracket strings and log-likelihoods
	dt = np.dtype([('regionString', np.unicode_, 15), ('log-lhood', np.float64, 1)])
	regionLogLhoods = np.array([(regionStrings[i], logLhoods[i]) for i in range(len(regionStrings))], dtype=dt)
	return np.sort(regionLogLhoods, order=['log-lhood'], axis=0)[::-1]


def sampleBracketsPowerModel(nSamples):
	"""Samples the given number of brackets 
	   by applying the power model 
	   forward (from Round 1 through Round 6). 
	"""
	brackets = []
	for sampleIndex in range(nSamples):
		bracket = []
		regionWinners = np.zeros(4)
		for regionIndex in range(4):
			regionVector, regionWinners[regionIndex] = sampleRegionPowerModel()
			bracket += regionVector
		# 2. Select outcomes of F4/NCG games (Rounds 5, 6)
		team0 = {'seed': regionWinners[0], 'region': 0}
		team1 = {'seed': regionWinners[1], 'region': 1}
		team2 = {'seed': regionWinners[2], 'region': 2}
		team3 = {'seed': regionWinners[3], 'region': 3}
		winProb1 = getWinProbability(team0, team1, r=5)
		winProb2 = getWinProbability(team2, team3, r=5)
		f4Result1 = 1 if random.random() < winProb1 else 0
		f4Result2 = 1 if random.random() < winProb2 else 0
		bracket.append(f4Result1)
		bracket.append(f4Result2)
		ncgSeeds = applyRoundResults(regionWinners, [f4Result1, f4Result2])

		# NCG
		ncgTeam1 = {'seed': ncgSeeds[0], 'region': -1}
		ncgTeam2 = {'seed': ncgSeeds[1], 'region': -1}
		winProb = getWinProbability(ncgTeam1, ncgTeam2, r=6)
		ncgResult = 1 if random.random() < winProb else 0
		bracket.append(ncgResult)
		brackets.append(bracket)
	return brackets


def sampleRegionPowerModel():
	"""Samples a region vector using the basic power model. 
	   
	   Returns [regionVector, regionWinner], 
	   where regionVector is list of 15 0s/1s and 
	   regionWinner is the seed of the regional champion.
	"""
	regionVector = []
	# Loop through regional rounds R64, R32, and S16
	seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
	for roundNum in range(1, 5):
		numGames = int(len(seeds) / 2)
		newSeeds = []
		for gameNum in range(numGames):
			s1 = seeds[2 * gameNum]
			s2 = seeds[2 * gameNum + 1]
			p = getWinProbability({'seed': s1}, {'seed': s2}, r=roundNum)

			rnd = random.random()
			regionVector.append(1 if rnd < p else 0)
			newSeeds.append(s1 if rnd < p else s2)
		seeds = newSeeds

	return [regionVector, seeds[0]]


def sampleBracketsAsRegions(nSamples, T=100):
	"""Samples brackets by the following procedure:
	   1. Randomly sample four region vectors 
		  (i.i.d.) from the most likely T region vectors, 
		  according to the power model. 
	   2. Select the outcomes of the last three games (F4 and NCG) 
		  using the power model. 

	   Returns list brackets as lists of 63 0's and/or 1's. 

	   Arguments:
	   nSamples - the number of samples to generate
	   T - cutoff for which region vectors to sample
	"""
	brackets = []
	
	for sampleIndex in range(nSamples):
		# 1. Sample four region vectors
		regionVecs = []
		regionWinners = []
		for i in range(4):
			regionHex = sampleMLRegion(T)
			regionVector = regionVectorFromHex(regionHex)
			regionVecs.append(regionVector)
			# 1.1 Determine region winners (F4 seeds)
			seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
			r1Winners = applyRoundResults(seeds, regionVector[:8])
			r2Winners = applyRoundResults(r1Winners, regionVector[8:12])
			r3Winners = applyRoundResults(r2Winners, regionVector[12:14])
			r4Winner = applyRoundResults(r3Winners, regionVector[14:])
			regionWinners.append(r4Winner[0])

		# 2. Select outcomes of F4/NCG games (Rounds 5, 6)
		# F4
		team0 = {'seed': regionWinners[0], 'region': 0}
		team1 = {'seed': regionWinners[1], 'region': 1}
		team2 = {'seed': regionWinners[2], 'region': 2}
		team3 = {'seed': regionWinners[3], 'region': 3}
		winProb1 = getWinProbability(team0, team1, r=5)
		winProb2 = getWinProbability(team2, team3, r=5)
		f4Result1 = 1 if random.random() < winProb1 else 0
		f4Result2 = 1 if random.random() < winProb2 else 0
		ncgSeeds = applyRoundResults(regionWinners, [f4Result1, f4Result2])

		# NCG
		ncgTeam1 = {'seed': ncgSeeds[0], 'region': -1}
		ncgTeam2 = {'seed': ncgSeeds[1], 'region': -1}
		winProb = getWinProbability(ncgTeam1, ncgTeam2, r=6)
		ncgResult = 1 if random.random() < winProb else 0
		brackets.append(regionVecs[0] + regionVecs[1] + regionVecs[2] + regionVecs[3] + [f4Result1, f4Result2, ncgResult])

	return brackets


def sampleMLRegion(T):
	"""Randomly samples a region vector from among the T 
	   most likely according to the distribution defined 
	   by their predicted likelihoods. 
	"""
	# TODO: Probably better to implement 
	#	   the alias method eventually, 
	#	   especially if we want a large sample size.
	T = min(T, 32768) # Max number of region vectors is 32768
	inputFilepath = 'regionVectorsSortedTop100.csv' if T <= 100 else 'regionVectorsSortedAll.csv'
	with open(inputFilepath, 'r') as f:
		reader = csv.reader(f)
		headers = next(reader)
		rawData = list(reader)
	
	regionHexs = []
	lhoods = np.zeros(T)
	for i in range(T):
		item = rawData[i]
		regionHexs.append(item[0][2:])
		lhoods[i] = item[2]

	totalProb = np.sum(lhoods)
	scaledLhoods = np.divide(lhoods, totalProb)

	return random.choices(regionHexs, weights=scaledLhoods)[0]


def evaluateAllLastThreeGames(maxSeed):
	"""Generates all possible F4 seed selections 
	   given the max seed number and computes 
	   the log-likelihood of each possible 
	   set of results for the last three games
	   (conditioned on the given F4 seeds).
	"""
	# Generate all possible 4-digit strings in base maxSeed
	seedsWithRepetition = ''
	for seed in range(1, maxSeed + 1):
		seedsWithRepetition += 4 * str(seed)
	allPossibleStrings = set(permutations(seedsWithRepetition, 4))
	
	for possibleString in allPossibleStrings:
		regionWinners = [int(possibleString[i]) for i in range(4)]

		for j in range(8):
			f4Game1 = j % 2
			f4Game2 = int(j / 2) % 2
			ncg = int(j / 4) % 2
			totalLogProb = 0.

			# Round 5 (F4)
			team1 = {'seed': regionWinners[0], 'region': 0}
			team2 = {'seed': regionWinners[1], 'region': 1}
			winProb = getWinProbability(team1, team2, r=5)
			ncgTeam1 = team1 if f4Game1 == 1 else team2
			totalLogProb += math.log(winProb if f4Game1 == 1 else (1 - winProb))

			team1 = {'seed': regionWinners[2], 'region': 2}
			team2 = {'seed': regionWinners[3], 'region': 3}
			winProb = getWinProbability(team1, team2, r=5)
			ncgTeam2 = team1 if f4Game2 == 1 else team2
			totalLogProb += math.log(winProb if f4Game2 == 1 else (1 - winProb))

			# Round 6 (NCG)
			winProb = getWinProbability(ncgTeam1, ncgTeam2, r=6)
			totalLogProb += math.log(winProb if ncg == 1 else (1 - winProb))
			partialLogProb = totalLogProb
			for regionWinner in regionWinners:
				totalLogProb += mostLikelyRegions[str(regionWinner)][1]
			print('{0},{1:03b},{2:.4f},{3:.4f}'.format(regionWinners, j, partialLogProb, totalLogProb))
	pass


if __name__ == '__main__':
	# # First, test pick favorite bracket
	# bracket = [int(pickFavoriteString[i]) for i in range(63)]
	# print('Pick Favorite Log-likelihood: {0:.3f}'.format(logLikelihood(bracket)))

	# # Next, test all 1's bracket
	# bracket = [1] * 63
	# print('All Ones Log-likelihood: {0:.3f}'.format(logLikelihood(bracket)))
	
	# Test all "special" brackets
	# for bracketKey in specialBrackets.keys():
	# 	bracketString = specialBrackets[bracketKey]
	# 	bracket = [int(bracketString[i]) for i in range(63)]
	# 	print('{0} log-likelihood: {1:.3f}\n'.format(bracketKey, logLikelihood(bracket)))


	# brackets = generatePossibleBrackets(100)
	# print('Generated {0} brackets.'.format(len(brackets)))

	# # Generate all 32,768 possible region vectors and print with log-likelihoods
	# regionStrings = generateAllRegionVectors()
	# sortedArray = evaluateAndSortRegionBrackets(regionStrings)
	# for pair in sortedArray:
	# 	print('\"0x{0}\",{1:.4f},{2}'.format(pair[0], pair[1], prettifyRegionVector(pair[0])))

	# Sample some brackets using sampleBracketsAsRegions and see how they score
	# nBrackets = 50000

	# print('Most Likely Regions Model')
	# brackets = sampleBracketsAsRegions(nBrackets)

	# # print(sortedArray[0])
	# # print(sortedArray[1])
	# # print('...')
	# # print(sortedArray[-1])

	# for year in range(2013, 2019 + 1):
	# 	historicalVector = [int(historicalBrackets[str(year)][i]) for i in range(63)]
	# 	scores = []
	# 	for bracketVector in brackets:
	# 		scores.append(scoreBracket(bracketVector, historicalVector)[0])
	# 	scores.sort()
	# 	pprint(scores[-1])
	# 	print()

	# print('\nPower Model')
	# brackets = sampleBracketsPowerModel(nBrackets)

	# for year in range(2013, 2019 + 1):
	# 	historicalVector = [int(historicalBrackets[str(year)][i]) for i in range(63)]
	# 	scores = []
	# 	for bracketVector in brackets:
	# 		scores.append(scoreBracket(bracketVector, historicalVector)[0])
	# 	scores.sort()
	# 	pprint(scores[-1])
	# 	print()
	evaluateAllLastThreeGames(4)
