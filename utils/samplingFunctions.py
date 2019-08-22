import csv
from math import ceil, floor, log
import numpy as np
import os
import random

######################################################################
# Author: 	Ian Ludden
# Date: 	08 August 2019
# 
# samplingFunctions.py
# 
# Utility functions for sampling NCAA tournament seeds 
# from various distributions such as the truncated geometric 
# distribution. 
# 
# The parameters stored at the top of this script are copied from 
# the TruncGeomFits.xlsx file. 
# 
######################################################################
# Seed sets for each half-region
TOP_SEEDS_SORTED = [1, 4, 5, 8, 9, 12, 13, 16]
BOTTOM_SEEDS_SORTED = [2, 3, 6, 7, 10, 11, 14, 15]

######################################################################
	# Old parameters for truncated geometric fits without 
	# modifying some frequencies. 
	# 
	# # Truncated geometric parameters for sampling NCG seeds, 
	# # from 2013 (index 0) through 2020
	# pNcg = [0.4375, 0.436090226, 0.405405405, 0.413333333, 0.418300654, 0.425806452, 0.427672956, 0.429447853]
	# pSumNcg = [0.989977404, 0.989774678, 0.984376884, 0.985967621, 0.986890425, 0.988184109, 0.988487911, 0.98877044]

	# # Truncated geometric parameters for sampling Final Four seeds, 
	# # from 2013 (index 0) through 2020
	# USE_ADJUSTED_11 = False # Toggle using fit with 11-seeds removed and then added back in
	# pF4 = [0.379661017, 0.370607029, 0.362537764, 0.363636364, 0.359550562, 0.358695652, 0.354166667, 0.35443038]
	# pSumF4 = [0.999519088, 0.999393613, 0.999256516, 0.999276755, 0.999198769, 0.999181484, 0.999083933, 0.999089899]
	# pF4_adjusted11 = [0.416030534, 0.403571429, 0.39261745, 0.392857143, 0.386996904, 0.385074627, 0.388235294, 0.387464387]
	# pSumF4_adjusted11 = [0.999817088, 0.99974359, 0.999656919, 0.999659079, 0.99960244, 0.999582017, 0.999615098, 0.999607263]
	# pF4_prob11 = [0.025, 0.024, 0.022, 0.022, 0.021, 0.02, 0.027, 0.026]

	# # Truncated geometric parameters for sampling Elite Eight seeds, 
	# # from 2013 (index 0) through 2020
	# pE8Top = [0.643678161, 0.630434783, 0.625, 0.629441624, 0.63681592, 0.640776699, 0.623853211, 0.625]
	# pSumE8Top = [0.99974014, 0.99965204, 0.999608934, 0.999644489, 0.999697299, 0.999722722, 0.999599264, 0.999608934]
	# pE8Bottom = [0.466666667, 0.471544715, 0.465116279, 0.466165414, 0.463768116, 0.456747405, 0.453333333, 0.45751634]
	# pSumE8Bottom = [0.993453792, 0.993917726, 0.993299996, 0.99340441, 0.993163701, 0.992413971, 0.992024081, 0.992499447]
######################################################################

# Note: All parameters are rounded to four digits after the decimal

# NCG: Truncated geometric parameters for sampling NCG seeds, 
# from 2013 (index 0) through 2020
pNcg = [0.4375, 0.4361, 0.4214, 0.4478, 0.4526, 0.4604, 0.4615, 0.4626]
pSumNcg = [0.99, 0.9898, 0.9874, 0.9914, 0.9919, 0.9928, 0.9929, 0.993]
pNcg_choose8 = [0.0281, 0.0267, 0.0411, 0.0416, 0.0404, 0.0395, 0.0383, 0.037]

# F4_A: Truncated geometric parameters for sampling Final Four seeds, 
# from 2013 (index 0) through 2020
pF4A = [0.3797, 0.3706, 0.3625, 0.3636, 0.3596, 0.3587, 0.3619, 0.362]
pSumF4A = [0.9995, 0.9994, 0.9993, 0.9993, 0.9992, 0.9992, 0.9992, 0.9992]
pF4A_choose11 = [0.0237, 0.0223, 0.0211, 0.0203, 0.0193, 0.0186, 0.0255, 0.0246]

# F4_B: Truncated geometric parameters for sampling Final Four seeds, 
# from 2013 (index 0) through 2020
pF4B = [0.4565, 0.4519, 0.4545, 0.4612, 0.464, 0.4667, 0.4692, 0.4649]
pSumF4B = [0.9742, 0.9729, 0.9737, 0.9755, 0.9763, 0.977, 0.9776, 0.9765]
pF4B_chooseTop6 = [0.9375, 0.931, 0.9167, 0.9113, 0.9062, 0.9015, 0.8971, 0.9]

# E8: Truncated geometric parameters for sampling Elite Eight seeds, 
# from 2013 (index 0) through 2020
pE8Top = [0.4876, 0.4925, 0.4894, 0.4966, 0.4966, 0.5067, 0.471, 0.4717]
pSumE8Top = [0.9952, 0.9956, 0.9954, 0.9959, 0.9959, 0.9965, 0.9939, 0.9939]
pE8_choose1 = [0.4223, 0.3858, 0.3771, 0.3727, 0.3923, 0.3835, 0.413, 0.4154]
pE8Bottom = [0.4667, 0.4715, 0.4651, 0.4662, 0.4638, 0.4567, 0.4592, 0.4633]
pSumE8Bottom = [0.9935, 0.9939, 0.9933, 0.9934, 0.9932, 0.9924, 0.9927, 0.9931]
pE8_choose11 = [0.0249, 0.024, 0.0301, 0.0286, 0.0267, 0.0319, 0.0382, 0.0371]


def getTruncGeom(p, pSum, maxVal=None, pFixedSeed=None, fixedSeed=None):
	"""Samples from a (two-stage) truncated geometric random variable 
	   with parameter p and probabilities that add to pSum.

	   The formula is derived via inversion. 

	   Parameters
	   ----------
	   p : float
		   The parameter of the (truncated) geometric distribution, 
		   fitted using the method of moments
	   pSum : float
		   The sum of the geometric probabilities for the support of 
		   the truncated distribution
	   maxVal : int
	       The maximum possible value (used to catch rare rounding errors)
	   pFixedSeed : float
	       The probability of choosing the given fixedSeed instead of 
	       sampling from the truncated geometric distribution
	   fixedSeed : int
	       The seed to sample with probability pFixedSeed

	   Returns
	   -------
	   sampledValue : int
	       A single sample from the two-stage truncated geometric distribution
	"""
	# If using two stages, sample fixed seed with probability pFixedSeed
	if fixedSeed is not None:
		u = random.random()
		if u < pFixedSeed:
			return fixedSeed

	# Sample from truncated geometric by inversion
	u = random.random()
	sampledValue = int(ceil(log(1 - u) / log(1 - p)))
	if maxVal is not None:
		sampledValue = min(maxVal, sampledValue)
	return sampledValue


def sampleNCG(year):
	"""Randomly samples two seeds to compete in 
	   the National Championship Game (NCG).

	   Parameters
	   ----------
	   year : int
		   The year of the tournament to be predicted

	   Returns
	   -------
	   [seed0, seed1] : list of ints
		   The sampled seeds
	"""
	# 1. Load NCG params
	p = pNcg[year - 2013]
	pSum = pSumNcg[year - 2013]
	pChoose8 = pNcg_choose8[year - 2013]

	# 2. Get two independent samples from two-stage truncated geometric distribution
	seed0 = getTruncGeom(p, pSum, maxVal=8, pFixedSeed=pChoose8, fixedSeed=8)
	seed1 = getTruncGeom(p, pSum, maxVal=8, pFixedSeed=pChoose8, fixedSeed=8)

	return [seed0, seed1]


def sampleF4A(year):
	"""Randomly samples four seeds to compete in 
	   the Final Four (F4).
	   The distribution is modified for the 11-seed and 
	   a two-stage sampling procedure is used. 

	   Parameters
	   ----------
	   year : int
		   The year of the tournament to be predicted

	   Returns
	   -------
	   [seed0, seed1, seed2, seed3] : list of ints
		   The sampled seeds
	"""
	# 1. Load F4_A parameters
	p = pF4A[year - 2013]
	pSum = pSumF4A[year - 2013]
	pChoose11 = pF4A_choose11[year - 2013]

	# 2. Get four independent samples from two-stage trunc. geom. distribution
	seeds = []
	for regionIndex in range(4):
		seeds.append(getTruncGeom(p, pSum, maxVal=16, pFixedSeed=pChoose11, fixedSeed=11))

	return seeds


def sampleF4B(year):
	"""Randomly samples four seeds to compete in 
	   the Final Four (F4). 
	   Only seeds 1-12 are considered, with 
	   a truncated geometric distribution used for seeds 1-6 and 
	   a custom two-value distribution used for seeds 7-12. 

	   Parameters
	   ----------
	   year : int
		   The year of the tournament to be predicted

	   Returns
	   -------
	   [seed0, seed1, seed2, seed3] : list of ints
		   The sampled seeds
	"""
	# 1. Load F4_B parameters
	p = pF4B[year - 2013]
	pSum = pSumF4B[year - 2013]
	probTop6 = pF4B_chooseTop6[year - 2013]

	bottom6Seeds = [7, 8, 9, 10, 11, 12]
	bottom6Probs = np.array([2., 2., 1., 1., 2., 1.]) / 9

	# 2. Get four independent samples from two-stage distribution
	seeds = []
	for regionIndex in range(4):
		if random.random() < probTop6:
			seeds.append(getTruncGeom(p, pSum, maxVal=6))
		else:
			seeds.append(np.random.choice(bottom6Seeds, size=1, p=bottom6Probs))

	return seeds


def sampleE8(year):
	"""Randomly samples eight seeds to compete in 
	   the Elite Eight (E8). 

	   Parameters
	   ----------
	   year : int
		   The year of the tournament to be predicted

	   Returns
	   -------
	   [topSeed0, bottomSeed0, topSeed1, bottomSeed1, ...] : list of ints
		   The sampled seeds
	"""
	# 1. Load E8 parameters
	pTop = pE8Top[year - 2013]
	pSumTop = pSumE8Top[year - 2013]
	pChoose1 = pE8_choose1[year - 2013]
	pBottom = pE8Bottom[year - 2013]
	pSumBottom = pSumE8Bottom[year - 2013]
	pChoose11 = pE8_choose11[year - 2013]

	# 2. Get four independent samples each from top and bottom distributions
	seeds = []
	for regionIndex in range(4):
		fixedSeed = 1 # 0 is the index of 1 in TOP_SEEDS_SORTED; must add 1 to account for subtraction by 1
		topSeedIndex = getTruncGeom(pTop, pSumTop, maxVal=8, pFixedSeed=pChoose1, fixedSeed=fixedSeed) - 1
		seeds.append(TOP_SEEDS_SORTED[topSeedIndex])
		fixedSeed = 6 # 5 is the index of 11 in BOTTOM_SEEDS_SORTED; must add 1 to account for subtraction by 1
		bottomSeedIndex = getTruncGeom(pBottom, pSumBottom, maxVal=8, pFixedSeed=pChoose11, fixedSeed=fixedSeed) - 1
		seeds.append(BOTTOM_SEEDS_SORTED[bottomSeedIndex])

	return seeds

if __name__ == '__main__':
	SAMPLE_SIZE = 10000
	MIN_YEAR = 2013
	MAX_YEAR = 2020

	for samplingFn in [sampleNCG, sampleF4A, sampleF4B, sampleE8]:
		print(str(samplingFn))
		for year in range(MIN_YEAR, MAX_YEAR + 1):
			seedFreqs = np.zeros(17) # index 0 is unused
			for i in range(SAMPLE_SIZE):
				seeds = samplingFn(year)
				for seed in seeds:
					seedFreqs[seed] += 1

			total = np.sum(seedFreqs)
			print(year)
			print('Seed,Proportion')
			for seed in range(1, 17):
				print('{0},{1:.4f}'.format(seed, seedFreqs[seed] / total))
			print()

