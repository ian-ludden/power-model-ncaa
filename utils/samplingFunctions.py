import csv
import math
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
# Truncated geometric parameters for sampling NCG seeds, 
# from 2013 (index 0) through 2020
pNcg = [0.4375, 0.436090226, 0.405405405, 0.413333333, 0.418300654, 0.425806452, 0.427672956, 0.429447853]
pSumNcg = [0.989977404, 0.989774678, 0.984376884, 0.985967621, 0.986890425, 0.988184109, 0.988487911, 0.98877044]

# Truncated geometric parameters for sampling Final Four seeds, 
# from 2013 (index 0) through 2020
USE_ADJUSTED_11 = False # Toggle using fit with 11-seeds removed and then added back in
pF4 = [0.379661017, 0.370607029, 0.362537764, 0.363636364, 0.359550562, 0.358695652, 0.354166667, 0.35443038]
pSumF4 = [0.999519088, 0.999393613, 0.999256516, 0.999276755, 0.999198769, 0.999181484, 0.999083933, 0.999089899]
pF4_adjusted11 = [0.416030534, 0.403571429, 0.39261745, 0.392857143, 0.386996904, 0.385074627, 0.388235294, 0.387464387]
pSumF4_adjusted11 = [0.999817088, 0.99974359, 0.999656919, 0.999659079, 0.99960244, 0.999582017, 0.999615098, 0.999607263]
pF4_prob11 = [0.025, 0.024, 0.022, 0.022, 0.021, 0.02, 0.027, 0.026]

# Truncated geometric parameters for sampling Elite Eight seeds, 
# from 2013 (index 0) through 2020
pE8Top = [0.643678161, 0.630434783, 0.625, 0.629441624, 0.63681592, 0.640776699, 0.623853211, 0.625]
pSumE8Top = [0.99974014, 0.99965204, 0.999608934, 0.999644489, 0.999697299, 0.999722722, 0.999599264, 0.999608934]
pE8Bottom = [0.466666667, 0.471544715, 0.465116279, 0.466165414, 0.463768116, 0.456747405, 0.453333333, 0.45751634]
pSumE8Bottom = [0.993453792, 0.993917726, 0.993299996, 0.99340441, 0.993163701, 0.992413971, 0.992024081, 0.992499447]


def getTruncGeom(p, pSum):
	"""Samples from a truncated geometric random variable 
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
	"""
    u = random.random() * pSum
    return int(ceil(log(1 - u) / log(1 - p)))


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

	# 2. Get two independent samples from truncated geometric distribution
	seed0 = getTruncGeom(p, pSum)
	seed1 = getTruncGeom(p, pSum)

	return [seed0, seed1]


def sampleF4(year):
	"""Randomly samples four seeds to compete in 
	   the Final Four (F4).

	   Parameters
	   ----------
	   year : int
	       The year of the tournament to be predicted

	   Returns
	   -------
	   [seed0, seed1, seed2, seed3] : list of ints
	       The sampled seeds
	"""
	# 1. Load F4 parameters
	p = pF4[year - 2013]
	pSum = pSumF4[year - 2013]

	# 2. Get four independent samples from trunc. geom. distribution
	seeds = []
	for regionIndex in range(4):
		seeds.append(getTruncGeom(p, pSum))

	return seeds


def sampleF4adjusted11(year):
	"""Randomly samples four seeds to compete in 
	   the Final Four (F4), with an adjustment made 
	   for the 11-seeds. 

	   Parameters
	   ----------
	   year : int
	       The year of the tournament to be predicted

	   Returns
	   -------
	   [seed0, seed1, seed2, seed3] : list of ints
	       The sampled seeds
	"""
	# 1. Load F4_adjusted11 parameters
	p = pF4_adjusted11[year - 2013]
	pSum = pSumF4_adjusted11[year - 2013]
	prob11 = pF4_prob11[year - 2013]

	# 2. Get four independent samples from two-stage distribution
	seeds = []
	for regionIndex in range(4):
		if random.random() < prob11:
			seeds.append(11)
		else:
			seeds.append(getTruncGeom(p, pSum))

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
	pBottom = pE8Bottom[year - 2013]
	pSumBottom = pSumE8Bottom[year - 2013]

	# 2. Get four independent samples each from top and bottom distributions
	seeds = []
	for regionIndex in range(4):
		seeds.append(getTruncGeom(pTop, pSumTop))
		seeds.append(getTruncGeom(pBottom, pSumBottom))

	return seeds
