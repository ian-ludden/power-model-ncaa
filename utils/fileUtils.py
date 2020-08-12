import csv
import os
import numpy as np

######################################################################
# Author: 	Ian Ludden
# Date: 	08 August 2019
# 
# fileUtils.py
# 
# Utility functions for reading/writing files related to NCAA bracket 
# generation and analysis. 
# 
######################################################################

def getCsvRaw(filepath):
	"""Returns a csv file as raw data."""
	if not os.path.exists(filepath):
		exit('Cannot find file with name \'{0}\'.'.format(filepath))

	with open(filepath, 'r') as f:
		reader = csv.reader(f)
		rawData = list(reader)

	return rawData


def loadPowerParams(filepath):
	"""Loads Round 1 alpha values and weighted average alpha-values for rounds 2-6."""
	rawData = getCsvRaw(filepath)

	# First dimension is year - 2013, second dimension is seed1
	r1Alphas = np.zeros((8, 9))
	# Load r1Alphas
	for year in range(2013, 2021):
		startIndex = 18 * (year - 2013) + 1
		for i in range(8):
			seed1 = int(rawData[startIndex + 2 * i][0])
			alpha = float(rawData[startIndex + 2 * i + 1][3])
			r1Alphas[year - 2013][seed1] = alpha

	# First dimension is year - 2013, second dimension is round number (2 to 6; 0 and 1 unused)
	alphaVals = np.zeros((8, 7))
	# Load alphaVals
	for r in range(2, 7):
		allYears = rawData[r + 143][1:]
		for year in range(2013, 2021):
			alphaVals[year - 2013][r] = float(allYears[year - 2013])

	return r1Alphas, alphaVals


def loadBradleyTerryParams(filepath):
	"""Loads Bradley-Terry parameters (pairwise win probabilities)."""
	rawData = getCsvRaw(filepath)

	# First dimension is year - 2013, second is seed1, third is seed2
	btProbs = np.zeros((8, 17, 17))

	for year in range(2013, 2021):
		startIndex = 17 * (year - 2013) + 1
		for seed1 in range(1, 17):
			for seed2 in range(1, 17):
				btProbs[year - 2013][seed1][seed2] = rawData[startIndex + (seed1 - 1)][seed2 - 1]
	
	return btProbs