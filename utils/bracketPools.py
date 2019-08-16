import csv
import json
from math import ceil, floor, log
import os
import random

import bracketGenerators as bg
import bracketManipulations as bm

######################################################################
# Author: 	Ian Ludden
# Date: 	14 August 2019
# 
# bracketPools.py
# 
# This script generates and evaluates pools of brackets using 
# the generators in bracketGenerators.py. 
# 
######################################################################

def generateBracketPool(size, year=2020, model='power', r=-1, samplingFnName=None):
	"""Generates a pool of brackets with the given parameters.

	   Parameters
	   ----------
	   size : int
	       The number of brackets to generate

	   year : int
	       The tournament year to predict

	   model : string
	       The name of the model ('power' or 'bradley-terry')

	   r : int
	       The round to fix (only for power model)

	   samplingFnName : string
	       The name of the sampling function from samplingFunctions.py to use (only for power model)

	   Returns
	   -------
	   brackets : list
	       A list of brackets, each as a 16-digit hexadecimal string (leading 0)
	"""
	brackets = []

	for index in range(size):
		if model == 'power':
			newBracket = bg.generateBracketPower(year, r, samplingFnName)
		else: # 'bradley-terry'
			newBracket = bg.generateBracketBradleyTerry(year)
		
		brackets.append(bm.stringToHex(bm.vectorToString(newBracket)))

	return brackets


def createAndSaveBracketPool(sampleSize, year=2020, model='power', r=1, samplingFnName=None, filepath=None, nReplications=1):
	"""Calls generateBracketPool with the given parameters and saves the results to a JSON file.
	"""
	brackets = []
	for replicationIndex in range(nReplications):
		brackets.append(generateBracketPool(sampleSize, year, model, r, samplingFnName))

	if filepath is None:
		filepath = generateFilepath(sampleSize, year=year, model=model, r=r, samplingFnName=samplingFnName, nReplications=nReplications)

	outputDict = {'year': year, 'sampleSize': sampleSize, 
					'nReplications': nReplications, 'model': model, 
					'r': r, 'samplingFnName': samplingFnName, 
					'brackets': brackets}

	with open(filepath, 'w') as outputFile:
		outputFile.write(json.dumps(outputDict))


def generateFilepath(sampleSize, year=2020, model='power', r=1, samplingFnName=None, nReplications=1):
	"""Generates the path to the JSON file containing the experiment batch with the given parameters."""
	homeDir = os.path.expanduser('~')
	filepath = '{0}/Documents/GitHub/power-model-ncaa/out'.format(homeDir)
	if not os.path.exists(filepath):
		os.makedirs(filepath)
	filepath += '/{0}_'.format(model)
	filepath += '{0}_x_'.format(nReplications)
	filepath += '{0}_'.format(sampleSize) if sampleSize < 1000 else '{0}k_'.format(sampleSize // 1000)
	filepath += '{0}'.format(year)
	if model == 'power':
		filepath += '_{0}_{1}'.format(r, samplingFnName)
	filepath += '.json'
	return filepath


def runSamples(nReplications, sampleSize):
	"""Generates bracket pool samples for the 
	   power model paper experiments."""
	for year in range(2013, 2020):
		# Bradley-Terry
		createAndSaveBracketPool(sampleSize, year=year, model='bradley-terry', 
			nReplications=nReplications)

		# Power: r = 1
		createAndSaveBracketPool(sampleSize, year=year, model='power', 
			nReplications=nReplications)

		# Power: r = 4
		createAndSaveBracketPool(sampleSize, year=year, model='power', r=4, 
			samplingFnName='sampleE8', nReplications=nReplications)

		# Power: r = 5, sampleF4
		createAndSaveBracketPool(sampleSize, year=year, model='power', r=5, 
			samplingFnName='sampleF4', nReplications=nReplications)

		# Power: r = 5, sampleF4adjusted11
		createAndSaveBracketPool(sampleSize, year=year, model='power', r=5, 
			samplingFnName='sampleF4adjusted11', nReplications=nReplications)

		# Power: r = 6
		createAndSaveBracketPool(sampleSize, year=year, model='power', r=6, 
			samplingFnName='sampleNCG', nReplications=nReplications)


def readAndScore(nReplications, sampleSize):
	"""Reads the JSON files for all experiment batches, 
	   scores the brackets, and 
	   stores the Max Score and ESPN Count results 
	   in a CSV file."""
	# TODO: implement
	pass


if __name__ == '__main__':
	sampleSize = 10
	nReplications = 25
	runSamples(nReplications=nReplications, sampleSize=sampleSize)
	readAndScore(nReplications=nReplications, sampleSize=sampleSize)
