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


def createAndSaveBracketPool(size, year=2020, model='power', r=1, samplingFnName=None, filepath=None):
	"""Calls generateBracketPool with the given parameters and saves the results to a JSON file.
	"""
	brackets = generateBracketPool(size, year, model, r, samplingFnName)

	if filepath is None:
		homeDir = os.path.expanduser('~')
		filepath = '{0}/Documents/GitHub/power-model-ncaa/out'.format(homeDir)
		if not os.path.exists(filepath):
			os.makedirs(filepath)
		filepath += '/{0}_'.format(model)
		filepath += '{0}_'.format(size) if size < 1000 else '{0}k_'.format(size // 1000)
		filepath += '{0}'.format(year)
		if model == 'power':
			filepath += '_{0}_{1}'.format(r, samplingFnName)
		filepath += '.json'

	outputDict = {'year': year, 'size': size, 'model': model, 
					'r': r, 'samplingFnName': samplingFnName, 
					'brackets': brackets}

	with open(filepath, 'w') as outputFile:
		outputFile.write(json.dumps(outputDict))


if __name__ == '__main__':
	size = 10
	for year in range(2013, 2020):
		# Bradley-Terry
		createAndSaveBracketPool(size, year=year, model='bradley-terry')

		# Power: r = 1
		createAndSaveBracketPool(size, year=year, model='power')

		# Power: r = 4
		createAndSaveBracketPool(size, year=year, model='power', r=4, samplingFnName='sampleE8')

		# Power: r = 5, sampleF4
		createAndSaveBracketPool(size, year=year, model='power', r=5, samplingFnName='sampleF4')

		# Power: r = 5, sampleF4adjusted11
		createAndSaveBracketPool(size, year=year, model='power', r=5, samplingFnName='sampleF4adjusted11')

		# Power: r = 6
		createAndSaveBracketPool(size, year=year, model='power', r=6, samplingFnName='sampleNCG')