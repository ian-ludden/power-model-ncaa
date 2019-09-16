import csv
import json
from math import ceil, floor, log
import numpy as np
import os
from pprint import pprint
import random
import sys

import bracketGenerators as bg
import bracketManipulations as bm
import scoringFunctions as sf

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


# total number of methods of generation
NUM_GENERATORS = 7



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

        if samplingFnName == "samplePower8Brute":
                # special case, one call to bg.generateBracketPower will return a vector of string brackets of size 2^7 (128).
                
                # initial loop to contain the number of fixed 8's to use
                # could be size / 128
                for index in range(int(size/128)):
                        newBracket = bg.generateBracketPower(year,r,samplingFnName)
                        for bracketIndex,aBracket in enumerate(newBracket):
                                bitString = "{:07b}".format(bracketIndex)
                                brackets.append(bm.stringToHex(bm.vectorToString(aBracket)+bitString))
                
                # loop thru the returned list of vectors,  convert each to string, apeend the 128 possible bit and append to brackets
        else:
                
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
        # brackets is a list of lists
        
        with open(filepath, 'w') as outputFile:
                outputFile.write(json.dumps(outputDict))


def generateFilepath(sampleSize, year=2020, model='power', r=1, samplingFnName=None, nReplications=1):
        """Generates the path to the JSON file containing the experiment batch with the given parameters."""
        homeDir = os.path.expanduser('~')
        filepath = '{0}/Documents/GitHub/power-model-ncaa/out'.format(homeDir)
        # my filepath
        filepath = '{0}/Documents/Research/Sheldon Jacobson/power-model-ncaa/Outputs'.format(homeDir)

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

# this will get me all the things in outputs
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

                # Power: r = 5, sampleF4A
                createAndSaveBracketPool(sampleSize, year=year, model='power', r=5, 
                        samplingFnName='sampleF4A', nReplications=nReplications)

                # Power: r = 5, sampleF4B
                createAndSaveBracketPool(sampleSize, year=year, model='power', r=5, 
                        samplingFnName='sampleF4B', nReplications=nReplications)

                 # Power: r = 6
                createAndSaveBracketPool(sampleSize, year=year, model='power', r=6, 
                        samplingFnName='sampleNCG', nReplications=nReplications)

               # Power: r=4
                createAndSaveBracketPool(sampleSize, year=year, model='power', r=4, 
                        samplingFnName='samplePower8Brute', nReplications=nReplications)



def readAndScore(nReplications, sampleSize):
        """Reads the JSON files for all experiment batches, 
        scores the brackets, and 
        stores the Max Score and ESPN Count results 
        in a CSV file."""
        # TODO: implement
        for year in range(2013, 2020):
                filepaths = []
                filepaths.append(generateFilepath(sampleSize, year=year, model='bradley-terry', 
                        nReplications=nReplications))
                filepaths.append(generateFilepath(sampleSize, year=year, model='power', 
                        nReplications=nReplications))
                filepaths.append(generateFilepath(sampleSize, year=year, model='power', r=4, 
                        samplingFnName='sampleE8', nReplications=nReplications))
                filepaths.append(generateFilepath(sampleSize, year=year, model='power', r=5, 
                        samplingFnName='sampleF4A', nReplications=nReplications))
                filepaths.append(generateFilepath(sampleSize, year=year, model='power', r=5, 
                        samplingFnName='sampleF4B', nReplications=nReplications))
                filepaths.append(generateFilepath(sampleSize, year=year, model='power', r=6, 
                        samplingFnName='sampleNCG', nReplications=nReplications))
                filepaths.append(generateFilepath(sampleSize,year = year, model = 'power', r = 4, samplingFnName='samplePower8Brute', nReplications = nReplications))

 
                # statistics to compute
                # in json file, will store a nReplications long array for each statistic

                totalFiles = len(filepaths)
                global NUM_GENERATORS
                NUM_GENERATORS = totalFiles
                
                maxScores = np.zeros((NUM_GENERATORS, nReplications))
                espnCounts = np.zeros((NUM_GENERATORS, nReplications))
                pfProps = np.zeros((NUM_GENERATORS, nReplications))
                variance = np.zeros((NUM_GENERATORS,nReplications))
                



                minEspnScore = sf.espnCutoffs[str(year)]
                maxPfScore = sf.pickFavoriteScore[str(year)]

               


                
               
                for fIndex, filepath in enumerate(filepaths):
                        with open(filepath, 'r') as f:
                                data = json.load(f)
                                print("opened file " + str(fIndex+1)+"/"+str(totalFiles))
 
                        brackets = data['brackets']
                        for repIndex, sample in enumerate(brackets):

                                # instantiates a vector that contains all scores in one sample. To compute statistics on scores, do it here.
                                scores = np.zeros(sampleSize)
                                for bracketIndex, bracketHex in enumerate(sample):
                                        bracketVector = bm.stringToVector(bm.hexToString(bracketHex))
                                        scores[bracketIndex] = sf.scoreBracket(bracketVector, year=year)[0]

                                # computing statistics 
                                maxScores[fIndex][repIndex] = np.max(scores)
                                espnCounts[fIndex][repIndex] = (scores >= minEspnScore).sum()
                                pfProps[fIndex][repIndex] = (scores >= maxPfScore).sum() * 1. / sampleSize
                                variance[fIndex][repIndex] = np.var(scores)

                        #append to the json files the statistics
                        statisticsOutput = {'maxScores': maxScores[fIndex][:].tolist(), 'espnCounts' : espnCounts[fIndex][:].tolist() , 'pfProps' : pfProps[fIndex][:].tolist(),'variance' : variance[fIndex][:].tolist() }

                        with open(filepath,'w') as outputFile:
                                data.update(statisticsOutput)
                                outputFile.write(json.dumps(data))
                                
                printResultsTables(year=year, maxScores=maxScores, espnCounts=espnCounts, pfProps=pfProps,variance = variance)


def printResultsTables(year=2020, maxScores=None, espnCounts=None, pfProps=None, variance = None):
        """Saves the Max Score, ESPN Count, and PF proportion results for a set of replications
        to a CSV file.
        Doesnt actually save to a csv file yet. 

        """
        modelHeaders = 'Replication,B-T,Power R64,Power E8,Power F4_A,Power F4_B,Power NCG'
        print(year)
        
        print('Max Scores')
        print(modelHeaders)
        for repIndex in range(maxScores.shape[1]):
                sys.stdout.write('{0},'.format(repIndex))
                for modelIndex in range(6):
                        sys.stdout.write('{0},'.format(int(maxScores[modelIndex][repIndex])))
                sys.stdout.write('\n')
        print()

        print('ESPN Counts')
        print(modelHeaders)
        for repIndex in range(espnCounts.shape[1]):
                sys.stdout.write('{0},'.format(repIndex))
                for modelIndex in range(6):
                        sys.stdout.write('{0},'.format(int(espnCounts[modelIndex][repIndex])))
                sys.stdout.write('\n')
        print()

        print('Pick Favorite Proportions')
        print(modelHeaders)
        for repIndex in range(pfProps.shape[1]):
                sys.stdout.write('{0},'.format(repIndex))
                for modelIndex in range(6):
                        sys.stdout.write('{0:.6f},'.format(pfProps[modelIndex][repIndex]))
                sys.stdout.write('\n')
        print()

        print('Variance')
        print(modelHeaders)
        for repIndex in range(pfProps.shape[1]):
                sys.stdout.write('{0},'.format(repIndex))
                for modelIndex in range(6):
                        sys.stdout.write('{0:.2f},'.format(variance[modelIndex][repIndex]))
                sys.stdout.write('\n')
        print()

      


def getScoreDistribution(nReplications=None, sampleSize=None, filepath=None):
        """Tallies and prints the distribution of scores for all brackets in the given filepath."""



if __name__ == '__main__':
        sampleSize = 50000
        nReplications = 25

        #runSamples(nReplications=nReplications, sampleSize=sampleSize)

        # solo power8Brute
        #for year in range(2013,2020):
         #       createAndSaveBracketPool(sampleSize,year = year,model = 'power',r=4,samplingFnName='samplePower8Brute',nReplications = nReplications)
        
        print("done samplying, starting scoring")
        readAndScore(nReplications=nReplications, sampleSize=sampleSize)


        quit()


        year = 2016
        filepath = generateFilepath(sampleSize, year=year, model='power', nReplications=nReplications)
        with open(filepath, 'r') as f:
                data = json.load(f)

        scoreTallies = np.zeros(193)

        brackets = data['brackets']
        for repIndex, sample in enumerate(brackets):
                scores = np.zeros(sampleSize)
                for bracketIndex, bracketHex in enumerate(sample):
                        bracketVector = bm.stringToVector(bm.hexToString(bracketHex))
                        scores[bracketIndex] = sf.scoreBracket(bracketVector, year=year)[0]
                        scoreTallies[int(scores[bracketIndex]) // 10] += 1

        for scoreDivTen in range(0, 193):
                print('{0},{1:.0f}'.format(scoreDivTen * 10, scoreTallies[scoreDivTen]))
