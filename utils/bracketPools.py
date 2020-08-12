import csv
import json
from math import ceil, floor, log
import numpy as np
import os
from pprint import pprint
import random
import sys
import itertools 



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

MODEL_TYPES = ['bradley-terry','power']
GENERATION_TYPES = [(0,1,None),(1,1,None),(1,4,'sampleE8'),(1,5,'sampleF4A'),(1,5,'sampleF4B'),(1,6,'sampleNCG'),(1,4,'samplePower8Brute'),(1,4,'samplePower8BrutePf'),(1,4,'samplePower8BruteRandom1'),(1,4,'samplePower8BruteRPickedUni'),(1,4,'samplePower8BruteRPickedTopN'),(1,4,'samplePower8BruteRPickedHistDist'),(1,5,'samplePower4ABrute'),(1,5,'samplePower4ABrutePf'),(1,5,'samplePower4ABrutePfNot'),(1,5,'samplePower4BBrute'),(1,5,'samplePower4BBrutePf'),(1,5,'samplePower4BBrutePfNot'),(1,5,'samplePower4BBruteRandom1'),(1,5,'samplePower4ARandom1'),(1,4,'samplePower8BrutePfNot'),(1,4,'samplePower8HPPMST1')]

#GENERATION_TYPES = [GENERATION_TYPES[-1]]

######################################################################
# Generator Naming 
#
# samplePower8Brute- : The picking of all 128 possible results post 
# fixing the elite 8 teams.
#
# -RPicked- : Predetermining the number of upsets to happen before 
# picking the non elite 8 influenced games to be a potential upset. 
#

# HistDist is historical based on seed matchups. 
######################################################################



# usually we run all, but since takes a while, here we select only the new ones to run
#GENERATION_TYPES = GENERATION_TYPES[]



# total number of methods of generations methods
NUM_GENERATORS = len(GENERATION_TYPES)
# is it worthwile for a global var if there will be more models in future
# NUM_GENERATORS get up to date number in read and score



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



        if 'MST' in samplingFnName:
                # mst
                roundIdp = None
                pickMethod = None
                modelName = 'power'
                
                if 'samplePower8HPPMST1' == samplingFnName:
                        pf128 = list() # all 128 pick favorite elite 8
                        brute128 = list() # all 128 bit string changes
                        for i in range(128):
                                pf = [1] * 60
                                bitString = "{:07b}".format(i)
                                bitString = [i for i in bitString]
                                brute128.append(bitString)
                                pf[14] = bitString[0]
                                pf[29] = bitString[1]
                                pf[44] = bitString[2]
                                pf[59] = bitString[3]
                                pf.extend(bitString[4:])
                                pf128.append(bm.stringToHex(bm.vectorToString(pf)))

                        adjacencyList = list()
                        adjacencyList.append(list()) # index 0 is pf, stores the edges in the mst, an edge is a tuple(z,x,y), z weight, x and y vertices

                        alTable = list() # vertices 0 to n, stores the bracket vectors, 60 bit version
                        pfStart = [1]*60
                        alTable.append(pfStart)

                        # only need to worry about the first 60 bits
                        brackets.extend(pf128) # initial points

                        index = 128
                        
                        while index < size:
                                nextBrackets = list()
                                beta = 7
                                for possibility in range(7):
                                        while(True):
                                                aBracket = bg.generateBracketPower(year,r,samplingFnName)
                                                if (aBracket not in alTable) and (sf.HPP(pfStart,aBracket,3) < 1250):
                                                        break # check if already a duplicate
                                                
                                        nextBrackets.append(aBracket)
                                # now we have the beta potential brackets, 60 bits
                                nextStepMST = list()
                                for newBracket in nextBrackets:
                                        nextStepMST.append(sf.mstNewBracket(adjacencyList, alTable,newBracket,3))

                                # gets index of maximum mst cost        
                                toAddIdx = nextStepMST.index(max(nextStepMST,key=lambda x:x[1]))

                                adjacencyList = nextStepMST[toAddIdx][0]
                                newBracket = nextBrackets[toAddIdx]
                                alTable.append(newBracket)

                                new_128 = list()
                                newBracket.extend([0]*3)
                                for bitString in brute128:
                                        
                                        newBracket[14] = bitString[0]
                                        newBracket[29] = bitString[1]
                                        newBracket[44] = bitString[2]
                                        newBracket[59] = bitString[3]
                                        newBracket[-3:] = bitString[-3:]
                                        new_128.append(bm.stringToHex(bm.vectorToString(newBracket)))
                        
                                brackets.extend(new_128)
                                print("128 done" + "  " + str(index))
                                index+=128
                                
                                        
                return brackets
        
        
        #if samplingFnName in ["samplePower8Brute","samplePower8BrutePf","samplePower8BrutePfNot"]
        
        elif 'samplePower8Brute' in samplingFnName:
                # special case, one call to bg.generateBracketPower will return a vector of string brackets of size 2^7 (128).
                
                # initial loop to contain the number of fixed 8's to use
                # could be size / 128
                for index in range(int(size/128)):
                        newBracket = bg.generateBracketPower(year,r,samplingFnName)
                        for bracketIndex,aBracket in enumerate(newBracket):
                                bitString = "{:07b}".format(bracketIndex)
                                bitString = [i for i in bitString]

                                # sampleRegion determines region outcome, need to overwrite region outcomes to a random outcome. 
                                aBracket[14] = bitString[0]
                                aBracket[29] = bitString[1]
                                aBracket[44] = bitString[2]
                                aBracket[59] = bitString[3]
                                aBracket.extend(bitString[4:])
                                brackets.append(bm.stringToHex(bm.vectorToString(aBracket)))
                               
        elif 'samplePower4' in samplingFnName:
                for index in range(int(size/8)):
                        newBracket = bg.generateBracketPower(year,r,samplingFnName)
                        for bracketIndex,aBracket in enumerate(newBracket):
                                bitString = "{:03b}".format(bracketIndex)
                                bitString = [i for i in bitString]
                                aBracket.extend(bitString)
                                brackets.append(bm.stringToHex(bm.vectorToString(aBracket)))

                        

                
                
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
                print("rep : " +str(replicationIndex))
                brackets.append(generateBracketPool(sampleSize, year, model, r, samplingFnName))

        if filepath is None:
                filepath = generateFilepath(sampleSize, year=year, model=model, r=r, samplingFnName=samplingFnName, nReplications=nReplications)

        outputDict = {'year': year, 'sampleSize': sampleSize, 
                                        'nReplications': nReplications, 'model': model, 
                                        'r': r, 'samplingFnName': samplingFnName, 
                                        'brackets': brackets}
        # brackets is a list of lists
        print("saving 1")
        with open(filepath, 'w') as outputFile:
                outputFile.write(json.dumps(outputDict))


def generateFilepath(sampleSize, year=2020, model='power', r=1, samplingFnName=None, nReplications=1, folder = "generatorOutputs"):
        """Generates the path to the JSON file containing the experiment batch with the given parameters."""

        # folder generatorOutputs : stores the generator outputs
        # folder compressedOutputs : stores summarized versions
        
        homeDir = os.path.expanduser('~')
        filepath = '{0}/Documents/GitHub/power-model-ncaa/out'.format(homeDir)
        # my filepath
        filepath = ('{0}/Documents/Research/Sheldon Jacobson/power-model-ncaa/Outputs/'+folder).format(homeDir)

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

                for one in GENERATION_TYPES:
                        createAndSaveBracketPool(sampleSize, year=year, model=MODEL_TYPES[one[0]], r=one[1], 
                        samplingFnName=one[2], nReplications=nReplications)

  
# scores a bracket from json file 
def readAndScore(nReplications, sampleSize):
        """Reads the JSON files for all experiment batches, 
        scores the brackets, and 
        stores the Max Score and ESPN Count results 
        in a CSV file. (updates the json file directly now)  """
        # TODO: implement
        for year in range(2013, 2020):
                filepaths = []
                for one in GENERATION_TYPES:
                        filepaths.append(generateFilepath(sampleSize,year = year, model = MODEL_TYPES[one[0]], r = one[1], samplingFnName=one[2], nReplications = nReplications))
                        
 
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
                                        if(bracketIndex) >= sampleSize:
                                                break
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
                for modelIndex in range(NUM_GENERATORS):
                        sys.stdout.write('{0},'.format(int(maxScores[modelIndex][repIndex])))
                sys.stdout.write('\n')
        print()

        print('ESPN Counts')
        print(modelHeaders)
        for repIndex in range(espnCounts.shape[1]):
                sys.stdout.write('{0},'.format(repIndex))
                for modelIndex in range(NUM_GENERATORS):
                        sys.stdout.write('{0},'.format(int(espnCounts[modelIndex][repIndex])))
                sys.stdout.write('\n')
        print()

        print('Pick Favorite Proportions')
        print(modelHeaders)
        for repIndex in range(pfProps.shape[1]):
                sys.stdout.write('{0},'.format(repIndex))
                for modelIndex in range(NUM_GENERATORS):
                        sys.stdout.write('{0:.6f},'.format(pfProps[modelIndex][repIndex]))
                sys.stdout.write('\n')
        print()

        print('Variance')
        print(modelHeaders)
        for repIndex in range(pfProps.shape[1]):
                sys.stdout.write('{0},'.format(repIndex))
                for modelIndex in range(NUM_GENERATORS):
                        sys.stdout.write('{0:.2f},'.format(variance[modelIndex][repIndex]))
                sys.stdout.write('\n')
        print()

      


def getScoreDistribution(nReplications=None, sampleSize=None, filepath=None):
        """Tallies and prints the distribution of scores for all brackets in the given filepath."""



if __name__ == '__main__':
        sampleSize = 50000
        nReplications = 25

        #nReplications = 9 # for mst takes to long

        #runSamples(nReplications=nReplications, sampleSize=sampleSize)

        # solo power8BrutePf and Pfnot
        # for year in range(2013,2020):
        #         createAndSaveBracketPool(sampleSize,year = year,model = 'power',r=4,samplingFnName='samplePower8BrutePf',nReplications = nReplications)
        #         createAndSaveBracketPool(sampleSize,year = year,model = 'power',r=4,samplingFnName='samplePower8BrutePfNot',nReplications = nReplications)
        
        #print("done samplying, starting scoring")
        #readAndScore(nReplications=nReplications, sampleSize=sampleSize)

        #quit()

        # getting the score pairwise difference distributions 


        # files = os.listdir("../Outputs")
        # entries = []

        # for file in files:
        #         name = file.split("_")
        #         method = name[0]
        
        #         sample = name[-1].split(".")[0]
        #         year = name[4] if method == "power" else sample
        #         if method == "power":
        #                 entries.append(["../Outputs/"+file,method+sample,int(year)])
        #         else:
        #                 entries.append(["../Outputs/"+file,"bradley-terry",int(year)])
                
        # distributions = dict()

        # count = 0
        # for file_path,generator,year in entries:
        #         count+=1
        #         print(count)
        #         if generator not in distributions:
        #                 distributions[generator] = dict()

        #         with open(file_path) as f:
        #                 data = json.load(f)
                
        #         size  = len(data["brackets"][0])
        #         # 25 different repeitions. From the 25 * 50k, sample 1000.
        #         sample = (np.random.choice([i for i in range(25 * size)],replace = False, size = 100000))
        #         sample1 = ([(np.floor(i/size),i%size) for i in sample])
        #         sample2 = [ bm.stringToVector(bm.hexToString(data["brackets"][int(i[0])][i[1]])) for i in sample1]
        #         # the 63 bit vectors
        #         differences = list()
        #         #lol = itertools.combinations(sample2,2)
        #         #for i in lol:
        #             #differences.append(1920 - sf.scoreBracket(i[0],i[1])[0])
        #         #distributions[generator][year]  = (np.histogram(differences, [i*10 for i in range(194)])[0]).tolist()
        #         for sample in sample2:
        #                 differences.append(1920 - sf.HPP([1]*60,sample,4))
        #         distributions[generator][year] = (np.histogram(differences,[i*10 for i in range(194)])[0]).tolist()
        #         print(generator,year)



        # with open('pfRel_HPP_2013_2019_models.json','w') as json_file:
        #         json.dump(distributions,json_file)

        # quit()

        
        pastWin = dict()
        with open("../allBracketsTTT.json","r") as file:
                historical = json.load(file)
        for i in historical["brackets"]:
                pastWin[i["bracket"]["year"]]=(bm.stringToVector(i["bracket"]['fullvector']))


        upUntilNow = dict()
        # using all availble previous years
        for year in range(1986,2020):
                allPrev = list()
                for prev in range(1985,year):
                        allPrev.append(pastWin[str(prev)])

                scores = list()
                for prev in allPrev:
                        scores.append(sf.HPP(pastWin[str(year)][:60],prev[:60],4))
                upUntilNow[year] = scores



                
        # for computing the historical pf results relative to actual results
        
        # pfBracket = [1] * 60
        # for year in pastWin:
        #         that_year = pastWin[year]
                #year_scores = list()
                #for i in range(0,8):
                #       bitString = "{:03b}".format(i)
                #      bitString = [i for i in bitString]
                        #year_scores.append(sum(sf.scoreBracket(pfBracket+bitString,actualResultsVector = that_year)))
                #print(1920 - max(year_scores))
                #pastWin[year] = 1920 - max(year_scores)
        #with open('HPP_pf_historical.json', 'w') as json_file:
        #       json.dump(pastWin, json_file)

        
        
        


        
        # for year in pastWin:
        #         that_year = pastWin[year]
        #         rotations = dict()
                
        #         region1 = that_year[:15]
        #         region2 = that_year[15:30]
        #         region3 = that_year[30:45]
        #         region4 = that_year[45:60]
        #         regions = [region1,region2,region3,region4]
        #         for i in itertools.permutations("1234"):
        #                 reordered = regions[int(i[0])-1] + regions[int(i[1])-1] + regions[int(i[2])-1] + regions[int(i[3])-1]
        #                 rotations[(i[0]+i[1]+i[2]+i[3])] = (sf.scoreBracket(reordered+that_year[-3:],actualResultsVector = that_year)[0])

        #         pastWin[year] = rotations

         
        with open('previousYearsHPPWithCurrent.json','w') as json_file:
                json.dump(upUntilNow,json_file)
        

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
