import csv
import json
from math import ceil, floor, log
import numpy as np
import os
from pprint import pprint
import random
import sys
import pandas as pd

import bracketGenerators as bg
import bracketManipulations as bm
import scoringFunctions as sf

######################################################################
# Author: 	Ian Ludden
# Date: 	16 August 2019
# 
# MCB.py
# 
# This script runs Hsu's multiple comparisons with the best (MCB)
# method on samples generated by bracketPools.py. 
# 
######################################################################


# for jupyter notebook created csv  (Results)
# must have existing file at path

def loadResults2(nReplications = 25, sampleSize=50000,filepath=None):

        data = pd.read_csv(filepath)
        years = (data['year'].unique())
        models = (data['model'].unique())
        year_count = len(years)
        model_count = len(models)
        print(model_count)
        # because the way I named the bradleyTerry models included the unique year at end of name, hence there are repeats

        maxScores = np.zeros((model_count, year_count, nReplications)) # Indexed by model, year-2013, and replication
        espnCounts = np.zeros((model_count, year_count, nReplications))
        pfProps = np.zeros((model_count, year_count, nReplications))
        stat_names = ["maxScore","espnCounts","pfProps"]
        
        
        for idx_m,model in enumerate(models):
                for idx_y,year in enumerate(years):
                        model_year = data[data.model.isin([model]) & data.year.isin([year])]
                        for i in range(0,25):
                                for idx_s,stat in enumerate( [maxScores,espnCounts,pfProps]):
                                        stat[idx_m][idx_y][i] = model_year[model_year.rep.isin([i])][stat_names[idx_s]]
                                       
        

        return models,model_count,[maxScores,espnCounts,pfProps]

def loadResults(nReplications=25, sampleSize=50000, filepath=None):
        """Loads the CSV file containing the max scores, ESPN counts, and pick favorite proportions."""
        YEAR_OFFSET = 10 + 3 * nReplications # Number of rows per year in the CSV file.

        maxScores = np.zeros((6, 7, nReplications)) # Indexed by model, year-2013, and replication
        espnCounts = np.zeros((6, 7, nReplications))
        pfProps = np.zeros((6, 7, nReplications))

        if filepath is None:
                return [maxScores, espnCounts]

        with open(filepath, 'r') as f:
                reader = csv.reader(f)
                data = list(reader)



        for year in range(2013, 2020):
                startMaxScores = YEAR_OFFSET * (year - 2013) + 3
                endMaxScores = startMaxScores + nReplications
                for rowIndex in range(startMaxScores, endMaxScores):
                        for colIndex in range(1, 7):
                                maxScores[colIndex - 1][year - 2013][rowIndex - startMaxScores] = int(data[rowIndex][colIndex])

                startEspnCounts = endMaxScores + 3
                endEspnCounts = startEspnCounts + nReplications
                for rowIndex in range(startEspnCounts, endEspnCounts):
                        for colIndex in range(1, 7):
                                espnCounts[colIndex - 1][year - 2013][rowIndex - startEspnCounts] = int(data[rowIndex][colIndex])

                startPfProps = endEspnCounts + 3
                endPfProps = startPfProps + nReplications
                for rowIndex in range(startPfProps, endPfProps):
                        for colIndex in range(1, 7):
                                pfProps[colIndex - 1][year - 2013][rowIndex - startPfProps] = float(data[rowIndex][colIndex])

        return [maxScores, espnCounts, pfProps]


def unbiased_std(matrix):
        """Computes the unbiased pooled estimate of the variance (sigma^2) 
        based on nu = t(n - 1) degrees of freedom. See Section 4.4 of Becchofer et al.'s
        book, 'Design and Analysis of Experiments.'
        """
        t, n = matrix.shape
        nu = t * (n - 1) # degrees of freedom
        y_bar = np.mean(matrix, axis=1)[:, np.newaxis]
        return np.sqrt(((matrix - y_bar) ** 2.).sum() / nu)


def mcb(matrix):
        """Runs Hsu's mutliple comparisons with the best (MCB) procedure 
        to compare the performance of the different models for each year/metric.
        """ 
        # Table from Bechhofer et al. Design and Analysis of Experiments textbook
        # 120 d.f. (close to 144 d.f.), p = t-1 = 5, alpha = 0.05
        h = 2.26 
        samples = matrix
        t, n = samples.shape
        s_v = unbiased_std(samples)
        y_bar = np.mean(samples, axis=1)
        lowerBounds = []
        upperBounds = []
        for i in range(t):
                mu_i = y_bar[i]
                not_indices = np.setxor1d(np.indices(y_bar.shape), [i])
                others = y_bar[not_indices]
                intervalCenter = mu_i - np.max(others)
                intervalMin = intervalCenter - h * s_v * np.sqrt(2. / n)
                intervalMax = intervalCenter + h * s_v * np.sqrt(2. / n)
                lowerBounds.append(min(intervalMin, 0))
                upperBounds.append(max(intervalMax, 0))
        return [lowerBounds, upperBounds]


def runMCB(nReplications=25, sampleSize=50000, filepath=None):
        """Runs Hsu's mutliple comparisons with the best (MCB) procedure 
        to compare the performance of the different models for each year/metric.
        """
        #maxScores, espnCounts, pfProps = loadResults(nReplications=nReplications, sampleSize=sampleSize, filepath=filepath)
        modelsName,modelsCount,[maxScores, espnCounts, pfProps] = loadResults2(nReplications=nReplications, sampleSize=sampleSize, filepath='{0}/Documents/Research/Sheldon Jacobson/power-model-ncaa/{1}_x_{2}k_output_2_dimensional.csv'.format(HOME_DIR, nReplications, sampleSize // 1000))

       
        modelsName = np.asarray(modelsName)
        
        resultsDict = dict()
        # (year,model) : (lb(ms), ub, lb(espn count), ub)
        
        for year in range(2013, 2020):
                print(year)
                yearIndex = year - 2013

                maxScoresMatrix = maxScores[:, yearIndex, :].reshape((modelsCount, nReplications))
                maxScoreMeans = maxScoresMatrix.mean(axis=1)
                #sys.stdout.write('Max Scores,\nAverage,')
               # for modelIndex in range(modelsCount):
                 #       sys.stdout.write('{0:.2f},'.format(maxScoreMeans[modelIndex]))
                #sys.stdout.write('\n')

                lowerBounds, upperBounds = mcb(maxScoresMatrix)
              
                
                #sys.stdout.write('Interval LB,')
                #for modelIndex in range(modelsCount):
                 #       sys.stdout.write('{0:.2f},'.format(lowerBounds[modelIndex]))
                #sys.stdout.write('\nInterval UB,')
                #for modelIndex in range(modelsCount):
                 #       sys.stdout.write('{0:.2f},'.format(upperBounds[modelIndex]))
                #sys.stdout.write('\n\n')

                espnCountsMatrix = espnCounts[:, yearIndex, :].reshape((modelsCount, nReplications))
                espnCountMeans = espnCountsMatrix.mean(axis=1)
                #sys.stdout.write('ESPN Counts,\nAverage,')
                #for modelIndex in range(modelsCount):
                 #       sys.stdout.write('{0:.2f},'.format(espnCountMeans[modelIndex]))
                #sys.stdout.write('\n')

                lowerBounds2, upperBounds2 = mcb(espnCountsMatrix)
            
                
                #sys.stdout.write('Interval LB,')
                #for modelIndex in range(modelsCount):
                 #       sys.stdout.write('{0:.2f},'.format(lowerBounds[modelIndex]))
                #sys.stdout.write('\nInterval UB,')
                #for modelIndex in range(modelsCount):
                 #       sys.stdout.write('{0:.2f},'.format(upperBounds[modelIndex]))
                #sys.stdout.write('\n\n')

                pfPropsMatrix = pfProps[:, yearIndex, :].reshape((modelsCount, nReplications))
                pfPropMeans = pfPropsMatrix.mean(axis=1)
                #sys.stdout.write('PF Proportions,\nAverage,')
                #for modelIndex in range(modelsCount):
                 #       sys.stdout.write('{0:.5f},'.format(pfPropMeans[modelIndex]))
                #sys.stdout.write('\n')

                
                for i in range(modelsCount):
                        resultsDict[(year,modelsName[i])] = [lowerBounds[i],upperBounds[i],lowerBounds2[i],upperBounds[i]]
        
        final_results = list()
        for one in resultsDict:
                final_results.append([one[0],one[1]]+resultsDict[one])
        print(final_results)
        final_results = pd.DataFrame(final_results)
        final_results.columns = ['year','model','maxScoreL','maxScoreU','espnCountsL','espnCountsU']
        results = pd.read_csv("../25_x_50k_generation_statistics.csv")
        final_results = (pd.merge(results,final_results,on=['year','model']))
        final_results.to_csv("../25_x_50k_generation_statistics.csv",index =False)
        
                

if __name__ == '__main__':
        sampleSize = 50000
        nReplications = 25
        HOME_DIR = os.path.expanduser('~')
        #filepath = '{0}/Documents/GitHub/power-model-ncaa/{1}_x_{2}k_output.csv'.format(HOME_DIR, nReplications, sampleSize // 1000)
        filepath = '{0}/Documents/Research/Sheldon Jacobson/power-model-ncaa/{1}_x_{2}k_output.csv'.format(HOME_DIR, nReplications, sampleSize // 1000)

        runMCB(nReplications=nReplications, sampleSize=sampleSize, filepath=filepath)



        # to append the mcb CI's to final results, make sure there exist an unappended csv of the other results at path, this csv file is generated by result
