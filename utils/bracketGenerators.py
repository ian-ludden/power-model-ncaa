from collections import defaultdict
import os
import random

import bracketManipulations as bm
import fileUtils
import samplingFunctions
from scoringFunctions import scoreBracket

######################################################################
# Author: 	Ian Ludden
# Date: 	08 August 2019
# 
# bracketGenerators.py
# 
# Functions for generating NCAA March Madness tournament brackets. 
# 
######################################################################
# Seed sets for each half-region
TOP_SEEDS = [1, 16, 8, 9, 5, 12, 4, 13]
BOTTOM_SEEDS = [6, 11, 3, 14, 7, 10, 2, 15]

# Filenames for power and Bradley-Terry model parameters
HOME_DIR = os.path.expanduser('~')

POWER_FILENAME = '{0}/Documents/GitHub/power-model-ncaa/powerParams.csv'.format(HOME_DIR)
BT_FILENAME = '{0}/Documents/GitHub/power-model-ncaa/btParams.csv'.format(HOME_DIR)

# My paths
POWER_FILENAME = '{0}/Documents/Research/Sheldon Jacobson/power-model-ncaa/powerParams.csv'.format(HOME_DIR)
BT_FILENAME = '{0}/Documents/Research/Sheldon Jacobson/power-model-ncaa//btParams.csv'.format(HOME_DIR)



# Array for storing power model Round 1 alpha values
R1_ALPHAS = None

# List for storing power model weighted average alpha values for each round
ALPHA_VALS = None

# Array for storing Bradley-Terry model win probabilities. 
# Indexed by year, seed1, seed2
BT_PROBS = None




def generateBracketPower(year, r, samplingFnName=None):
    """Generates a bracket for the given year using the power model. 
       The given sampling function is used to sample the seeds which reach round r. The pseudo code in the paper. 
    
       With the inclusion of samplePower8Brute, this function is now a dependent function dependent on samplingFnName

       Parameters
       ----------
       year : int
           The tournament year for which to generate the bracket. 
           This affects which estimated win probabilities are used.
       r : int
           The round for which seeds will be sampled. 
       samplingFnName : string
           The name of the sampling function to be used for sampling 
           the seeds which reach round r. 
           If r is 1, then the sampling function is unnecessary. R
.

       Returns
       -------
       bracket : list of ints
           A list of 63 0s and/or 1s representing the predicted game outcomes.

       
    """
    global TOP_SEEDS
    bracket = []
    
    # you either let the power model choose for you all before final 4 or everything. The truncated geometric as of now only gives for those in rounds 4 +
    
    if (r > 1 and r < 4) or (r > 6):
        exit('Round {0} is not supported by generateBracketPower.'.format(r))

    # Sample fixed seeds, if necessary
    nSamples = 2 ** (7 - r) if r > 1 else 0
    # note that if I pick r = 6, that means i want to sample the 2 seeds that will appear in the NCG

    
    # need to call the sampling functions and generate the seeds from those samples' paths in the bracket

    if nSamples > 0:
        
        # when using samplePower8Brute I am actually using sampleE8
        if samplingFnName in ["samplePower8Brute","samplePower8BrutePf","samplePower8BrutePfNot"]:
            samplingFn = getattr(samplingFunctions, "sampleE8")
            sampledSeeds = samplingFn(year) 
        else:
            samplingFn = getattr(samplingFunctions, samplingFnName)
            sampledSeeds = samplingFn(year)  
            # sampledSeeds will be vector of sampled seed values. (not 0 or 1s)
            
        
    fixedChampions = [-1 for i in range(4)] # final 4. champs from each region

    # each region has two parts, top and bottom
    fixedTopE8s = [-1 for i in range(4)] #  for each region top part winner
    fixedBottomE8s = [-1 for i in range(4)] # for each region bot part winner
    
    # Handle each case of r
    if r == 6: # NCG
        # Fix F4 seeds as needed
        ncgTeam0Region = 0 if random.random() < 0.5 else 1
        ncgTeam1Region = 2 if random.random() < 0.5 else 3
        # the two sides. Pick one from each side
        
        fixedChampions[ncgTeam0Region] = sampledSeeds[0]
        fixedChampions[ncgTeam1Region] = sampledSeeds[1]

        # Fix E8 seeds as needed
        # need to propagate backwards the fixed seeds in NCG
        if sampledSeeds[0] in TOP_SEEDS:
            fixedTopE8s[ncgTeam0Region] = sampledSeeds[0]
        else:
            fixedBottomE8s[ncgTeam0Region] = sampledSeeds[0]

        if sampledSeeds[1] in TOP_SEEDS:
            fixedTopE8s[ncgTeam1Region] = sampledSeeds[1]
        else:
            fixedBottomE8s[ncgTeam1Region] = sampledSeeds[1]

    elif r == 5: # F4
        # Fix E8 seeds as needed
        for regionIndex in range(4):
            if sampledSeeds[regionIndex] in TOP_SEEDS:
                fixedTopE8s[regionIndex] = sampledSeeds[regionIndex]
            else:
                fixedBottomE8s[regionIndex] = sampledSeeds[regionIndex]

    elif r == 4: # E8
        # Fix E8 seeds; E8 samples come in top/bottom pairs
        for regionIndex in range(4):
            fixedTopE8s[regionIndex] = sampledSeeds[2 * regionIndex]
            fixedBottomE8s[regionIndex] = sampledSeeds[2 * regionIndex + 1]

    else: # By check at beginning, r must be 1
        pass # Leave all "fixed" seeds as -1 

    
    f4Seeds = [-1 for i in range(4)]
    # Select outcomes of each of the four regions
    # the fixed results prior to the elite 8, so for each region fix the free 8
    # pf and pfnot
    # 

            
    if samplingFnName in ["samplePower8Brute","samplePower8BrutePf","samplePower8BrutePfNot"]:
        if samplingFnName == "samplePower8BrutePf":
            modelName = "pf"
        if samplingFnName == "samplingPower8BrutePfNot":
            modelName = "pfNot"
        
        for i in range(128):
            aBracket = []
            for regionIndex in range(4):
                regionVector, f4Seeds[regionIndex] = sampleRegion(
                    fixedChampion=fixedChampions[regionIndex], 
                    fixedTopE8=fixedTopE8s[regionIndex], 
                    fixedBottomE8=fixedBottomE8s[regionIndex], 
                    year=year, 
                    model=modelName)
                aBracket += regionVector
            bracket.append(aBracket)
        # at this point we are still missing the last 7 bits strings, so note that this may break the score function   
 
else:
        for regionIndex in range(4):
            regionVector, f4Seeds[regionIndex] = sampleRegion(
                fixedChampion=fixedChampions[regionIndex], 
                fixedTopE8=fixedTopE8s[regionIndex], 
                fixedBottomE8=fixedBottomE8s[regionIndex], 
                year=year, 
                model='power')
            bracket += regionVector


       
        
        return bracket
            
    
   
    # [region1,region2,region3,region4]
        
    # Select outcomes of F4/NCG games (Rounds 5, 6)
    winProb0 = getWinProbability({'seed': f4Seeds[0]}, {'seed': f4Seeds[1]}, r=5, year=year, model='power')
    winProb1 = getWinProbability({'seed': f4Seeds[2]}, {'seed': f4Seeds[3]}, r=5, year=year, model='power')
   
    f4Result0 = 1 if random.random() < winProb0 else 0
    f4Result1 = 1 if random.random() < winProb1 else 0

    
    # NCG case, where F4 results are fixed
    if r == 6: 
        f4Winners = [1 if x==y else 0 for (x,y) in zip(f4Seeds,fixedChampions)]
        f4Result0 = 1 if f4Winners[0] == 1 else 0
        f4Result1 = 1 if f4Winners[2] == 1 else 0
    

    bracket.append(f4Result0)
    bracket.append(f4Result1)
    ncgSeeds = bm.applyRoundResults(f4Seeds, [f4Result0, f4Result1])

    # NCG
    # when feeding to getWinProbability, only seed value is used
    ncgTeam0 = {'seed': ncgSeeds[0], 'region': -1}
    ncgTeam1 = {'seed': ncgSeeds[1], 'region': -1}
    winProb = getWinProbability(ncgTeam0, ncgTeam1, r=6, year=year, model='power')
    ncgResult = 1 if random.random() < winProb else 0
    bracket.append(ncgResult)

    return bracket


def sampleRegion(fixedChampion=-1, fixedTopE8=-1, fixedBottomE8=-1, year=2020, model='power'):
    """Samples a region vector using the given model. 

       Parameters
       ----------
       fixedChampion : int (optional)
           The seed of the regional champion, if fixed in advance. 
       fixedTopE8 : int (optional)
           The seed of the Elite Eight team from the top half, if fixed in advance. 
       fixedBottomE8 : int (optional)
           The seed of the Elite Eight team from the bottom half, if fixed in advance. 
       year : int
           The year of the tournament to be predicted
       
       Returns
       -------
       regionVector : list of ints
           A list of 15 0s and/or 1s representing the game outcomes in the region, 8-4-2-1
      

       regionWinner : int
           The seed of the regional champion
    """
    global TOP_SEEDS, BOTTOM_SEEDS
    regionVector = []
    seeds = TOP_SEEDS + BOTTOM_SEEDS

    # Loop through Rounds 1 (R64), 2 (R32), 3 (S16), and 4 (E8)
    # 1 is the better seed, 0 the latter
    # top to bottom matchup wise first, then left to right round wise 

    
    for roundNum in range(1, 5):
        numGames = int(len(seeds) / 2)
        newSeeds = []
        for gameNum in range(numGames):
            seed1 = seeds[2 * gameNum]
            seed2 = seeds[2 * gameNum + 1]

            isSeed1WinAutomatically = (seed1 == fixedChampion) or (roundNum < 4 and (seed1 == fixedTopE8 or seed1 == fixedBottomE8))
            isSeed2WinAutomatically = (seed2 == fixedChampion) or (roundNum < 4 and (seed2 == fixedTopE8 or seed2 == fixedBottomE8))
            if isSeed1WinAutomatically:
                p = 1.
            elif isSeed2WinAutomatically:
                # should it be -1? 
                p = 0.
            else:
                # pf and pf not implemented here
                p = getWinProbability({'seed': seed1}, {'seed': seed2}, r=roundNum, year=year, model=model)

            rnd = random.random()
            regionVector.append(1 if rnd < p else 0)
            newSeeds.append(seed1 if rnd < p else seed2)
        seeds = newSeeds

    return [regionVector, seeds[0]]






def getWinProbability(team1, team2, r, year, model):
    """Returns the predicted probability that team1 defeats team2 
       in the given round r. 
       Can be modified to use different models. 

       Parameters
       ----------
       team1 : dict
           A dict with seed and region of the "upper" team 
           in the standard bracket representation
       team2 : dict
           A dict with seed and region of the "lower" team
       r     : int
           The round index (1 to 6)
       year : int
           The tournament year to be predicted
       model : string
           The name of the model to use : power, bradleyTerry, pf, pfNot
    
           

       Returns
       -------
       winProbability : float 
           The estimated win probability (between 0 and 1). 
    """
    global R1_ALPHAS, ALPHA_VALS, BT_PROBS
    seed1 = team1['seed']
    seed2 = team2['seed']

    if model == 'power':
        # Load parameters if not yet loaded
        if R1_ALPHAS is None or ALPHA_VALS is None:
            R1_ALPHAS, ALPHA_VALS = fileUtils.loadPowerParams(POWER_FILENAME)

        # Use ALPHA_VALS for other rounds (unless seeds are same)
        if seed1 == seed2:
            return 0.5

        # Use R1_ALPHAS for Round 1
        if r == 1:
            if not (seed1 + seed2 == 17):
                exit('Invalid round 1 matchup: seeds {0} vs. {1}.'.format(seed1, seed2))
            alpha = R1_ALPHAS[year - 2013][seed1] if seed1 < seed2 else R1_ALPHAS[year - 2013][seed2]
        else:
            # For Rounds 2-6, use weighted average alpha-value
            alpha = ALPHA_VALS[year - 2013][r]

        seed1a = (seed1 * 1.0) ** alpha
        seed2a = (seed2 * 1.0) ** alpha
        return seed2a / (seed1a + seed2a)

    elif model == 'bradley-terry':
        # Load parameters if not yet loaded
        if BT_PROBS is None:
            BT_PROBS = fileUtils.loadBradleyTerryParams(BT_FILENAME)

        return BT_PROBS[year - 2013][seed1][seed2]

    elif model == 'pf':
        if seed1 < seed2 :
            return 1
        elif seed1 > seed2:
            return 0
        else:
            exit('Invalid model \'{0}\' provided to getWinProbability.'.format(model))
            

    elif model == 'pfNot':
        if seed1 < seed2:
            return 0
        elif seed1 > seed2:
            return 1
        else:
            exit('Invalid model \'{0}\' provided to getWinProbability.'.format(model))

            
    else:
        exit('Invalid model \'{0}\' provided to getWinProbability.'.format(model))


def generateBracketBradleyTerry(year):
    """Generates a bracket for the given year using the Bradley-Terry model. 
       The given sampling function is used to sample the seeds which reach round r.

       Parameters
       ----------
       year : int
           The tournament year for which to generate the bracket. 
           This affects which estimated win probabilities are used.

       Returns
       -------
       bracket : list of ints
           A list of 63 0s and/or 1s representing the predicted game outcomes
    """
    bracket = []
    f4Seeds = [-1 for i in range(4)]
    
    # Sample each of the four regions independently
    for r in range(4):
        regionVector, f4Seeds[r] = sampleRegion(year=year, model='bradley-terry')
        bracket += regionVector

    # Choose outcomes of F4 and NCG games
    winProb0 = getWinProbability({'seed': f4Seeds[0]}, {'seed': f4Seeds[1]}, r=5, year=year, model='bradley-terry')
    f4Result0 = 1 if random.random() < winProb0 else 0
    winProb1 = getWinProbability({'seed': f4Seeds[2]}, {'seed': f4Seeds[3]}, r=5, year=year, model='bradley-terry')
    f4Result1 = 1 if random.random() < winProb1 else 0
    bracket.append(f4Result0)
    bracket.append(f4Result1)

    ncgSeeds = bm.applyRoundResults(f4Seeds, [f4Result0, f4Result1])
    winProb = getWinProbability({'seed': ncgSeeds[0]}, {'seed': ncgSeeds[1]}, r=5, year=year, model='bradley-terry')
    ncgResult = 1 if random.random() < winProb else 0
    bracket.append(ncgResult)

    return bracket


def generateBracketsPickFavorite():
    """Returns the pool of eight pick favorite brackets."""
    brackets = []
    regionHex = '7fc5'
    bracketString = bm.hexToString(regionHex) * 4
    for i in range(8):
        fullString = bracketString + '{0:03b}'.format(i)
        brackets.append(bm.stringToVector(fullString))

    return brackets


if __name__ == '__main__':
    # testPowerBracket = generateBracketPower(2019, 1)
    # print(scoreBracket(testPowerBracket, year=2019))

    # # import pdb; pdb.set_trace()

    #testNcgBracket = generateBracketPower(2019, 6, 'sampleNCG')
    # # print(scoreBracket(testNcgBracket, year=2019))

    # # import pdb; pdb.set_trace()

    # # testF4ABracket = generateBracketPower(2019, 5, 'sampleF4A')
    # # print(scoreBracket(testF4ABracket, year=2019))

    # # import pdb; pdb.set_trace()

     #testF4BBracket = generateBracketPower(2019, 5, 'sampleF4B')
    #print(scoreBracket(testF4BBracket, year=2019))

    # import pdb; pdb.set_trace()

    testPower8BruteBracket = generateBracketPower(2019, 4, 'samplePower8Brute')
    print(testPower8BruteBracket
    )
   # print(scoreBracket(testF4BBracket, year=2019))

    import pdb; pdb.set_trace()

  # SAMPLE_SIZE = 10000
    # year = 2016
    # for i in range(SAMPLE_SIZE):
    #     testE8Bracket = generateBracketPower(year, 4, 'sampleE8')
    #     scores = scoreBracket(testE8Bracket, year=year)
    #     # if 2 in seedsPerRound[4] or scores[0] > 950: 
    #     if scores[0] > 1600:
    #         print(scores, bm.bracketToSeeds(testE8Bracket))

    # import pdb; pdb.set_trace()

    # testBradleyTerryBracket = generateBracketBradleyTerry(2019)
    # print(scoreBracket(testBradleyTerryBracket, year=2019))

    pickFavoriteBrackets = generateBracketsPickFavorite()
    print('Year,Max PF Score')
    for year in range(2013, 2020):
        pfScores = [scoreBracket(pickFavoriteBrackets[i], year=year)[0] for i in range(8)]
        print('{0},{1}'.format(year, max(pfScores)))
