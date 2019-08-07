#!/usr/bin/env python
import time
import json
import numpy as np
import os.path
import random
import sys
from math import log, ceil, floor

from samplingUtils import getTruncGeom
from samplingUtils import getE8SeedBottom, getE8SeedTop
from samplingUtils import getF4SeedSplit, getF4SeedTogether
from samplingUtils import getChampion, getRunnerUp

from scoringUtils import applyRoundResults
from scoringUtils import getActualBracketVector
from scoringUtils import scoreBracket
from utils.runtimeSummary import RuntimeSummary

######################################################################
# Author:
#     Ian Ludden
#
# Date created:
#     15 Apr 2018
#
# Last modified:
#     16 Apr 2018
#
# This general version handles all parameters previously implemented
# separately in runExperimentsFixedAlpha.py,
# runExperimentsSampleF4.py, and runExperimentsSampleE8.py.
#
# Specifically, this version supports:
# - "Forward" Power Model
# - "Reverse" Power Model (generate champ and runner-up, then forward)
# - "Reverse" Power Model with F4 (also generate other two F4 seeds)
# - F4 Model 1, where F4 seeds are generated using "Model 1,"
#   then power model is applied to games before and after
# - F4 Model 2, where F4 seeds are generated using "Model 2,"
#   then power model is applied to games before and after
# - E8 Model, where E8 seeds are generated,
#   then power model is applied to games before and after
#
# Also, the Round 1 alpha values are optionally grouped as:
# (1, 16) alone
# (2, 15) alone
# (3, 14), (4, 13)
# (5, 12), (6, 11), (7, 10)
# (8, 9) alone and fixed at 0.5 probability (alpha = 0)
#
# By default, all weighted alpha values are computed using the
# standard weighting (multiply each alpha by [# matchups]).
#
# If "isSeedWeighted" is set to "True", then the seed-weighted
# average alpha values are used.
#
# This version no longer requires models to specify the alpha value
# parameters for each round. Round 1 is always matchup-specific
# (with optional grouping), and Rounds 2-6 always use a
# weighted average.
######################################################################

perturbed_ps = {
    'model10': [
        1.0,
        0.4229103348680731,
        0.8686253918935398,
        1.0,
        0.8123704992250859,
        1.0,
        0.7077991428824237,
        1.0,
    ],
    'model11': [
        1.0,
        0.426407803982116,
        0.8185896601561352,
        1.0,
        0.7023989582786588,
        1.0,
        0.6732989797077819,
        1.0,
    ],
    'model12': [
        1.0,
        0.40952672906312887,
        0.7854648378949232,
        1.0,
        0.7606474009097475,
        1.0,
        0.6539118110780404,
        1.0,
    ],
    'model13': [
        1.0,
        0.47544408691902595,
        0.7559601878920554,
        0.9487586660088168,
        0.7109945653089622,
        1.0,
        0.6877373278227099,
        1.0,
    ],
    'model14': [
        1.0,
        0.6897381544845143,
        0.6574444680210119,
        0.9879908588573346,
        0.6011024159891302,
        0.9211618267386112,
        0.8727621098790732,
        1.0,
    ],
    'model15': [
        0.9926470588235294,
        0.5,
        0.6544117647058824,
        0.7941176470588235,
        0.625,
        0.8455882352941176,
        0.5069725126093771,
        0.9411764705882353,
    ],
    'model20': [
        1.0,
        0.7294731173878907,
        0.6258223531539129,
        0.9164091224773702,
        0.5715239568648995,
        1.0,
        0.7069538395226441,
        1.0,
    ],
    'model21': [
        1.0,
        0.6810812994722260,
        0.6782287991306060,
        1.0,
        0.6065824706992460,
        0.9954042914277010,
        0.6102388677298300,
        1.0
    ],
    'model23': [
        1.0,
        0.7074624431810751,
        0.7470166821001302,
        1.0,
        0.5968548284479815,
        0.9691087987086129,
        0.8588856269144272,
        1.0
    ],
    'model30': [
        1.0,
        0.0,
        0.15639768832340517,
        1.0,
        1.0,
        0.823616960019198,
        0.5363817471415633,
        1.0,
    ],
    '25_1985': [
        1.0,
        0.4229103348680731,
        0.8686253918935398,
        1.0,
        0.8123704992250859,
        1.0,
        0.7077991428824237,
        1.0,
    ],
    '26_1985': [
        1.0,
        0.426407803982116,
        0.8185896601561352,
        1.0,
        0.7023989582786588,
        1.0,
        0.6732989797077819,
        1.0,
    ],
    '27_1985': [
        1.0,
        0.40952672906312887,
        0.7854648378949232,
        1.0,
        0.7606474009097475,
        1.0,
        0.6539118110780404,
        1.0,
    ],
    '28_1985': [
        1.0,
        0.47544408691902595,
        0.7559601878920554,
        0.9487586660088168,
        0.7109945653089622,
        1.0,
        0.6877373278227099,
        1.0,
    ],
    '30_1985': [
        1.0,
        0.6897381544845143,
        0.6574444680210119,
        0.9879908588573346,
        0.6011024159891302,
        0.9211618267386112,
        0.8727621098790732,
        1.0,
    ],
    '31_1985': [
        0.9926470588235294,
        0.5,
        0.6544117647058824,
        0.7941176470588235,
        0.625,
        0.8455882352941176,
        0.5069725126093771,
        0.9411764705882353,
    ],
    '29_2002': [
        1.0,
        0.7294731173878907,
        0.6258223531539129,
        0.9164091224773702,
        0.5715239568648995,
        1.0,
        0.7069538395226441,
        1.0,
    ],
    '28_2002': [
        1.0,
        0.6810812994722260,
        0.6782287991306060,
        1.0,
        0.6065824706992460,
        0.9954042914277010,
        0.6102388677298300,
        1.0
    ],
    '30_2002': [
        1.0,
        0.7074624431810751,
        0.7470166821001302,
        1.0,
        0.5968548284479815,
        0.9691087987086129,
        0.8588856269144272,
        1.0
    ]
}

newone = {}
for key in perturbed_ps.keys():
    sorted_p = np.array(perturbed_ps[key])[[0, 7, 5, 3, 2, 4, 6, 1]]
    newone[key] = [None] + sorted_p.tolist()
perturbed_ps = newone

# Returns the estimated probability that s1 beats s2
def getP(s1, s2, model, year, roundNum):
    if model.get('annealing_model') is not None and roundNum == 1:
        return perturbed_ps[model.get('annealing_model')][min(s1, s2)]
    alpha = getAlpha(s1, s2, model, year, roundNum)
    s1a = (s1 * 1.0) ** alpha
    s2a = (s2 * 1.0) ** alpha
    return s2a / (s1a + s2a)


TOP_SEEDS = [1, 16, 8, 9, 5, 12, 4, 13]
BOTTOM_SEEDS = [6, 11, 3, 14, 7, 10, 2, 15]


# This function generates a 63-element list of 0s and 1s
# to represent game outcomes in a bracket. The model specifies
# which alpha value(s) to use for each round.
def generateBracket(model, year):
    pooled = model.get('pooled', False)
    generator = model.get('generator', None)
    fmt = model.get('format', 'TTT')
    override_f4 = model.get('overrideF4', 'TTT')

    bracket = []

    # random.seed()

    endModel = 'None'
    if 'endModel' in model:
        endModel = model['endModel']

    NCG_E8 = 'NCG_E8'
    e8Seeds = []
    if endModel == 'E8' or endModel == NCG_E8:
        for i in range(4):
            e8Seeds.append(getE8SeedTop(year, model))
            e8Seeds.append(getE8SeedBottom(year, model))
    else:
        e8Seeds = [-1, -1, -1, -1, -1, -1, -1, -1]

    f4Seeds = []
    if endModel == 'F4_1':
        for i in range(4):
            seed = getF4SeedTogether(year, model)
            f4Seeds.append(seed)
            if endModel == NCG_E8:
                if seed in TOP_SEEDS:
                    e8Seeds[i * 2] = seed
                else:
                    e8Seeds[i * 2 + 1] = seed
    elif endModel == 'F4_2':
        for i in range(4):
            f4Seeds.append(getF4SeedSplit(year, model))
    else:
        f4Seeds = [-1, -1, -1, -1]

    ncgSeeds = [-1, -1]
    if 'Rev' in endModel or endModel == NCG_E8:
        champion = getChampion(year, model)
        runnerUp = getRunnerUp(year, model)
        champRegion = int(floor(random.random() * 4))
        champHalf = champRegion / 2
        ruRegion = int(floor(random.random() * 2))

        if champHalf == 0:
            ncgSeeds = [champion, runnerUp]
        else:
            ncgSeeds = [runnerUp, champion]

        ffrRegion = 1 - ruRegion

        if champRegion < 2:
            ruRegion += 2
            ffrRegion += 2
            ffcRegion = 1 - champRegion
        else:
            ffcRegion = 5 - champRegion

        f4Seeds[champRegion] = champion
        f4Seeds[ruRegion] = runnerUp

        if endModel == NCG_E8:
            if champion in TOP_SEEDS:
                e8Seeds[champRegion * 2] = champion
            else:
                e8Seeds[champRegion * 2 + 1] = champion
            if runnerUp in TOP_SEEDS:
                e8Seeds[ruRegion * 2] = runnerUp
            else:
                e8Seeds[ruRegion * 2 + 1] = runnerUp

    else:
        champRegion = -1
        ruRegion = -1

    if endModel == 'Rev_4':
        f4Seeds[ffcRegion] = getF4SeedTogether(year, model)
        f4Seeds[ffrRegion] = getF4SeedTogether(year, model)

    if not generator:
        # Loop through regional rounds R64, R32, and S16
        for region in range(4):
            seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
            for roundNum in range(1, 5):
                numGames = int(len(seeds) / 2)
                newSeeds = []
                for gameNum in range(numGames):
                    s1 = seeds[2 * gameNum]
                    s2 = seeds[2 * gameNum + 1]

                    # Force any fixed F4/E8 seeds to make it through
                    s1Wins = (s1 == f4Seeds[region]) or ((roundNum < 4) and ((s1 == e8Seeds[2*region]) or (s1 == e8Seeds[2*region + 1])))
                    s2Wins = (s2 == f4Seeds[region]) or ((roundNum < 4) and ((s2 == e8Seeds[2*region]) or (s2 == e8Seeds[2*region + 1])))

                    if s1Wins:
                        p = 1
                    elif s2Wins:
                        p = 0
                    else:
                        p = getP(s1, s2, model, year, roundNum)

                    if random.random() <= p:
                        bracket.append(1 if fmt == 'TTT' else (1 if s1 < s2 else 0))
                        newSeeds.append(s1)
                    else:
                        bracket.append(0 if fmt == 'TTT' else (1 if s2 < s1 else 0))
                        newSeeds.append(s2)
                seeds = newSeeds
            f4Seeds[region] = seeds[0]
        bracket = bracket + [-1, -1, -1]

    # Round 5:
    for gameNum in range(2):
        s1 = f4Seeds[2 * gameNum]
        s2 = f4Seeds[2 * gameNum + 1]

        if 'Rev' in endModel:
            if (2 * gameNum == champRegion) or (2 * gameNum == ruRegion):
                p = 1
            elif (2 * gameNum + 1 == champRegion) or (2 * gameNum + 1 == ruRegion):
                p = 0
            else:
                p = getP(s1, s2, model, year, 5)
        else:
            p = getP(s1, s2, model, year, 5)

        if random.random() <= p:
            bracket[60 + gameNum] = 1
            ncgSeeds[gameNum] = s1
        else:
            bracket[60 + gameNum] = 0
            ncgSeeds[gameNum] = s2

    # Round 6:
    s1 = ncgSeeds[0]
    s2 = ncgSeeds[1]

    if 'Rev' in endModel:
        if champHalf == 0:
            p = 1
        else:
            p = 0
    else:
        p = getP(s1, s2, model, year, 6)

    if random.random() <= p:
        bracket[-1] = 1
    else:
        bracket[-1] = 0

    # assert len(bracket) == 63
    # assert np.count_nonzero(np.array(bracket) == -1) == 0
    return bracket


otherAlphas = {
    'fixed10': [
        [2.0, 1.4108692733407024, 2.0, 1.3196474396785278, 1.3038570916197816,
         1.390660212727737, 1.2877841487352095],
        [1.0095261202192172, 1.042433554665708, 1.1252835165072113,
         1.0127669975888731, 2.0, 1.0915569133849663, 1.0108415509280697],
        [0.49348535996812848, 0.34455488639172854, 0.32896149660506729,
         0.24482951533228789, 0.25371038128353141, 0.29243699166273351,
         0.23378870332685775],
        [0.78668980967436719, 0.82717191347865293, 0.72955875777125934,
         0.71960083188622204, 0.7732117800073226, 0.76785259241699599,
         0.83093577802955143],
        [1.6307769609703682, 1.8506689881586398, 1.4998981924422568,
         1.7433925119535418, 1.3095428779096656, 1.3653554797198355,
         1.4778725904750136]
    ],
    'fixed20': [
        [2.0, 1.4474409352158495, 2.0, 2.0, 2.0, 2.0, 1.7032657440713581],
        [2.0, 2.0, 2.0, 2.0, 2.0, 1.258704146604319, 1.3271005026998224],
        [0.63660689580180285, 0.53208204238264734, 0.40456592054926921,
         0.38572619897984284, 0.31335311265081839, 0.46398743518041724,
         0.3146818626821023],
        [1.0042354049979449, 0.962953595740205, 0.78284871486288732,
         0.91410068283166412, 0.8465148485943953, 0.92865493168788427,
         0.92176087078518831],
        [2.0, 1.7081225333343879, 1.6702825615437937, 1.7035751769754379, 1.8592131117797068, 1.3220711049812315, 1.8027503009550137]
    ]
}


# This function returns the alpha value to use for
# predicting the outcome of a game in the given round
# between the given seeds s1, s2.
def getAlpha(s1, s2, model, year, roundNum):
    # Round 1 grouped alpha values for predicting 2013-2020,
    # where the first index is the better seed and the
    # second index is [year - 2013]. The grouping is:
    # 1, 2, 3-4, 5-7, 8
    r1GroupedAlphas = [
        [],
        [2,2,2,2,2,2,1.7692038993],
        [1.4252197727,1.3625656943,1.3804523797,1.3977167918,1.3440101948,1.3602838429,1.3760407791],
        [1.1327439019,1.1199577274,1.1293433883,1.1189438356,1.1083639629,1.1365944197,1.1245771481],
        [1.1327439019,1.1199577274,1.1293433883,1.1189438356,1.1083639629,1.1365944197,1.1245771481],
        [0.992260877,0.996405404,0.9802794213,1.0053287528,0.9535901796,0.947768435,0.9722115628],
        [0.992260877,0.996405404,0.9802794213,1.0053287528,0.9535901796,0.947768435,0.9722115628],
        [0.992260877,0.996405404,0.9802794213,1.0053287528,0.9535901796,0.947768435,0.9722115628],
        [0,0,0,0,0,0,0]]

    r1GroupedAlphasSeedWeighted = [
        [],
        [2,2,2,2,2,2,1.7692038993],
        [1.4252197727,1.3625656943,1.3804523797,1.3977167918,1.3440101948,1.3602838429,1.3760407791],
        [1.1239462408,1.1116452654,1.120961419,1.1095749793,1.0990011131,1.1266664795,1.1160620929],
        [1.1239462408,1.1116452654,1.120961419,1.1095749793,1.0990011131,1.1266664795,1.1160620929],
        [0.9481178098,0.9341775923,0.9014353156,0.928223297,0.8792796956,0.8692346352,0.8944760731],
        [0.9481178098,0.9341775923,0.9014353156,0.928223297,0.8792796956,0.8692346352,0.8944760731],
        [0.9481178098,0.9341775923,0.9014353156,0.928223297,0.8792796956,0.8692346352,0.8944760731],
        [0,0,0,0,0,0,0]]

    # Round 1 separated alpha values for predicting 2013-2020,
    # where the first index is the better seed and the
    # second index is [year - 2013].
    r1SeparateAlphas = [
        [],
        [2,2,2,2,2,2,1.7692038993],
        [1.4252197727,1.3625656943,1.3804523797,1.3977167918,1.3440101948,1.3602838429,1.3760407791],
        [1.1631440406,1.1437646,1.1260389104,1.0702482606,1.0570363456,1.0808615169,1.1038431398],
        [1.1023437632,1.0961508547,1.1326478663,1.1676394106,1.1596915802,1.1923273226,1.1453111565],
        [0.7612823908,0.6898202312,0.6242869483,0.6828764698,0.6603066747,0.6767844807,0.7293107575],
        [1.0995538121,1.1222629842,1.0820607889,1.0447312883,0.9537101213,0.86947563,0.8427577133],
        [1.1159464279,1.1771329966,1.2344905268,1.2883785003,1.2467537426,1.2970451943,1.3445662175],
        [0,0,0,0,0,0,0]]

    # Rounds 2-6 weighted average alpha values for predicting
    # 2013-2020, where the first index is [roundNum - 2] and
    # the second index is [year - 2013].
    r2to6Alphas = [
        [1.0960226368,1.0255184405,1.0280047853,1.0169015383,1.0085075325,1.0517190671,1.0349243918],
        [0.9074472394,0.8963083681,0.8581664326,0.8815834483,0.9021714769,0.9088993287,0.8644826467],
        [0.3579691718,0.2302351327,0.1909716145,0.2167374254,0.136706458,0.1188463061,0.1504395788],
        [0.6673769231,0.6983681575,0.5784406838,0.6093441472,0.6389325696,0.674510496,0.7010202861],
        [1.4133971593,1.4171625002,1.441447396,1.441447396,1.1671880002,1.1671880002,1.199219231]]

    r2to6AlphasSeedWeighted = [
        [0.7300894976,0.7256867311,0.7262248008,0.7307164601,0.7402819935,0.7447518994,0.7411460851],
        [0.7154059556,0.7241293547,0.7009183735,0.7114893029,0.7391687766,0.7355894309,0.7143276345],
        [0.5619089765,0.5264272183,0.4945600154,0.4919846802,0.4697685742,0.4595952558,0.4593748076],
        [0.8217277548,0.8254069985,0.6989678614,0.7047260475,0.7106414922,0.7227985752,0.7466939569],
        [0.711033407,0.7322638446,0.8439593326,0.8439593326,0.813649393,0.813649393,0.8276050547]]

    isR1Grouped = 'False'
    if 'isR1Grouped' in model:
        isR1Grouped = 'True' in model['isR1Grouped']

    isSeedWeighted = ('isSeedWeighted' in model) and ('True' in model['isSeedWeighted'])

    alpha = 0

    if (isR1Grouped) and (roundNum == 1):
        if isSeedWeighted:
            alpha = r1GroupedAlphasSeedWeighted[s1][year - 2013]
        else:
            alpha = r1GroupedAlphas[s1][year - 2013]
    elif roundNum == 1:
        alpha = r1SeparateAlphas[s1][year - 2013]
    else:
        if isSeedWeighted:
            alpha = r2to6AlphasSeedWeighted[roundNum - 2][year - 2013]
        else:
            if model.get('perturbation'):
                if roundNum in model['perturbation'].get('rounds', []):
                    alpha = otherAlphas[model['perturbation']['alphaMLEprobs']][roundNum - 2][year - 2013]
                else:
                    alpha = r2to6Alphas[roundNum - 2][year - 2013]
            else:
                alpha = r2to6Alphas[roundNum - 2][year - 2013]
    return alpha


# Unused: if we want to measure this later, we can.
#
# # This function computes how many picks a bracket
# # got correct given the bracket's score vector.
# def calcCorrectPicks(scoreVector):
# 	numCorrectPicks = 0
# 	for roundNum in range(1, 7):
# 		numCorrectPicks += scoreVector[roundNum] / (10 * (2 ** (roundNum - 1)))
# 	return numCorrectPicks


# This function generates and scores brackets
# for the given year using the given model.
# It prints the results in JSON format.
def performExperiments(numTrials, year, batchNumber, model):
    summarizer = RuntimeSummary(model)
    correctVector = getActualBracketVector(year)

    scores = [None] * numTrials
    for n in range(numTrials):
        newBracketVector = generateBracket(model, year)
        summarizer.analyze_bracket(np.array(newBracketVector))
        newBracketScore = scoreBracket(newBracketVector, correctVector)
        # numCorrectPicks = calcCorrectPicks(newBracketScore)
        scores[n] = newBracketScore[0]

    bracketListDict = {'year': year, 'actualBracket': ''.join(str(bit) for bit in correctVector), 'scores': scores}

    if numTrials < 1000:
        folderName = 'Experiments/{0}Trials'.format(numTrials)
    else:
        folderName = 'Experiments/{0}kTrials'.format(int(numTrials / 1000))
    batchFolderName = '{0}/Batch{1:02d}'.format(folderName, batchNumber)

    outputFilename = '{2}/generatedScores_{0}_{1}.json'.format(model['modelName'], year, batchFolderName)
    summaryFilename = '{2}/vectorStats_{0}_{1}.json'.format(model['modelName'], year, batchFolderName)
    with open(outputFilename, 'w') as outputFile:
        outputFile.write(json.dumps(bracketListDict))
    summarizer.to_json(summaryFilename)


######################################################################
# This script runs experiments with the given models,
# number of trials, and number of batches for 2013 through 2018.
######################################################################

# Load models
modelFilename = sys.argv[3]
with open(modelFilename, 'r') as modelFile:
    modelsDataJson = modelFile.read().replace('\n', '')

modelsDict = json.loads(modelsDataJson)
modelsList = modelsDict['models']

numTrials = int(sys.argv[1])
numBatches = int(sys.argv[2])
if len(sys.argv) == 5:
    years = [int(sys.argv[4])]
else:
    years = range(2013, 2020)

for modelDict in modelsList:
    modelName = modelDict['modelName']

    print '{0:<8s}: {1}'.format(modelName, time.strftime("%Y-%m-%d %H:%M"))

    for year in years:
        print '\t {0}: {1}'.format(year, time.strftime("%Y-%m-%d %H:%M"))
        for batchNumber in range(numBatches):
            print '\t\t {0}: {1}'.format(batchNumber, time.strftime("%Y-%m-%d %H:%M"))
            if numTrials < 1000:
                folderName = 'Experiments/{0}Trials'.format(numTrials)
            else:
                folderName = 'Experiments/{0}kTrials'.format(int(numTrials / 1000))

            if not os.path.exists(folderName):
                os.makedirs(folderName)

            batchFolderName = '{0}/Batch{1:02d}'.format(folderName, batchNumber)
            if not os.path.exists(batchFolderName):
                os.makedirs(batchFolderName)

            performExperiments(numTrials, year, batchNumber, modelDict)
