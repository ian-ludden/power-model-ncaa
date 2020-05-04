import json
import scoringFunctions as sf
import bracketManipulations as bm
import fileUtils
import numpy as np
import math
import scipy.stats
import itertools
from heapq import nlargest
import scipy.stats as sp
import os
from heapq import heapify, heappush, heappushpop,heappop
import csv
import cvxpy as cp


TOPSIDE = [1,16,8,9,5,12,4,13]
BOTSIDE = [6,11,3,14,7,10,2,15]

class MaxHeap():
    def __init__(self, top_n):
        self.h = []
        self.length = top_n
        heapify( self.h)

    def add(self, element):
        if len(self.h) < self.length:
            heappush(self.h, element)
        else:
            heappushpop(self.h, element)

    def getTop(self):
        return sorted(self.h, reverse=True)

    def popout(self):
        return heappop(self.h)

### filters are constructed via a rule definition



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
        if model == 'power' or model == "neoPower" or "powerRule" in model:
                filepath += '_{0}_{1}'.format(r, samplingFnName)
        filepath += '.json'
        return filepath





# rule creation from certain filters, filters being :
"""
r : round specification  (set of ints)
s : seed specification  (set of ints)

c : conditionals  ([upsets, both]) list
      upsets : only counts games in which upsets win
      or
      favs : only counts games in which favs win
      or
      normal 


      both : on seed specification, only look at matchups in which both seeds in matchup is in


e : evaluation method ([sumOfSeeds,prodOfSeeds,sumOfWins,sumOfSeedDif,sumOfSeedProp]) single string
      sumOfSeedSum : sum of matchup seeds, trivial for r 1
      sumOfSeedProd : product of matchup seeds, trivial for r1
      sumOfWins : +1 if seed that wins is in s 
      sumOfSeedWin : +seed if seed that wins is in s, trivial for s that is of size 1
      sumOfSeedDif/Prop : dif/Prop of matchup seeds trivial for r 1, prop is always winner / loser. dif is winner - loser


After calling the aggregate function in bracketManipulations on a particular bracket. I am garnered a data type that index's the bracket by round, seed. 


threshold: the max or min on  the evaluation method, evaluation <= threshold or evlauation >= threshold, at most, at least



"""
ROUND_DICT = {3:{1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:4,10:4,11:6,12:6,13:7,14:7,15:8,16:8}}



# r,s,c,e,t
# remember it is always count < t
RULEV1 = [
[[4],[1,2,3,4],[],"sumOfWins",[3]],
[[5],[i for i in range(1,17)],[],"sumOfSeedSum",[15]],
[[5],[i for i in range(1,17)],[],"sumOfSeedProd",[18]], [[4],[5,6,7,8],[],"sumOfWins",[2]],
[[4],[12,13,14,15,16],[],"sumOfWins",[1]] ,
[[5],[i for i in range(1,17)],[],"sumOfSeedDif",[5]], [[4],[1,2,3,4],[],"sumOfSeedWin",[10]]
]

# sum of seed prod remove
# sum of seed dif remove

RULEV2 = [
[[4],[1,2,3,4],[],"sumOfWins",[3]],
[[5],[i for i in range(1,17)],[],"sumOfSeedSum",[15]],
[[4],[5,6,7,8],[],"sumOfWins",[2]],
[[4],[12,13,14,15,16],[],"sumOfWins",[1]] ,
[[4],[1,2,3,4],[],"sumOfSeedWin",[10]]
]


RULEV3 = [
[[4],[1,2,3,4],[],"sumOfWins",[1,2,3,4]],


[[4],[5,6,7,8],[],"sumOfWins",[1,2,3,4]],


[[4],[12,13,14,15,16],[],"sumOfWins",[1,2,3,4]] ,


[[4],[1],[],"sumOfWins",[1,2,3,4]],


[[4],[1,2],[],"sumOfWins",[1,2,3,4]],

[[4],[1,2,3],[],"sumOfWins",[1,2,3,4]],

        [[4],[1,2,3,4],[],"sumOfSeedWin",[i for i in range(4,17)]],

[[5],[i for i in range(1,17)],[],"sumOfSeedSum",[i for i in range(4,29)]]

]

RULEV4 = [

        ##########################################################
        # final 4
        [[4],[1,2,3,4],[],"sumOfWins",[i for i in range(0,5)]],
    
        # final 4 top 4 winning seed sum
        [[4],[1,2,3,4],[],"sumOfSeedWin",[i for i in range(0,17)] ],
        # final 4 medium seed sum
    
        [[4],[5,6,7,8,9,10],[],"sumOfSeedWin",[i for i in range(0,41)]],
        # final 4 bot seed sum
        [[4],[11,12,13,14,15,16],[],"sumOfSeedWin",[i for i in range(0,65)]],
        [[4],[5,6,7,8,9,10],[],"sumOfWins",[i for i in range(0,41)]],
        # final 4 bot seed sum
        [[4],[11,12,13,14,15,16],[],"sumOfWins",[i for i in range(0,65)]],
    
        # final 4 seed sum
        [[4],[i for i in range(1,17)],[],"sumOfSeedWin",[i for i in range(0,65)]],


        [[4],[1,2,3],[],"sumOfWins",[i for i in range(0,5)]],
        [[4],[1,2],[],"sumOfWins",[i for i in range(0,5)]],
        [[4],[1],[],"sumOfWins",[i for i in range(0,5)]],

    
        [[4],[i for i in range(17)],[],"betterLoses",[i for i in range(0,5)]],

        ##########################################################
        # elite 8s
        [[3],[1,16,8,9,5,12,4,13],[],"sumOfSeedWin",[i for i in range(0,41)]],
        # top half seed sum
        [[3],[6,11,3,14,7,10,2,15],[],"sumOfSeedWin",[i for i in range(0,41)]],
        # bot half seed sum

        # condition is trivial
        [[3],[1,16,8,9,5,12,4,13],[3],"sumOfSeedWinReordered",[i for i in range(0,33)]],
        # top half seed sum, reordered
        [[3],[6,11,3,14,7,10,2,15],[3],"sumOfSeedWinReordered",[i for i in range(0,33)]],
        # bot half seed sum, reordered

        
        [[3],[1,4],[],"sumOfWins",[0,1,2,3,4]], # high
        [[3],[2,3],[],"sumOfWins",[0,1,2,3,4]],
        [[3],[1,4],[],"sumOfSeedWin",[i for i in range(0,17)]], # high
        [[3],[2,3],[],"sumOfSeedWin",[i for i in range(0,13)]],
    
        [[3],[16,12,13],[],"sumOfWins",[0,1,2,3,4]], # low
        [[3],[11,14,15],[],"sumOfWins",[0,1,2,3,4]],
        [[3],[16,12,13],[],"sumOfSeedWin",[i for i in range(0,33)]], # low
        [[3],[11,14,15],[],"sumOfSeedWin",[i for i in range(0,33)]],
    
        [[3],[5,8,9],[],"sumOfWins",[0,1,2,3,4]], # mid 
        [[3],[6,7,10],[],"sumOfWins",[0,1,2,3,4]],
        [[3],[5,8,9],[],"sumOfSeedWin",[i for i in range(33)]], # mid 
        [[3],[6,7,10],[],"sumOfSeedWin",[i for i in range(33)]],


        [[3],[1,16,8,9,5,12,4,13],[],"sumOfSeedDif",[i for i in range(-10,100)]],
        [[3],[6,11,3,14,7,10,2,15],[],"sumOfSeedDif",[i for i in range(-10,100)]],
        ############################################################


        # first round second round
        [[1],[i for i in range(1,17)],[],"sumOfUpsets",[i for i in range(0,25)] ],
        [[2],[i for i in range(1,17)],[],"sumOfUpsets",[i for i in range(0,13)]],

        [[1],[i for i in range(1,17)],[],"betterLoses",[i for i in range(0,25)] ],
        [[2],[i for i in range(1,17)],[],"betterLoses",[i for i in range(0,13)]],



]

# to be used in powerRule
RULEV5 = [    
    # sum of seeds 
    [[4],[i for i in range(1,17)],[],"sumOfSeedWin",[i for i in range(4,29)] ],

    [[3],[1,16,8,9,5,12,4,13],[],"sumOfSeedWin",[i for i in range(4,36)]],
    # top half seed sum
    [[3],[6,11,3,14,7,10,2,15],[],"sumOfSeedWin",[i for i in range(8,36)]],
    # bot half seed sum



    # better loses, lower seed wins
    [[4],[i for i in range(1,17)],[],"betterLoses",[i for i in range(0,5)]],
    [[3],[i for i in range(1,17)],[],"betterLoses",[i for i in range(0,8)]],
    [[2],[i for i in range(1,17)],[],"betterLoses",[i for i in range(0,10)]],
    [[1],[i for i in range(1,17)],[],"betterLoses",[i for i in range(0,14)]],
    [[3],TOPSIDE,[],"betterLoses",[i for i in range(0,5)]],
    [[3],BOTSIDE,[],"betterLoses",[i for i in range(0,5)]],


    # seed appearences
    [[1],[1],[],"sumOfWins",[i for i in range(0,5)]],
    [[1],[1,2,3,4],[],"sumOfWins",[i for i in range(0,17)]],

    [[1],[5,6,7,8,9,10],[],"sumOfWins",[i for i in range(0,25)]],
    [[1],[11,12,13,14,15,16],[],"sumOfWins",[i for i in range(0,25)]],

    
    [[2],[1],[],"sumOfWins",[i for i in range(0,5)]],
    [[2],[1,2,3,4],[],"sumOfWins",[i for i in range(0,17)]],

    [[2],[5,6,7,8,9,10],[],"sumOfWins",[i for i in range(0,25)]],
    [[2],[11,12,13,14,15,16],[],"sumOfWins",[i for i in range(0,25)]],

    [[3],[1],[],"sumOfWins",[i for i in range(0,5)]],
    [[3],[1,2,3,4],[],"sumOfWins",[i for i in range(0,17)]],

    [[3],[5,6,7,8,9,10],[],"sumOfWins",[i for i in range(0,25)]],
    [[3],[11,12,13,14,15,16],[],"sumOfWins",[i for i in range(0,25)]],
    
    [[4],[1,2,3,4],[],"sumOfWins",[i for i in range(0,5)]],
    
    [[4],[1,2],[],"sumOfWins",[i for i in range(0,5)]],
    
    [[4],[1],[],"sumOfWins",[i for i in range(0,5)]],
    
    [[4],[5,6,7,8,9,10],[],"sumOfWins",[i for i in range(0,5)]],

    [[4],[11,12,13,14,15,16],[],"sumOfWins",[i for i in range(0,5)]],


    # seed sum order
    [[4],[i for i in range(1,17)],[1],"maxSums",[i for i in range(0,17)]],
    [[4],[i for i in range(1,17)],[-3],"maxSums",[i for i in range(0,49)]],
    [[4],[i for i in range(1,17)],[2],"maxSums",[i for i in range(0,33)]],
    [[4],[i for i in range(1,17)],[-2],"maxSums",[i for i in range(0,33)]],
    [[4],[i for i in range(1,17)],[3],"maxSums",[i for i in range(0,49)]],
    [[4],[i for i in range(1,17)],[-1],"maxSums",[i for i in range(0,17)]],
  

    [[3],[1,16,8,9,5,12,4,13],[1],"maxSums",[i for i in range(0,17)]],
    [[3],[1,16,8,9,5,12,4,13],[-3],"maxSums",[i for i in range(0,49)]],
    [[3],[1,16,8,9,5,12,4,13],[2],"maxSums",[i for i in range(0,33)]],
    [[3],[1,16,8,9,5,12,4,13],[-2],"maxSums",[i for i in range(0,33)]],
    [[3],[1,16,8,9,5,12,4,13],[3],"maxSums",[i for i in range(0,49)]],
    [[3],[1,16,8,9,5,12,4,13],[-1],"maxSums",[i for i in range(0,17)]],


    [[3],[6,11,3,14,7,10,2,15],[1],"maxSums",[i for i in range(0,17)]],
    [[3],[6,11,3,14,7,10,2,15],[-3],"maxSums",[i for i in range(0,49)]],
    [[3],[6,11,3,14,7,10,2,15],[2],"maxSums",[i for i in range(0,33)]],
    [[3],[6,11,3,14,7,10,2,15],[-2],"maxSums",[i for i in range(0,33)]],
    [[3],[6,11,3,14,7,10,2,15],[3],"maxSums",[i for i in range(0,49)]],
    [[3],[6,11,3,14,7,10,2,15],[-1],"maxSums",[i for i in range(0,17)]],
  
    
    ]


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))



# to prune out some trivial rules
def trivialRule(r,s,c,e):
    if e in ["sumOfSeedSum","sumOfSeedProd"]:
        if 1 in r:
            return True

    if e in ["sumOfSeedWin"]:
        if len(s) == 1:
            return True



# rule to str, t really gets ignored
def ruleName(r,s,c,e,t):
    name = ""
    for round in r:
        name += str(round) +","
    name += "|"
    for seed in s:
        name += str(seed) + ","
    name += "|"
    for condition in c:
        name += str(condition) + ","
    name += "|"
    name += str(e) + "|"

        

    return name

def ruleNameNice(r,s,c,e,b,t):
    seeds = ""
    for seed in s:
        seeds+=str(str(seed)+",")
    if len(c) == 1:
        c = str(c[0])
    else:
        c= "" 
    return str(r[0])+" "+ seeds + e + c + b + str(t)

# str to rule
def nameRule(ruleString):
    parts = ruleString.split("|")

    r = [int(i) for i in parts[0].split(",")[:-1]]

    s = [int(i) for i in parts[1].split(",")[:-1]]

    c =  parts[2].split(",")[:-1]

    e = parts[3]

    t = parts[4]

    return [r,s,c,e,t]


# generate the set(list) of different round specification sets
def genR(v):

    if v == 1:  #r = 4 only, final four only
        R = [[3],[4]]
        
        return R

    else:
        R = []
        for i in range(1,7):
            cur = 1
            while cur+i <= 7:
                R.append([j for j in range(cur,cur+i)])
                cur += 1
        return R
    

# generate the set(list) of different set speciication sets
def genS():
    return [[1,2,3,4],[5,6,7,8],[13,14,15,16],[i for i in range(1,17)]]

def genC(v):

    if v == 1:
        return [[]]
        
    else:
        conditions = ['upsets','both']
        
        return list(powerset(conditions)) 


def genE():
    evaluations = ['sumOfSeedSum','sumOfSeedProd','sumOfWins','sumOfSeedWin','sumOfSeedDif','sumOfSeedProp']

    return evaluations



# used when e needs all matchups to compute
# matchups is a dict mapping matchup (seed1,seed2) to seed1: wins, seed2: wins
def evalMatchups(matchups,s,c,e):
    count = 0

    if e == "maxSums":
        winners = []
        for matchup in matchups:
           results = matchups[matchup]
           for one in results:
               if one in s:
                   wins = results[one]
                   for i in range(wins):
                       winners.append(one)

        winners.sort()
        if c[0] > 0:
            count = sum(winners[:c[0]])
        else:
            count = sum(winners[c[0]:])

    return count

# result here is a dict, with key being one of matchup (a pair). Value being number of wins
def evalMatchup(matchup,result,s,c,e):
    count = 0
    # conditions for both or not
    if ("both" in c) and (not (matchup[0] in s and matchup[1] in s) ):
        return count
    if ("both" not in c) and (not (matchup[0] in s or matchup[1] in s) ):
        return count

    
    # conditions if to only look at upsets 
    if "upsets" in c:
        fav = matchup[0] if matchup[0] < matchup[1] else matchup[1]
        for one in result:
            if one == fav:
                result[one] = 0

    # counts current
    if e == "sumOfSeedSum":
        seedSum = matchup[0]+matchup[1]
        for one in result:
            count += result[one] * seedSum

    # matchup1*matchup2 
    elif e == "sumOfSeedProd":
        seedProd = matchup[0]*matchup[1]
        for one in result:
            count += result[one] * seedProd

    elif e == "sumOfWins":
        for one in result:
            if one in s:
                count += result[one]

    # applied to current, so counts next round advances
    elif e == "sumOfSeedWin":
        for one in result:
            if one in s:
                count += result[one] * one

    
    elif e == "sumOfSeedDif":
        for one in result:
            if one == matchup[0]:
                count += result[one] * (matchup[1]-matchup[0])
            else:
                count += result[one] * (matchup[0] - matchup[1])
                
    elif e == "sumOfSeedProp":
        for one in result:
            if one == matchup[0]:
                count += result[one] * (matchup[0] / matchup[1])
            else:
                count += result[one] * (matchup[1] / matchup[0])
    # c will store the round number here
    elif e == "sumOfSeedWinReordered":
        ranks = ROUND_DICT[3]
        for one in result:
            if one in s:
                count += result[one] * ranks[one]

    elif e == "sumOfUpsets":
        if abs(matchup[0]-matchup[1]) == 1:
            return count 
        # 1 seed difference no difference
        bigger = matchup[0] if matchup[0] > matchup[1] else matchup[1]
        for one in result:
            if one == bigger:
                count += result[one]
    elif e == "betterLoses":
        if matchup[0]== matchup[1]:
            return count
        worst = matchup[0] if matchup[0] > matchup[1] else matchup[1]

        for one in result:
            if one == worst:
                count+= result[one]
                
    else:
        print("invalid evaluator")

                
    return count
            
    

# returns two brackets, x,y corresponding to those that follow the rule, and those that dont

# thresholds are determined here, or pre determined

# returns a x,y that are dicts for each threshold

#bracket 
def applyRule(brackets,scores,r,s,c,e,t=None,cotc = 0.007):

    # cream of the crop, percentage of top to use, top 0.7 %
    
    
    x = dict()
    y = dict()


    counts = []

    cur = 0
    for bracket in brackets:
        results = bm.aggregate([bracket])
        
        cur+=1
        print("\r"+str(cur)+":"+str(len(brackets)),end="")
        
        count = 0

        for round in r:
            allMatchups = []
            for matchup in results[round]:
                count += evalMatchup(matchup,results[round][matchup],s,c,e)
                allMatchups.append(matchup)

                
                
        counts.append(count)
        
    print("brackets evaluated")


    # To use default thresholds, make it take in None, instead of [...]
    # can be determined in other ways
    thresholds = t
    if t is None:
        thresholds = list(np.percentile(counts,[10,20,30,40,50,60,70,80,90]))
        for idx,i in enumerate(thresholds):
                thresholds[idx] = int(np.ceil(i))
    # default thresholds
    
    print(thresholds)
    
    for threshold in set(thresholds):
        x[threshold] = []
        y[threshold] = []

        x2 = []
        y2 = []
        for aCount,aScore in zip(counts,scores):
        
            if aCount <= threshold:
                x2.append(aScore)
            else:
                y2.append(aScore)

        
        x[threshold] = nlargest(int(len(x2)*cotc),x2)
        y[threshold] = nlargest(int(len(y2)*cotc),y2)

   
    return x,y,counts






# rule = (r,s,c,e,threshhold)
# returns   size X ,U statistic, p value, and if p value <= alpha 

def toFollowOrNotToFollowRule(brackets,scores,rule,alpha):

    x,y,counts = applyRule(brackets,scores,*rule)

    resultDict = dict()

    count = 0

    for threshold in x:
        print(count,len(x),"threshold")
        count += 1
        
        if min(len(x[threshold]),len(y[threshold])) <= 20:
            resultDict[threshold] =  [len(x[threshold]),-1,-1,False] # insufficient sample size
            continue

        U,p = scipy.stats.mannwhitneyu(x[threshold],y[threshold],alternative = "greater")

        # for now alpha value does not matter
        if p <= alpha :
            resultDict[threshold] =  {"size":len(x[threshold]),"x":x[threshold],"y":y[threshold],"U":U,"p":p}
        else:
            resultDict[threshold] =  {"size":len(x[threshold]),"x":x[threshold],"y":y[threshold],"U":U,"p":p}


    return resultDict


    

# bracket here should be of list format

# output file format, json.  Year -> rule - threshold -> threshold -> [len(x),U,p, toFollow ]
# modelName e.x sampleF4A

def runRulesSaveResults(modelName,ruleSet = None,v=1):

    outputFilepath = modelName + "V"+str(v)+"35" + ".json"
    #outputFilepath = "test.json"
    outputDict = dict()


    # 111111
    
    
    R = genR(v)
    S = genS()
    C = genC(v)
    E = genE()

    alpha = 0.2

    outputDict["R"] = R
    outputDict["S"] = S
    outputDict["C"] = C
    outputDict["E"] = E

    outputDict["alpha"] = alpha
    


    # file:///home/nd2/Documents/Research/Sheldon Jacobson/power-model-ncaa/Outputs/generatorOutputs/power_20_x_500k_2013_1_sampleF4A.json is the 10 mil
    bracketFile = generateFilepath(500000,year = 2013,r = 5,samplingFnName =modelName,nReplications = 20)


    #bracketFile = generateFilepath(50000,year = 2013,r = 5,samplingFnName =modelName,nReplications = 25,model="powerRule")
    with open(bracketFile, 'r') as f:
        data = json.load(f)    

        
    #for year in range(2013,2020):
    for year in range(1985,2020):
        yearDict = dict()




        # uses 5 mil
        brackets = []
        scores = []

        count = 0

        # a total of 5 million
        for i in range(0,10):
            for bracket in data["brackets"][i]:
                aBracket = bm.stringToVector(bm.hexToString(bracket))
                brackets.append(aBracket)
                scores.append(sf.scoreBracket(aBracket,year=year)[0])

                count+=1
                print("\r"+str(count)+":"+str(len(data["brackets"][i])),end='')
                
        print(str(year) + " loaded!")
        print("next")
            
        totalRules = len(R) * len(S) * len(C) * len(E)
        count = 0

        if ruleSet is None:
            for r in R:
                for s in S:
                    for c in C:
                        for e in E:
                            count+=1
                            print(count,totalRules)

                            if trivialRule(r,s,c,e):
                                continue

                            rule = [r,s,c,e]
                            resultDict = toFollowOrNotToFollowRule(brackets,scores,rule,alpha)
                            name = ruleName(r,s,c,e)
                            print(name)
                            yearDict[name] = resultDict
                            break
                        break
                    break
                break
        else:
            for rule in ruleSet:
                 count+=1
                 print(count,len(ruleSet))
                 resultDict = toFollowOrNotToFollowRule(brackets,scores,rule,alpha)
                 name = ruleName(*rule)
                 print(name)
                 yearDict[name] = resultDict

        outputDict[year] = yearDict

        
                    
    
    with open(outputFilepath,'w') as outputFile:
        outputFile.write(json.dumps(outputDict))
 


# Takes in input file, returns a dict of the probabilities for each rule each t

# [rule][t]: probability
# calculates probw

# input file is rulesV4.json
# returns x %'s for each year each rule

def ruleProbs(inputFile,yearPredict = 2019):
    with open(inputFile, 'r') as f:
        results = json.load(f)


    # for each rule, for each t, stores percent list : [absolute dif, indicator, count dif]
    
    bins = dict()

    
    

    top100Count = dict()
    # rule to count
    # pdf

    for rule in results['partitionCount']:
        top100Count[rule] = dict()
        # this dict is a threshold to [top 100 partition, total partition]
        for year in range(1985,2020):
            year = str(year)
            top100Count[rule][year] = dict()
            for t in results['partitionCount'][rule]:
                top100Count[rule][year][t] = [0,results['partitionCount'][rule][t]/1000000 ]

    for rule in results['top100']:
        for year in range(1985,2020):
            year = str(year)
            for t in results['top100'][rule][year]:
                cur = top100Count[rule][year][t]
                cur[0] += len(results['top100'][rule][year][t]['x']) / 100
                top100Count[rule][year][t] = cur

    # stores the propotion satisfying each rule. top100, 1mil
    allPs = dict()
    for rule in top100Count:
        top100Ps = []
        milPs = []
        for year in range(1985,yearPredict+1):
            idx = year-1985
            idx1 = int(np.floor(idx/5))
            idx2 = idx % 5
            year = str(year)

            top100P = []
            for idx,t in enumerate(top100Count[rule][year]):
                if idx == 0:
                    top100P.append(top100Count[rule][year][t][0])
                else:
                    top100P.append(top100Count[rule][year][t][0]-top100Count[rule][year][str(int(t)-1)][0])

            milP = []
            for idx,t in enumerate(top100Count[rule][year]):
                if idx == 0:
                    milP.append(top100Count[rule][year][t][1])
                else:
                    milP.append(top100Count[rule][year][t][1]-top100Count[rule][year][str(int(t)-1)][1])
            ts = list(top100Count[rule][year].keys())

            top100Ps.append(top100P)
            milPs.append(milP)
        allPs[rule] = ([top100Ps,milPs])
       


    # stores the list of percentages for each rule for each performance metric
    # for each rule, stores dict t : percent

    # rule : info
    X1s = dict()
    X2s = dict()
    X3s = dict()

    for rule in top100Count:
        ps = allPs[rule]

        # matrics
        A1 = [] # absolute dif
        A2 = [] # +1 -1 indicator 
        A3 = [] # indicator count no lp
        
        for year in range(1985,yearPredict+1):
            year = str(year)
            top100P = allPs[rule][0][int(year)-1985]
            milP = allPs[rule][1][int(year)-1985]
            
            
            # rows
            a1 = []
            a2 = []
            a3 = []
            
            for idx,t in enumerate(list(top100Count[rule][year].keys())):
                a1.append(top100P[idx]-milP[idx])
                a2.append(1 if top100P[idx] > milP[idx] else -1)
                a3.append(1 if top100P[idx] > milP[idx] else 0)
                
            A1.append(a1)
            A2.append(a2)
            A3.append(a3)
        
        # counts
        colSums = [0 for t in range(len(A1[0]))]
        for col in range(len(colSums)):
            colSum = 0
            for year in range(1985,yearPredict+1):
                colSum += A3[year-1985][col]
            colSums[col] = colSum

        X3 = [0 if sum(colSums) == 0 else float(i)/sum(colSums) for i in colSums]
        X3s[rule] = dict()
        for idx,t in enumerate(list(top100Count[rule]['1985'].keys())):
            X3s[rule][t] = X3[idx]
        
        #LP time

        m = len(A1) # m years
        n = len(A1[0]) # n t  
        z = cp.Variable()
        x = cp.Variable(n)
        #x = cp.Variable(m)

        a = cp.Parameter((m,n))

        a.value = np.asarray(A1)
        obj = cp.Maximize(z)
        constraints = []
        for one in range(m):
            constraints.append(z - (a[one,:]@x)  <= 0)   
        constraints += [sum(x) == 1, x >= 0]    
        prob = cp.Problem(obj,constraints)
        prob.solve()
        X1Raw = [int(i*100) for i in x.value]

        a.value = np.asarray(A2)
        obj = cp.Maximize(z)
        constraints = []
        for one in range(m):
            constraints.append(z - (a[one,:]@x)  <= 0)   
        constraints += [sum(x) == 1, x >= 0]    
        prob = cp.Problem(obj,constraints)
        prob.solve()
        X2Raw = [int(i*100) for i in x.value]
        
        X1 = [0 if i == 0 else i/sum(X1Raw) for i in X1Raw]
        X2 = [0 if i == 0 else i/sum(X2Raw) for i in X2Raw]
        
        X1s[rule] = dict()
        X2s[rule] = dict()
        for idx,t in enumerate(list(top100Count[rule]['1985'].keys())):
            X1s[rule][t] = X1[idx]
            X2s[rule][t] = X2[idx]
            
    return [X1s,X2s,X3s]






def ruleThresholdDetermine(ruleSet=RULEV5,v=1,alpha = 0.05):
    # as of now, the ruleV51Mil.json file containts the following
    # alphaBetas should suffice
    
   # with open("rulesV"+str(v)+"1Milstats.json","r") as f:
    #    results = json.load(f)

    with open("alphaBetas.json","r") as f:
        results = json.load(f)
    
 

    # go thru each rule, for each t tally the 35 count, compute the binomial test, and keep rule if it passes. do this for both <=t and >t


    niceRules = dict()
    for rule in results:
        alphas = results[rule][0]
        beta = results[rule][1]

        S = 0
        for alpha in alphas:
            if alpha > (1-beta):
                S+=1
        
                
        # lowerbounds
        p = 1 - beta
        if beta < 0.1:
            continue

        pStar = sp.binom.sf(100*p,100,p)

        if S == 0:
            continue
        if sp.binom.sf(S-1,7,pStar) <= alpha:
            niceRules[rule] = S

    for rule in niceRules:
        print(rule)
    return niceRules

def ruleTopResults(ruleSet, v = 1,year = 2019):
        with open("sampleF4AScored1Mil35Years.json","r") as f:
            brackets = json.load(f)
        # 1 mil brackets

        


        
        # for testing 
        '''
        lol = dict()
        total = 0
        lel = list(brackets.keys())
        while total < 10:
                total+=1
                lol[lel[total]] = brackets[lel[total]]
        brackets = lol
        '''
          
        # result dict is the following
        # result["partitionCount"] ["ruleName"][threshold] = number out of 1 mil that satisfy rule.

        
        # result["top100"]["ruleName"][year][threshold] = ([total scores],[total scores])
        # now just store count that follows that
        result = dict()


        # now also stores each individual bracket stats, along with each of the 35 years top 100s

        result["partitionCount"] = dict()
        result["top100"] = dict()

        
        result["1mil"] = dict()
        # result["1mil"][bracket][ruleName] = count
        
        
        for rule in ruleSet:
                
                name = ruleName(*rule)
                result["partitionCount"][name] = dict()
                result["top100"][name] = dict()
                
                for year in range(1985,2020):
                        result["top100"][name][year] = dict()
                
                        for t in rule[-1]:
                                result["top100"][name][year][t] = 0
                                result["partitionCount"][name][t] = 0 # redundant for all years but its ok
                                #result["top100"][name][year][t]["x"] = []
                                #result["top100"][name][year][t]["y"] = []

                       
        # brackets are counted with respect to the respective rule
        # save this bracket count
        bracketCounts = dict()
        done = 0
        for bracket in brackets:
                print("finished",done)
                done+=1
                
                bracketCounts[bracket] = dict()
                # stores count for each rule for each bracket
                for rule in ruleSet:
                        r = rule[0]
                        s = rule[1]
                        c = rule[2]
                        e = rule[3]
                        results = bm.aggregate([bm.stringToVector(bm.hexToString(bracket))])
        
       
                        count = 0

                        if e in ["maxSums",""]:
                            for round in r:
                                count += evalMatchups(results[round],s,c,e)
                            
                        else:
                            for round in r:
                                for matchup in results[round]:
                                    count += evalMatchup(matchup,results[round][matchup],s,c,e)
                                
                        bracketCounts[bracket][ruleName(*rule)] = count
        print("allCounts tallied")
        # find top 100 per year

        # store all the counts relative to rule
        #result["1mil"] = bracketCounts
        
        for year in range(1985,2020):
                # use heap, with pair (score, bracket)
                
                top100 = MaxHeap(100)
                for bracket in brackets:
                        top100.add((brackets[bracket][year-1985][0],bracket))
                # should have top 100
                
                while len(top100.h) > 0 :
                        cur = top100.popout()
                        for rule in ruleSet:
                                # evaluate rule on this bracket
                                name = ruleName(*rule)
                                for t in result["top100"][name][year]:
                                        if bracketCounts[cur[1]][name] <= t:
                                                #result["top100"][name][year][t]["x"].append(cur[1])
                                                result["top100"][name][year][t]+=1
                                        else:
                                                #result["top100"][name][year][t]["y"].append(cur[1])
                                                continue
                
        # tally partition sizes
        for rule in ruleSet:
                for t in rule[-1]:
                        count = 0
                        name = ruleName(*rule)
                        for bracket in brackets:
                                if bracketCounts[bracket][name] <= t:
                                        count+=1
                        result["partitionCount"][name][t] = count

        with open("rulesV"+str(v)+".json","w") as f:
                f.write(json.dumps(result))



        
def randomRules(probDict,year):
    boolList = [0 for i in range(len(probDict))]

    for idx,rule in enumerate(probDict):
        rand = np.random.uniform()
        if rand < float(probDict[rule][year]):
            boolList[idx] = 1
            
    return boolList


# probDict is a dict that is output of ruleProbs
# 0 dont add
# 1 add
# 2 donezo

def rulesSatisfy(ruleProbs, bracket,metric):

    rules = ruleProbs[metric]
    # doing the nash way


    results = bm.aggregate([bracket])

    counts = [0 for i in range(len(rules))]
    
    for idx,one in enumerate(rules):
        rule = nameRule(one)

        count = 0

        #########################
        #if 4 in rule[0]:
            #continue
        #########################
        for round in rule[0]:
            # skip the final 4 restrictions
   
            
            round = int(round)
            
            

            for matchup in results[round]:

               
                count += evalMatchup(matchup, results[round][matchup],rule[1],rule[2],rule[3])

        if str(count) not in rules[one]:
            #print("out of bounds")
            return 0
        if rules[one][str(count)] == 0:
            #print("no need")
            return 0

        counts[idx] = (count)

    # decrement bins
    for idx,one in enumerate(rules):
        # skip final 4
        #######################
        #if 4 in rule[0]:
            #continue
        #######################
        rules[one][str(counts[idx])] -= 1

    return 1



# takes in a score bracket and returns dict with the following
# result[top100][year] = returns brackets that are top 100
# result[brackets][bracket][statName][value]
# stat name is in ruleName form

# score bracket is just dict, bracket:score
def bracketStatCounting(ruleSet=RULEV5, scoredBrackets):
        #with open(scoredBrackets,"r") as f:
         #   brackets = json.load(f)
        # 1 mil brackets

    brackets = scoredBrackets
    result = dict()
    result["brackets"] = dict()
    result["top100"] = dict()

    for year in range(2013,2020):
            # use heap, with pair (score, bracket)
        top100s = []
        top100 = MaxHeap(100)
        for bracket in brackets:
            top100.add((brackets[bracket][year-1985],bracket))
            # 1985 if using 35 years

            
            # dont need [0] later

            # need [0] if using old, check the file writes too

            # should have top 100

        while len(top100.h) > 0 :
            cur = top100.popout()
            top100s.append(cur[1])
        result["top100"][year] = top100s
        print("year found")

    lel = 0
    for bracket in brackets:
        print(lel)
        lel+=1
        stats = dict()
        results = bm.aggregate([bm.stringToVector(bm.hexToString(bracket))])

        for rule in ruleSet:
            r = rule[0]
            s = rule[1]
            c = rule[2]
            e = rule[3]


            count = 0

            if e in ["maxSums",""]:
                for round in r:
                    count += evalMatchups(results[round],s,c,e)

            else:
                for round in r:
                    for matchup in results[round]:
                        count += evalMatchup(matchup,results[round][matchup],s,c,e)

            stats[ruleName(*rule)] = count

        result["brackets"][bracket] = stats

    v= 5
    print("ready to dump")

    with open("rulesV"+str(v)+"1Milstats.json","w") as f:
        json.dump(result, f, ensure_ascii=False)


# inputs are file outputted by bracketStatCounting, and ruleSet
def alphaBetas(bracketStats,ruleSet = RULEV5):
    print("start to read")
    with open(bracketStats,"r") as f:
        stats = json.load(f)
    results = dict()
    # results[rule] = [[alphas], beta ]

    
    #rules = rules[1:]
    
    print("starting")
    count = 0
    for rule in ruleSet:
        print(count)
        count+=1

        
        betas = [[0 for i in range(len(rule[-1]))],[0 for i in range(len(rule[-1]))]] # <= and >, number reject in total

        
        # compute how many reject in 1mil
        # compute for each year how many accept

        name = ruleName(*rule)
        
        for bracket in stats["brackets"]:
            bracketStats = stats["brackets"][bracket]
            t = bracketStats[name]

            for idx,T in enumerate(rule[-1]):
                if t > T:
                    betas[0][idx] += 1
                if t <= T:
                    betas[1][idx] +=1
                
            
        alphas = [[list() for i in range(len(rule[-1]))],[list() for i in range(len(rule[-1]))]]
        

        for idx,T in enumerate(rule[-1]):
        #for year in range(2013,2020):
            a1 = [0 for i in range(2013,2020)]
            a2 = [0 for i in range(2013,2020)]
            for year in range(2013,2020):
                top100s = stats["top100"][str(year)]
                for bracket in top100s:
                    bracketStats = stats["brackets"][bracket]
                    t = bracketStats[name]

                    if t <= T:
                        a1[year-2013] += 1
                    if t > T:
                        a2[year-2013] +=1
                        
            alphas[0][idx] = ([i/100 for i in a1])
            alphas[1][idx] = ([i/100 for i in a2])

        for idx,T in enumerate(rule[-1]):
            results[name+"#<=#"+str(T)] = [alphas[0][idx],betas[0][idx]/1000000]
            results[name+"#>#"+str(T)] = [alphas[1][idx],betas[1][idx]/1000000]


    with open("alphaBetas.json","w") as f:
        f.write(json.dumps(results))

# bracket stats is rulesV5stats.json
# rules will be a dict of the nicer rules, maps round to rule

def alphasBetaCombinations(bracketStats,rules):
    # iterate thru all possible combinations of 4
    results = dict()

    with open("rulesV5stats.json","r") as f:
        brackets = json.load(f)

    # compute the alphas and beta for that rule set
    # results[rule] = [[alphas],beta]
    # rule is of the form 1 X 2 X 3 X 4


    count = 0
    print(len(rules[1]),len(rules[2]),len(rules[3]), len(rules[4]))
    final = len(rules[1])*len(rules[2])*len(rules[3]) * len(rules[4])
    
    
    for a in rules[1]:
        
        for b in rules[2]:
            for c in rules[3]:
                for d in rules[4]:
                    print(count,final)
                    count+=1
                    
                    ruleSet = [a,b,c,d]
                    name = ""
                    for one in ruleSet:
                        name+= str(str(one[0]) + str(one[1]) + str(one[2]) + " X ")
                    name = name[:-2]

                    print(name)

                    beta = 0
                    for bracket in brackets["brackets"]:
                        stats = brackets["brackets"][bracket]
                        passes = True
                        for rule in ruleSet:
                            t = stats[rule[0]]
                            # the way rules are formatted now from analysis.py, is of the form ()
                            
                            if rule[1] == "<=":
                                if t > rule[2]:
                                    passes = False
                                    break
                            elif rule[1] == ">":
                                if t <= rule[2]:
                                    passes = False
                                    break
                            
                        if not passes:
                            beta+=1
                            
                    alphas = []
                    for year in range(2013,2020):
                        alpha = 0
                        for bracket in brackets["top100"][str(year)]:
                            stats = brackets["brackets"][bracket]
                            passes = True
                            for rule in ruleSet:
                                t = stats[rule[0]]
                                if rule[1] == "<=":
                                    if t > rule[2]:
                                        passes = False
                                        break
                                elif rule[1] == ">":
                                    if t <= rule[2]:
                                        passes = False
                                        break

                            if  passes:
                                alpha+=1
                        alphas.append(alpha/100)
                    # first find beta for 1 mil
                    # for each year find alpha 

                    results[name] = [alphas,beta/1000000]
                    

    with open("alphaBetasCombinations.json","w") as f:
        json.dump(results, f, ensure_ascii=False)

# finds all brackets using lexi, then finds 5 closest rules.
# outputs rule in latex



# alphaBetas contains rules and associated pf
# first apply lm to rules we care about, the sets results from lm resides in topDogs
# actual names is for use in alphasBetaCombinations

# whats needs to be printed potentially is the results from LM and the results from closest



def LMClosest(alphaBetas,rulesToConsider, combinations = False, rounds = False, perfect = False,allAtOnce = False):
    
    with open(alphaBetas,"r") as f:
        rules = json.load(f)
        
    with open(rulesToConsider,"r") as f:
        passed = json.load(f)

    results = dict()
    for i in range(1,5):
        results["$"+str(i)+"$"] = list()

    for rule in rules:

        if not combinations:
            if rule not in passed:
                continue

        if not combinations:
            ruleName = ruleToLatex(rule,combinations)
        else:
            ruleName = rule

        if rounds and not combinations:
            results[ruleName[0]].append( rules[rule][0]+ [rules[rule][1]] + [rule])
        else:
            results["$"+str(1)+"$"].append( rules[rule][0]+ [rules[rule][1]] + [rule])

    # find lm best ones
    topDogs = dict()
    for i in range(1,5):
        topDogs[i] = dict()
    bestCounts = dict()
    needed = 1*2*3*4*5*6*7*8
    for i in range(1,5):
        bests = dict()
        count = 0
        print(count,needed)
        for comb in itertools.permutations([0,1,2,3,4,5,6,7]):
                print(count,needed)
                count+=1
                ordered = sorted(results["$"+str(i)+"$"], key=lambda x: ( x[comb[0]],x[comb[1]], x[comb[2]], x[comb[3]], x[comb[4]], x[comb[5]], x[comb[6]] , x[comb[7]]  ))
                ordered.reverse()
                for j in range(0,1):
                    theBest = ordered[0]
                    if theBest[8] not in bests:
                        bests[theBest[8]] = 1
                    else :
                        bests[theBest[8]] += 1
        bestCounts[i] = bests
        inOrder = {k: v for k, v in sorted(bests.items(), key=lambda item: -item[1])}

        #if combinations:
            #break
        # dont need to see top dogs in tis case

        print("LM results start here -------------------")
        for one in inOrder:
            if not combinations:
                p = rules[one][0] + [rules[one][1]]

                pVec = ""
                for obj in p :
                    pVec += "$"+str(obj)[:4] + "$&"
                pVec = pVec[:-1]
                rule = ruleToLatex(one,combinations)
                counts = inOrder[one]
                print(rule[0]+"&"+rule[1]+"&"+rule[2]+"&"+rule[3]+"&"+"$"+str(counts)+"$ &" +pVec  +"  \\\\")

        topDogs[i] = inOrder
        
        if not rounds:
            print("ends here ---------")
            break # only one i to look at

    actualNames = dict()
    for i in range(1,5):
        actualNames[i] = list()

    for i in range(1,5):
        for rule in topDogs[i]:
            if combinations:
               break
            else:
                entire = rule.split("#")
                ruleName = entire[0].split("|")
                print(entire)
                r = ruleName[0].split(",")[0]
                s = ruleName[1]
                c = ruleName[2]
                e = ruleName[-2]
                bound = int(entire[-1])
                side = entire[-2]
                actualName = (str(r)+",|"+s+"|"+c+"|"+e+"|")
                if not allAtOnce:
                    actualNames[i].append([actualName,side,bound])

        # find close points
        distBests = []
        norm1 = dict()
        for rule in results["$"+str(i)+"$"]:
            rule = rule[8]
            pr =  rules[rule][0]+ [rules[rule][1]] + [rule]
            n1 = 0
            for one in topDogs[i]:
                if not perfect:
                    px = rules[one][0]+[rules[one][1]]
                    n1 += (np.linalg.norm([pr[i] - px[i] for i in range(8)],ord = 2) *  bestCounts[i][one] / pr[7] )
                    #n1 += (np.linalg.norm([pr[i] - px[i] for i in range(8)],ord = 2)  )
                else:
                    px = [1,1,1,1,1,1,1,0.83]
                    #n1 += (np.linalg.norm([pr[i] - px[i] for i in range(8)],ord = 2)  / pr[7] )
                    n1 += (np.linalg.norm([pr[i] - px[i] for i in range(8)],ord = 2)  )

            norm1[rule] = n1

        N1 = {k: v for k, v in sorted(norm1.items(), key=lambda item: item[1])}
        lol = 0
        print("outputting top 5 closest -------------")
        for one in N1:
            lol+=1
            if combinations:
                allRules = one.split(" X ")
                print(allRules)
                for rule in allRules:
                    rule = ruleToLatex(rule,combinations)
                    print(rule[0]+"&"+rule[1]+"&"+rule[2]+"&"+rule[3]+"  \\\\")
                    print(rules[one][0] + [rules[one][1]])
            else:
                p = rules[one][0] + [rules[one][1]]

                pVec = ""
                for obj in p :
                    pVec += "$"+str(obj)[:4] + "$&"
                pVec = pVec[:-1]
                rule = ruleToLatex(one,combinations)
                print(rule[0]+"&"+rule[1]+"&"+rule[2]+"&"+rule[3]+"&" +pVec  +"  \\\\")
            
            distBests.append(one)
            # be careful what lol is
            if lol == 5:
                print("-------------")
                # output 5 closest
                break

        if allAtOnce:
            closest5Counts = [0,0,0,0]
        if not combinations:
            # gotta add to get the combinations
            for rule in list(set(distBests)):
                if rule not in topDogs[i]:
                    entire = rule.split("#")
                    ruleName = entire[0].split("|")
                    r = ruleName[0].split(",")[0]
                    r = int(r)
                    if allAtOnce:
                        if closest5Counts[r-1] == 5:
                            continue
                    #already have 5
                    
                    s = ruleName[1]
                    c = ruleName[2]
                    e = ruleName[-2]
                    bound = int(entire[-1])
                    side = entire[-2]

                    actualName = (str(r)+",|"+s+"|"+c+"|"+e+"|")
                    if allAtOnce:
                        actualNames[r].append([actualName,side,bound])
                        closest5Counts[r-1] += 1
                    else:
                        actualNames[i].append([actualName,side,bound])
                    
        if not round and not allAtOnce:
            break
    
    for one in topDogs:
        print(len(topDogs[one]))

    for one in results:
        print(len(results[one]))

    for one in actualNames:
        print(len(actualNames[one]))

    if (not combinations and rounds) or (not combinations and allAtOnce):
        alphasBetaCombinations(1,actualNames)

        

def ruleToLatex(rule,combinations = False):
    a = rule.split("|")
    r = "$"+a[0][:-1]+"$"
    s = a[1]
    if s == "1,":
        s = "$1$"
    elif s == "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,":
        s = "$All$"
    elif s== "1,2,3,4,":
        s = "$Top$"
    elif s== "5,6,7,8,9,10,":
        s = "$Middle$"
    elif s == "11,12,13,14,15,16,":
        s = "$Bottom$"
    elif s== "1,16,8,9,5,12,4,13,":
        s = "$Upper$"
    elif s == "6,11,3,14,7,10,2,15,":
        s = "$Lower$"
    e = a[-2]+a[-3]
    if "maxSum" in e:
        e2 = e.replace("maxSums","")
        e = "$maxSums_{"+e2[:-1]+"}$"
    else:
        e = "$"+e+"$"

    if not combinations:
        t = a[-1].split("#")
        sign = t[1]
        value = t[2]
        if "<=" in sign:
            sign = "\leq"
        else:
            sign = ">"
        t = "$"+sign+" "+value+"$"
        return (r,s,e,t)
    else:
        t = a[-1]
        if "<=" in t:
            t = t.replace("<=","\leq ")
            t = "$" + t + "$"
        return (r,s,e,t)
        
if __name__ == '__main__':
    '''
    alphaBetas("rulesV51Milstats.json")
    quit()
    
    ruleSet = RULEV5
    with open("sampleF4AScored1Mil35Years.json","r") as f:
            brackets = json.load(f)
    bracketStatCounting(ruleSet,brackets)

    quit()
    #ruleTopResults(RULEV5,v=5)
    #quit()
    '''
    rulesToUse = ruleThresholdDetermine(RULEV5,v=5)

    with open("RULES.json","w") as f:
        f.write(json.dumps(rulesToUse))
    

    
    quit()
    ################### rule follower


    print(ruleProbs("rulesV4.json")[2])
    quit()

    runRulesSaveResults("sampleF4A",RULEV4,4)




    quit()

    ####################
    
    sampleSize = 50000
    nReplications = 25 - 24
    MODEL_TYPES = ['bradley-terry','power','neoPower']
    BBs = [(0,1,None),(1,1,None),(1,4,'sampleE8'),(1,5,'sampleF4A'),(1,5,'sampleF4B'),(1,4,'samplePower8Brute'),(1,4,'samplePower8BruteRandom1'),(1,5,'samplePower4ABrute'),(1,5,'samplePower4BBrute'),(1,5,'samplePower4BBruteRandom1'),(1,5,'samplePower4ARandom1'),(2,5,'sampleNeoPower4ABrute'),(2,5,"sampleNeoPower4BBrute"),(2,4,"sampleNeoPower8Brute")]

    # add the neo powers if they do well
    limit = 100000

    results = dict()
    # file -> year -> (t4 wins,m3 wins) : List of scores scored against that year 
    for one in BBs:
        name = one[2]
        if name is None:
            name = ""
        results[MODEL_TYPES[one[0]]+name] = dict()
        for year in range(2013,2020):
            results[MODEL_TYPES[one[0]]+name][year] = dict()
            for t4 in range(16+1):
                for m3 in range(12+1):
                    results[MODEL_TYPES[one[0]]+name][year][str(t4)+"-"+str(m3)] = list()


    print(results)
    
    for year in range(2013,2020):
        print(year)

        for one in BBs:
            name = one[2]
            if name is None:
                name = ""
                
            print(one)
            filepath = bp.generateFilepath(sampleSize,year = year, model = MODEL_TYPES[one[0]], r = one[1], samplingFnName=one[2], nReplications = 25)
            with open(filepath,"r") as f:
                brackets = json.load(f)
            count = 0
            while count < limit:
                # just pull by rep 
                for rep in range(nReplications):
                    thatRep = brackets["brackets"][rep]
                    for bracket in thatRep:
                        count+=1
                        bracket = bm.stringToVector(bm.hexToString(bracket))
                        theScore = sf.scoreBracket(bracket,year = year)
                        t4m3 = np.asarray(regionUpsetsT4M3(bm.bracketToSeeds(bracket)))
                        t4 =  sum(t4m3[:,0])
                        m3 = sum(t4m3[:,1])
                        results[MODEL_TYPES[one[0]]+name][year][str(t4)+"-"+str(m3)].append(theScore)
                        count+=1
                        if count > limit:
                            break
                    if count > limit :
                        break


                
        with open("BB_filter_stats.json",'w') as f:
            f.write(json.dumps(results))
