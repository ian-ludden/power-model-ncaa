from bracketManipulations import applyRoundResults
from heapq import heappush, heappop

######################################################################
# Author: 	Ian Ludden
# Date: 	14 August 2019
# 
# scoringFunctions.py
# 
# Utility functions for scoring NCAA tournament brackets.
# 
######################################################################
# Historical brackets in TTT format
# [15 bits Region 0][15 bits Region 1][...][...][2 bits F4][1 bit NCG]
# Each region: [8 bits Round 1][4 bits Round 2][2 bits Round 3][1 bit Round 4 (E8)]
historicalBrackets = {
        '2019': '100111011000100100111011000110100011011100111101101111100001010', 
        '2018': '001001110111110101111110001110101011011100111111101111110101010', 
        '2017': '111101110001000111101111010111101101111001110110111011000101000', 
        '2016': '101001111100100110101011000100101110111111111100101001011100111', 
        '2015': '111111111101111111110011010101111101110001000111100111110101000', 
        '2014': '100101011011111110111111001000110111111010100111100110010101001', 
        '2013': '110111111100101100010010010101111101101000011100111111000011111'
}

# Minimum scores from ESPN Leaderboard (Top 100)
espnCutoffs = {
        '2019': 1730, 
        '2018': 1550, 
        '2017': 1650, 
        '2016': 1630, 
        '2015': 1760, 
        '2014': 1520, 
        '2013': 1590
}

# Maximum scores from pick favorite pool
pickFavoriteScore = {
        '2019': 1240, 
        '2018': 1130, 
        '2017': 1460, 
        '2016': 870, 
        '2015': 1530, 
        '2014': 680, 
        '2013': 1120
}

# takes in vectors
def scoreBracket(bracketVector, actualResultsVector=None, year=2020, isPickFavorite=False):
        """Scores the given bracket vector according to the 
        ESPN Bracket Challenge scoring system. The isPickFavorite
        flag indicates whether the bracket being scored is from the
        Pick Favorite model, in which case we assume that it correctly
        guesses the Final Four and National Championship outcomes.
        Round score subtotals, with only indices 1-6 used
        as actual subtotals. The 0th element is the overall total.
        """
        if actualResultsVector is None:
                actualResultsString = historicalBrackets[str(year)]
                actualResultsVector = [int(actualResultsString[i]) for i in range(63)]

        roundScores = [0, 0, 0, 0, 0, 0, 0]

        regionWinners = []
        actualRegionWinners = []

        # Compute Rounds 1-4 scores
        for region in range(4):
                start = 15 * region
                end = start + 8
                regionVector = bracketVector[start:end]
                regionResultsVector = actualResultsVector[start:end]

                seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
                actualSeeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

                for r in range(1, 5):
                        seeds = applyRoundResults(seeds, regionVector)
                        actualSeeds = applyRoundResults(actualSeeds, regionResultsVector)

                        matches = [i for i, j in zip(seeds, actualSeeds) if i == j]
                        roundScores[r] += 10 * (2 ** (r-1)) * len(matches)

                        start = end
                        end += int(len(seeds) / 2)
                        regionVector = bracketVector[start:end]
                        regionResultsVector = actualResultsVector[start:end]

                regionWinners.append(seeds[0])
                actualRegionWinners.append(actualSeeds[0])

        # Compute Rounds 5-6 scores
        finalFourVector = bracketVector[-3:]
        actualFinalFourVector = actualResultsVector[-3:]

        if isPickFavorite:
                finalFourVector = actualFinalFourVector

        isCorrectFirstSemifinal = (finalFourVector[0] == actualFinalFourVector[0]) and ((finalFourVector[0] == 1 and (regionWinners[0] == actualRegionWinners[0])) or (finalFourVector[0] == 0 and (regionWinners[1] == actualRegionWinners[1])))
        if isCorrectFirstSemifinal:
                roundScores[5] += 160

        isCorrectSecondSemifinal = (finalFourVector[1] == actualFinalFourVector[1]) and ((finalFourVector[1] == 1 and (regionWinners[2] == actualRegionWinners[2])) or (finalFourVector[1] == 0 and (regionWinners[3] == actualRegionWinners[3])))

        if isCorrectSecondSemifinal:
                roundScores[5] += 160

        isCorrectChampion = (finalFourVector[2] == actualFinalFourVector[2]) and ((finalFourVector[2] == 1 and isCorrectFirstSemifinal) or (finalFourVector[2] == 0 and isCorrectSecondSemifinal))
        if isCorrectChampion:
                roundScores[6] += 320

        roundScores[0] = sum(roundScores)
        return roundScores



def HPP(bracketVector,actualResultsVector,upTo = 4,computeType = "-+"):
        # bascially scoreBracket, but allows best case scenario up to a round
        # can take in both 60 bit and 63 bit inputs    

        maxR = 6 # index 1, so maxR indicates which r is last, while r <= maxR
        maxR = upTo
        roundScores = [0] * (maxR+1)

      
        maxRMatches = [0] * 4

        regionFinalists = []
        actualRegionFinalists = []
        
        for region in range(4):

                
                start = 15 * region
                end = start + 8
                regionVector = bracketVector[start:end]
                regionResultsVector = actualResultsVector[start:end]

              

                seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
                actualSeeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

                for r in range(1, min(5,maxR+1) ):
                        seeds = applyRoundResults(seeds, regionVector)
                        actualSeeds = applyRoundResults(actualSeeds, regionResultsVector)

                        #print(seeds)
                        #print(actualSeeds)
                        matches = [i for i, j in zip(seeds, actualSeeds) if i == j]
                        roundScores[r] += 10 * (2 ** (r-1)) * len(matches)
                        if r == maxR:
                                maxRMatches[region] = len(matches)
                        start = end
                        end += int(len(seeds) / 2)
                        regionVector = bracketVector[start:end]
                        regionResultsVector = actualResultsVector[start:end]
                regionFinalists.extend(seeds)
                actualRegionFinalists.extend(actualSeeds)

        #print(regionFinalists)
        #print(actualRegionFinalists)
        #print("--=--")
        HPP = sum(roundScores)

        if computeType == "-":
                return HPP

        # now compute best case scneraio.
        # if maxR = 4 : [R1,R2] [R3,R4] => [a,b]
        # if maxR = 3 : [R1] [R2] [R3] [R4] => maxR = 4


        #HPP_4+
        # for each pair if one possible then you get it
        # best case scenario given that you have the last 3 games left.
        
        
        possible = []
        roundMatches = []
        for i in range(6 - maxR):
                
                if i == 0:
                        for j in range(int(len(regionFinalists)/2)):
                                matched = False
                                if(regionFinalists[2*j] == actualRegionFinalists[2*j]) or (regionFinalists[2*j+1] == actualRegionFinalists[2*j +1]):
                                        matched = True
                                possible.append(matched)
                        roundMatches.append(len([i for i in possible if i ]))
                else:
                        newPossible = []
                        for j in range(int(len(possible)/2)):
                                matched = False
                                if possible[2*j] or possible[2*j+1]:
                                        matched = True
                                newPossible.append(matched)
                        possible = newPossible
                        roundMatches.append(len([i for i in possible if i]))
        #print(roundMatches)
                        
        #print(HPP)
        HPPplus = 0
        for i in range(len(roundMatches)):
                score = int(320/ (2**(i)) )
                
                matches = roundMatches[-(i+1)]
                #print(score,matches)
                HPPplus +=  (score)*matches

                
        HPP += HPPplus
        if computeType == "+":
                return HPPplus
  

        return 1920 - HPP

def mstNewBracket(adjacencyList, alTable,newBracket,upTo):
        # speed or size issue?
        # returns the new adjacencyList corresponding to new mst, and cost
        # do not modify adjacencyList 
        
        nbHpp = list() # size n, nbHpp[idx] = (HPP(n,idx),n,idx)
        n  = len(adjacencyList)
        newAdjList = [[]] * (n+1)
        # newAdjList[idx] = {(HPP(idx,x),idx,x) all x such that edge (idx,x)} exists in mst

        for idx,oldBracket in enumerate(alTable):
                nbHpp.append((HPP(oldBracket,newBracket,upTo),n,idx)) 
                adjacencyList[idx].append((nbHpp[idx][0],idx,n)) # update old graph
        pq = []
        seen = set()
        seen.add(0)
        for edge in adjacencyList[0]:
                heappush(pq,edge)

        
        while len(seen) < n+1:
                nextOne = heappop(pq)
                #(z,x,y)
                if nextOne[2] in seen:
                        continue
                else:
                        seen.add(nextOne[2])
                        
                        newAdjList[nextOne[1]].append((nextOne[0],nextOne[1],nextOne[2]))
                        newAdjList[nextOne[2]].append((nextOne[0],nextOne[2],nextOne[1]))
                        if nextOne[2] == n:
                                # old adjacencyList does not have the new bracket
                                for edge in nbHpp:
                                        heappush(pq,edge)
                        else:
                                for edge in adjacencyList[nextOne[2]]:
                                        heappush(pq,edge)

        totalCost = 0
        for index in adjacencyList:
                for edge in index:
                        totalCost+= edge[0]
                        
        return (newAdjList, totalCost / 2)








if __name__ == '__main__':
        # testing hpp

        
        quit()
