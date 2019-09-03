from .bracketManipulations import applyRoundResults

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