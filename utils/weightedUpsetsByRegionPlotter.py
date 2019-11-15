import json
import os
import bracketManipulations as bm
import matplotlib.pyplot as plt
import pandas as pd


def upsetsWeightedCalc(year, strVector, regions):
    vector = bm.stringToVector(strVector)
    firstRoundList = bm.bracketToSeeds(vector)[0]
    firstRoundRegions = []
    for x in range (0, len(firstRoundList), 16):
        region = []
        for y in range (0, 16):
            region.append(firstRoundList[x+y])
        firstRoundRegions.append(region)

    secondRoundList = bm.bracketToSeeds(vector)[1]
    secondRoundRegions = []
    for x in range (0, len(secondRoundList), 8):
        region = []
        for y in range (0, 8):
            region.append(secondRoundList[x+y])
        secondRoundRegions.append(region)

    thirdRoundList = bm.bracketToSeeds(vector)[2]
    thirdRoundRegions = []
    for x in range (0, len(thirdRoundList), 4):
        region = []
        for y in range (0, 4):
            region.append(thirdRoundList[x+y])
        thirdRoundRegions.append(region)

    fourthRoundList = bm.bracketToSeeds(vector)[3]
    fourthRoundRegions = []
    for x in range (0, len(fourthRoundList), 2):
        region = []
        for y in range (0, 2):
            region.append(fourthRoundList[x+y])
        fourthRoundRegions.append(region)

    fifthRoundList = bm.bracketToSeeds(vector)[4]
    fifthRoundRegions = []
    for x in range (0, len(fifthRoundList), 1):
        region = []
        for y in range (0, 1):
            region.append(fifthRoundList[x+y])
        fifthRoundRegions.append(region)

    #upsetsList = [0, 0, 0, 0]
    upsetsList = {'Year': year, regions[0]: 0, regions[1]: 0, regions[2]: 0, regions[3]: 0}
    #Python tuples are immutable so using a dictionary or map may be better.

    for i in range (0, len(firstRoundRegions)):
        numUpsetsForRegion = 0

        firstRoundRegion = firstRoundRegions[i]
        secondRoundRegion = secondRoundRegions[i]

        pairs = list(zip(firstRoundRegion[::2], firstRoundRegion[1::2]))

        for j in range (0, len(secondRoundRegion)):
            tupl = pairs[j]
            seedOne = tupl[0]
            seedTwo = tupl[1]
            if (secondRoundRegion[j] == max(seedOne, seedTwo)):
                numUpsetsForRegion += secondRoundRegion[j] - min(seedOne, seedTwo)
        
        #upsetsList[i] += numUpsetsForRegion
        upsetsList[regions[i]] += numUpsetsForRegion

    for i in range (0, len(secondRoundRegions)):
        numUpsetsForRegion = 0

        secondRoundRegion = secondRoundRegions[i]
        thirdRoundRegion = thirdRoundRegions[i]

        pairs = list(zip(secondRoundRegion[::2], secondRoundRegion[1::2]))

        for j in range (0, len(thirdRoundRegion)):
            tupl = pairs[j]
            seedOne = tupl[0]
            seedTwo = tupl[1]
            if (thirdRoundRegion[j] == max(seedOne, seedTwo)):
                numUpsetsForRegion += thirdRoundRegion[j] - max(seedOne, seedTwo)
        
        # upsetsList[i] += numUpsetsForRegion
        upsetsList[regions[i]] += numUpsetsForRegion

    for i in range (0, len(thirdRoundRegions)):
        numUpsetsForRegion = 0

        thirdRoundRegion = thirdRoundRegions[i]
        fourthRoundRegion = fourthRoundRegions[i]

        pairs = list(zip(thirdRoundRegion[::2], thirdRoundRegion[1::2]))

        for j in range (0, len(fourthRoundRegion)):
            tupl = pairs[j]
            seedOne = tupl[0]
            seedTwo = tupl[1]
            if (fourthRoundRegion[j] == max(seedOne, seedTwo)):
                numUpsetsForRegion += fourthRoundRegion[j] - min(seedOne, seedTwo)
        
        # upsetsList[i] += numUpsetsForRegion
        upsetsList[regions[i]] += numUpsetsForRegion

    for i in range (0, len(fourthRoundRegions)):
        numUpsetsForRegion = 0

        fourthRoundRegion = fourthRoundRegions[i]
        fifthRoundRegion = fifthRoundRegions[i]

        pairs = list(zip(fourthRoundRegion[::2], fourthRoundRegion[1::2]))

        for j in range (0, len(fifthRoundRegion)):
            tupl = pairs[j]
            seedOne = tupl[0]
            seedTwo = tupl[1]
            if (fifthRoundRegion[j] == max(seedOne, seedTwo)):
                numUpsetsForRegion += fifthRoundRegion[j] - min(seedOne, seedTwo)
        
        # upsetsList[i] += numUpsetsForRegion
        upsetsList[regions[i]] += numUpsetsForRegion
    return upsetsList


# current_directory = os.path.dirname(__file__)
# parent_directory = os.path.split(current_directory)[0]
# filePath = os.path.join(os.path.split(current_directory)[0], 'allBracketsTTT.json')

filePath = 'allBracketsTTT.json'

brackets = []
with open (filePath) as jsonFile:
    data = json.load(jsonFile)
    for bracketDict in data['brackets']:
        bracket = bracketDict['bracket']
        year = bracket['year']
        vector = bracket['fullvector']
        regionOrder = []
        for region in bracketDict['bracket']['regions']:
            regionOrder.append(region['name'])
        brackets.append((year, vector, regionOrder))

upsetsList = []

for tupl in brackets:
    upsetsList.append(upsetsWeightedCalc(tupl[0], tupl[1], tupl[2]))

df = pd.DataFrame(upsetsList)

worstRegions = df.max(axis=1).tolist()
bestRegions = df.min(axis=1).tolist()

years = []
for year in range(1985, 2020):
    years.append(year)

fig, axs = plt.subplots(2)

axs[0].scatter(years, worstRegions)
axs[0].plot(years, worstRegions)
axs[0].set_title("Max Weighted Upsets")
axs[1].scatter(years, bestRegions)
axs[1].plot(years, bestRegions)
axs[1].set_title("Min Weighted Upsets")

fig.tight_layout()
plt.show()