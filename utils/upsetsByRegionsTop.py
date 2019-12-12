import json
import os
import bracketManipulations as bm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit

def upsetsCalc(year, strVector, regions):
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

    upsetsList = {'Year': year, regions[0]: 0, regions[1]: 0, regions[2]: 0, regions[3]: 0}
    upsetsListTop = {'Year': year, regions[0]: 0, regions[1]: 0, regions[2]: 0, regions[3]: 0}

    for i in range (0, len(firstRoundRegions)):
        numUpsetsForRegion = 0
        numUpsetsBottom = 0

        firstRoundRegion = firstRoundRegions[i]
        secondRoundRegion = secondRoundRegions[i]

        pairs = list(zip(firstRoundRegion[::2], firstRoundRegion[1::2]))

        for j in range (0, len(secondRoundRegion)):
            tupl = pairs[j]
            seedOne = tupl[0]
            seedTwo = tupl[1]
            if (secondRoundRegion[j] == max(seedOne, seedTwo) and (seedOne != 8 and seedTwo != 9)):
                if (max(seedOne, seedTwo) == 12 or max(seedOne, seedTwo) == 11 or max(seedOne, seedTwo) == 10):
                    numUpsetsBottom += 1
                numUpsetsForRegion += 1
        upsetsList[regions[i]] += numUpsetsForRegion
        upsetsListTop[regions[i]] += numUpsetsBottom

    return upsetsListTop

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
    upsetsList.append(upsetsCalc(tupl[0], tupl[1], tupl[2]))
    # print(upsetsList)

#Data table that with count of upsets for each region for each year
df = pd.DataFrame(upsetsList)

print (df)

worstRegions = df.max(axis=1).tolist()
bestRegions = df.min(axis=1).tolist()

years = []
for year in range(1985, 2020):
    years.append(year)

fig, axs = plt.subplots(2)

# axs[0].scatter(years, worstRegions)
# axs[0].plot(years, worstRegions)
# b, m = polyfit(years, worstRegions, 1)
# # axs[0].plot(years, b + np.asarray(m) * years, '-')
# axs[0].set_title("Max Upsets")
# print(b)
# print(m)

# axs[1].scatter(years, bestRegions)
# axs[1].plot(years, bestRegions)
# b, m = polyfit(years, bestRegions, 1)
# # axs[1].plot(years, b + np.asarray(m) * years, '-')
# axs[1].set_title("Min Upsets")
# print(b)
# print(m)

worstRegions = [round(x) for x in worstRegions]
bestRegions = [round(x) for x in bestRegions]

print (worstRegions)
print (bestRegions)

axs[0].hist(worstRegions, bins = [0,1,2,3])#, [2,3,4,5,6,7,8,9,10], (2,10))
axs[0].set_xticks([0, 1, 2, 3])
axs[0].set_title("Max Upsets")
axs[1].hist(bestRegions,bins = [0,1,2])#, [0,1,2,3,4,5,6], (1,6))
axs[1].set_xticks([0, 1, 2])
axs[1].set_title("Min Upsets")

fig.tight_layout()
plt.show()