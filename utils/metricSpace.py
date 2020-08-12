import json
import scoringFunctions as sf
import bracketManipulations as bm
import bracketPools as bp
import fileUtils
import numpy as np

# This file provides functions that serve the purpose of compressing a certain generator's bracket space. How do I get a SMALLEr subset of the bracketspace or a SMALLER set of brackets that reasonable "resemble" that of the original bracket space. 



MODEL_TYPES = ['bradley-terry','power']


GENERATION_TYPES = [(0,1,None),(1,1,None),(1,4,'sampleE8'),(1,5,'sampleF4A'),(1,5,'sampleF4B'),(1,6,'sampleNCG'),(1,4,'samplePower8Brute'),(1,4,'samplePower8BrutePf'),(1,4,'samplePower8BruteRandom1'),(1,5,'samplePower4ABrute'),(1,5,'samplePower4BBrute'),(1,5,'samplePower4BBruteRandom1'),(1,5,'samplePower4ARandom1'),(1,4,'samplePower8BrutePfNot')]

GENERATION_TYPES = GENERATION_TYPES[2:]
# specify center of a r ball. Returns true if HPP(center,toCheck) <= r
# make sure passed in brackets are only first 60, but HPP should be able to account for full 63 bits
def withinBall(center,toCheck,r):
    # default HPP goes up to r4, if not psecified
    
    if(sf.HPP(center,toCheck) <= r ):
        return True
    else:
        return False

# final 4
def withinPfBall(toCheck,r):
    if(sf.HPP(toCheck,[1]*len(toCheck)) <= r):
        return True
    else:
        return False

# filters out dupes and makes sure within that pfBall
# also does conversion
def spaceFilterDupePfBall(brackets,r = 1200):
    brackets = list(set([(bm.hexToString(i))[:60] for i in brackets]))
    brackets = [bm.stringToVector(i) for i in brackets]
    # only care about first 60(up to final 4), and I need to remove dupes
    brackets = [i for i in brackets if withinPfBall(i,r)]
    return brackets
    

# json file is a json.load(f) object
# returns a partitioning of the generator's space (set of centers)
def finiteMetricSpacePartition(jsonFile, r,repLimit = 25,upperBound = 5000):
    
    brackets = jsonFile['brackets'] # 25 repetitions


    partitions = list() # stores partitions for each rep
    
    for rep in range(repLimit):
        
        curRep = spaceFilterDupePfBall(brackets[rep])

        sample = np.random.choice([i for i in range(len(curRep))], size = min(upperBound,len(curRep)),replace=False)

        curRep = [curRep[i] for i in sample]
        
        alpha = np.random.uniform(1/2,1)
        R = alpha * r
        size = len(curRep)
        permutation = np.random.permutation([i for i in range(size)])
        
        clusterMembership = list()  # idx -> idx of cluster
        outliers = list() # stores points such that idx == clusterMembership[idx]
        # runtime is O(n^2) * (HPP compute time), n being the upperbound
        
        count = 0
        for x in permutation:
            print([r,rep,repLimit,count,size],end="\r")
            count+=1
            found = False
            for y in permutation:
                if y == x:
                    continue
                if(withinBall(curRep[x],curRep[y],R)):
                    clusterMembership.append(y)
                    found = True
                    break

            if not found:
                clusterMembership.append(x)
                outliers.append(x) # do some analysis on the accuracy of membership in outliers
        partition = [curRep[i] for i in list(set(clusterMembership)) + outliers]
        # outliers are not really outliers per say, just that the cluster they picked happened to be themselves.
        
        partitions.append(partition)
        
    return partitions # contains repLimit partitions


# takes in specified k, returns a size k set representative of the centers
# greedy algorithm for 2 factor approximation for optimal k centers

def kCenterClustering(bracket,k):

    curRep = spaceFilterDupePfBall(bracket)

    # introduce some randomness
    permutation = np.random.permutation([i for i in range(len(curRep))])

    curRep = [curRep[i] for i in permutation]

    size = len(curRep)

    distance = [1930] * size # tracks current distance from each point to centers currently in centers set. Note max distance cannot exceed 1920

    centers = list()
    centers.append(0) # first arbitrary point (idx)
    distance[0] = 0
    newCenterIdx = 0

    for i in range(k-1):
        maxIdx = 0
        maxDist = 0
        for x in range(size):
            if distance[x] == 0:
                continue
            distance[x] = min(distance[x], sf.HPP(curRep[x],curRep[newCenterIdx]))
            if (distance[x] >= maxDist):
                maxIdx = x
                maxDist = distance[x]
        print([i,k-1],end="\r")

        centers.append(maxIdx)
        distance[maxIdx] = 0
        newCenterIdx = x
    r = max(distance) # the radius of them balls, 2 X approximation for optimal r given this k
  

    return centers,r


# give a bracket, return the brackets that are within HPP r (give in 1,2,4,8 form, non trailing zero), HPP here is computed up to upTo round, inclusive of that round. Finishes that round
# given bracket is in vector form

# each bracket's 2^i - 1 bit represenation, B, can undergo a xor operation with another 2^i - 1 bit string, N, such that B xor N = 1^(2^i - 1)

# not advisable to go beyond r = 60, 100 k for 70 and 300k for 80
# r input is on 1,2,4,8 scale
def findBracketsWithinR(bracket,r,upTo = 6):
    brackets = [list() for i in range(r+1)] # each index stores the bracket such that they are index away from correct
 
    # write a [15][15][15][15][3] -> [32][16][8][4][2][1]  to round by round bits
    
    # A_i,k, A's ith round k game. In bit string,
    # possible k is 2^(6-i) for round i, 1 to 6
    # [32][16][8][4][2][1]
    # per 2 is a unit, unit has the (even,odd)

    bitTracking = [list() for i in range(r+1)] # actual bits, each index stores a list of the bit brackets with its corresponding last match, 1 being last one matches
    bitTracking[0] = [[[1],1]]
    bitTracking[1] = [[[0],0]]
    
    rCounts = [0 for i in range(193)]
    rCounts[0] =1
    rCounts[1] =1
    
    matchCounts = [0 for i in range(193)]
    matchCounts[0] = 1
    matchCounts[1] = 0

    for I in range(1,upTo):
        tempRCounts = [0 for i in range(193)]
        tempMatchCounts = [0 for i in range(193)]
        #print(rCounts)
        for idx1,count1 in enumerate(rCounts):
            if(count1 == 0):
                continue
            for idx2, count2 in enumerate(rCounts):
                if(count2 == 0):
                    continue
                tempRCounts[idx1+idx2] += count1 * count2                
                tempMatchCounts[idx1+idx2] += matchCounts[idx1]*count2

                if idx1+idx2 < r+1:
                    for part1 in bitTracking[idx1]:
                        for part2 in bitTracking[idx2]:
                            brackets[idx1+idx2].append([part1[0]+part2[0],part1[1]])
                
                #print(count1*count2)

        rCounts = [0 for i in range(193)]
        matchCounts = [0 for i in range(193)]

        bitTracking = [list() for i in range(r+1)]

        for idx,count in enumerate(tempRCounts):
            if idx+2**I > 192:
                continue
            rCounts[idx] += tempMatchCounts[idx]
            matchCounts[idx] += tempMatchCounts[idx]
            rCounts[idx+2**I] += (tempMatchCounts[idx] + 2*(tempRCounts[idx]-tempMatchCounts[idx]))

            
            if idx < r+1:
                for preC in brackets[idx]:
                    if(preC[1] == 1):
                        # match case, stay here
                        bitTracking[idx].append([preC[0]+[1],1])
                    else:
                        # unmatch case despite picking top
                        if(idx+2**I < r+1):
                            bitTracking[idx+2**I].append([preC[0]+[1],0])

                    
                    if(idx+2**I < r+1):
                        bitTracking[idx + 2**I].append([preC[0]+[0],0])

        brackets = [list() for i in range(r+1)] 
    #print(rCounts)
    #print(bitTracking)

    # generate xor bit from original
    pfBin = bm.vectorToString([1]*63)
    rotation = "{0:063b}".format(int(bm.vectorToString(bracket),2) ^ int(pfBin,2))

    
    finalBrackets = dict() # hex representation to distance
    for r,allR in enumerate(bitTracking):
        for idx,aBracket in enumerate(allR):
            #print(idx)
            # have to reformat aBracket, still in [32][16]. . .[1] format 

            #actual = bm.stringToVector("{0:063b}".format(int(bm.vectorToString(aBracket[0]),2) ^ int(rotation,2)))
            actual = aBracket[0]
            
            #print("before",actual)
            actual = bm.binaryCombineToRound(actual)
            #print("after",actual)
            # I need to rotate
            # actual = aBracket xor Rotation, it is off, I need to compare the bits for actual and 1
           
            actual = bm.pfToOriginal(actual,bm.regionToRound(bracket))
            actual = bm.roundToRegion((actual))
          
            
            
            finalBrackets[ bm.stringToHex(bm.vectorToString(bm.roundToRegion(actual)))  ] = r
            #finalBrackets[bm.vectorToString((actual))] = r

            # verifies if correct
            # if incorrect will out put wrong
            if sf.scoreBracket(actual,actualResultsVector=bracket)[0] != 1920 - (10*r):
                #print("score",sf.scoreBracket(actual,actualResultsVector=bracket)[0])
                #print("actual",bm.roundToRoundBits(bm.regionToRound(actual)))
                #print("bracket",bm.roundToRoundBits(bm.regionToRound(bracket)))
                print("wrong")
                break


    
        
    return  finalBrackets

            
def generateFilepath(generatorPath,technique= "partition"):
    filePath = generatorPath[:-5]+"_"+ technique +".json"
    return filePath

    

if __name__ == '__main__':

    #test = [1]*8+[0]+[1]*54   # first region's second round first game swapped 
    #findBracketsWithinR(test,3)

    #quit()
    with open("../allBracketsTTT.json") as f:
        data = json.load(f)
    region_dict = {"West":0,"East":1,"South":2,"Midwest":3,"Southeast":2,"Southwest":3}
    # seems that southeast appears when south is missing, southwest when midwest is missing
    # each region has 15 games, 12 games before elite 8
# if fixing elite 8, 2 teams are fixed in each region. This means the results of 1,2,3 * 2 = 6 games are fixed, leaving 8 free
# if looking at entire year, then 48 games



    all_results = dict()
    # "year" "round" "game"


    # bracketToSeeds may be helpful
    brackets = (data["brackets"])
    for one in brackets:
        bracket = one["bracket"]
        results = bracket['fullvector']
        results_list = [int(i) for i in results]
        # [15][15][15][15][2][1]
        year = bracket['year']
        regions = bracket["regions"]

        region_idx = [region_dict[one['name']] for one in regions]
        all_results[year] = (bm.bracketToSeeds(results_list),results_list)
        # all results stores seeds reaching each round, and 63 bit string list
        
    hist35 = [all_results[i][1] for i in all_results]
    hist35p = hist35[-7:]
    his35m = hist35[:-7]

    hpp60hist35 = dict()
    for idx,year in enumerate(hist35):
        print(idx)
        hpp60hist35[1985+idx] = findBracketsWithinR(year,6)
        
    with open("hpp60hist35.json","w") as f:
        f.write(json.dumps(hpp60hist35))


        
    quit()



    
    if False:
        roundBits = []
        start = 0
        for round in range(6):
            roundBits.append([start+i for i in range(2**(5-round))])
            start+= 2**(5-round)

    # print(roundBits)

        roundIdx = [0] * 6

        roundCounts = [0] * 6
        test = []
        while len(test) != 63:
            for one in range(2):
                roundCounts[0]+=1
                test.append(roundBits[0][roundIdx[0]])
                roundIdx[0]+=1

            for round in range(1,6):
                if roundCounts[round-1] % 2 == 0 and roundCounts[round-1] > 0:
                    roundCounts[round-1] = 0
                    #print(roundCounts[round-1])
                    #print(roundCounts[round-1])
                    #print(round)
                    roundCounts[round]+= 1
                    test.append(roundBits[round][roundIdx[round]])
                    roundIdx[round] += 1
                    #print(test)
                    #print(roundCounts)


        #print(roundBits)

        print(test)
        testActual = []
        for i in roundBits:
            testActual+=i
        print(bm.binaryCombineToRound(test)) # should look like [i for i in range(63)]
        print(bm.roundToRegion(testActual))

        quit()
        

    testing = findBracketsWithinR([0]*63,6)

    
    quit()
    sum = 0
    for bracket in testing:
        r = testing[bracket]
        
        actual =  sf.scoreBracket(bm.stringToVector(bm.hexToString((bracket))),actualResultsVector = [1]*63)
        if 1920-actual[0] != 10*r:
            sum += 1
        print(r*10,1920-actual[0])
    print(sum)
    
    quit()
    if False:
    # apply partitioning to existing genrator spaces. (for now only one year)
        sampleSize = 50000
        nReplications = 25
        #for year in range(2013, 2020):
        for year in range(2013,2014):

            # for now since only one year this works, once I scale this I will have to change the logic
            filepaths = []
            saveToPaths = []
            for one in GENERATION_TYPES:
                filepaths.append(bp.generateFilepath(sampleSize,year = year, model = MODEL_TYPES[one[0]], r = one[1], samplingFnName=one[2], nReplications = nReplications))
                saveToPaths.append(generateFilepath(bp.generateFilepath(sampleSize,year = year, model = MODEL_TYPES[one[0]], r = one[1], samplingFnName=one[2], nReplications = nReplications,folder = "compressedOutputs")))

        for one in range(len(filepaths)):
            # load in the file first
            with open(filepaths[one],"r") as f:
                generator = json.load(f)
            print("opened 1")
            compressedDict = dict()
            for r in range(100,500,100):
                print(filepaths[one])
                compressed = finiteMetricSpacePartition(jsonFile = generator,r=r,repLimit = 5)
                for rep in range(5):

                    #for j in range(8):
                    #bitString = "{:03b}".format(j)
                    bitString = "000" # I am storing the pre 60. Later on I will brute force all final 8 results
                    compressedDict[str(rep)+"_"+str(r)] = [bm.stringToHex(bm.vectorToString(i)+bitString) for i in compressed[rep]]

            with open(saveToPaths[one],"w") as outputFile:
                outputFile.write(json.dumps(compressedDict))
            print("saved 1")


            
    if True:
        # grab all the compressed spaces. Basically all the files in compressedOutputs
        # to grab files can use genreateFile path to query correct ones

        sampleSize = 50000
        nReplications = 25
        #for year in range(2013, 2020):
        for year in range(2013,2014):
            partitionPaths = []
            for one in GENERATION_TYPES:
                partitionPaths.append(generateFilepath(bp.generateFilepath(sampleSize,year = year, model = MODEL_TYPES[one[0]], r = one[1], samplingFnName=one[2], nReplications = nReplications,folder = "compressedOutputs")))

            aPgs = dict() # (rep_r)  -> union space
            # union the compressed spaces, apply the pfBall to it.

            for rep in range(5):
                for r in range(100,500,100):
                    for one in partitionPaths:
                        if str(rep)+"_"+str(r) not in aPgs:
                            aPgs[str(rep)+"_"+str(r)] = set()
                        
                        with open(one,"r") as f:
                            partition = json.load(f)
                        aPgs[str(rep)+"_"+str(r)].update(partition[str(rep)+"_"+str(r)])
                        print([rep,r],end="\r")

            # this step can wait, cus running is pretty quick
            #for key in aPgs:
             #   aPgs[key] = spaceFilterDupePfBall(aPgs[key])
              #  print(key)
            #with open("../Outputs/compressedOutputs/2013_compressed_5_reps.json",'w') as f:
             #   f.write(json.dumps(aPgs))

            # aPgs now contains no dupes and all within pfBall.
 
            
            print("now finding centers")
            kCenters = dict() # for each (rep,r), run the various 

            
            
            for key in aPgs:
                print(key)
                for pct in np.arange(0.33,1,0.2):
                    k = int(len(aPgs[key]) * pct)
                    kCs,mDs = kCenterClustering(aPgs[key],k)

                    kCenters[key+"_"+str(pct)] = {"kCenters":kCs,"maxDists":mDs}
                    print([key,pct])

                    with open("../Outputs/compressedOutputs/2013_kCenters_5_reps.json",'w') as f:
                        f.write(json.dumps(kCenters))
                    break
            
                    
        # compute a variety of k centers , store those k centers and corresponding r
        # save in compressedOutputs. name it _kcenter.json . . . .
