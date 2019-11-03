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

# repLimit in place if not tryna run all reps
def kCenterClustering(jsonFile,k,repLimit = 25):

    brackets = jsonFile['brackets'] # 25 repetitions

    clusters = list() # stores clusters for each rep
    rs = list() # stores ball radius for each rep's cluster
    
    for rep in range(repLimit):
        curRep = spaceFilterDupePfBall(brackets[rep])
        
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
                    
            centers.add(maxIdx)
            distance[maxIdx] = 0
            newCenterIdx = x
        r = max(distance) # the radius of them balls, 2 X approximation for optimal r given this k
        clusters.append(centers)
        rs.append(r)
        
        

            
def generateFilepath(generatorPath,technique= "partition"):
    filePath = generatorPath[:-5]+"_"+ technique +".json"
    return filePath

    

if __name__ == '__main__':

    if True:
    # apply partitioning to existing genrator spaces. (for now only one year)
        sampleSize = 50000
        nReplications = 25
        #for year in range(2013, 2020):
        for year in range(2013,2014):
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


            
    if False:
        # grab all the compressed spaces. Basically all the files in compressedOutputs
        # to grab files can use genreateFile path to query correct ones

        sampleSize = 50000
        nReplications = 25
        #for year in range(2013, 2020):
        for year in range(2013,2014):
            partitionPaths = []
            for one in GENERATION_TYPES:
                partitionPaths.append(generateFilepath(bp.generateFilepath(sampleSize,year = year, model = MODEL_TYPES[one[0]], r = one[1], samplingFnName=one[2], nReplications = nReplications,folder = "compressedOutputs")))

            aPgs = [set() for i in range(5)] # "s" because there are multiple reps
        # union the compressed spaces, apply the pfBall to it.
            
            for one in partitionPaths:
                with open(one,"r") as f:
                    parition = json.load(f)
                brackets = partition['bracket']
                for rep in range(5):
                    aPgs[rep].update(spaceFilterDupePfBall(brackets[rep]))

            # aPgs now contains no dupes and all within pfBall
                
            

            kCenters = dict()
        # compute a variety of k centers , store those k centers and corresponding r
        # save in compressedOutputs. name it _kcenter.json . . . .
