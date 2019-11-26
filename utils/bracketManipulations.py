from collections import defaultdict
import queue
import copy

######################################################################
# Author: 	Ian Ludden
# Date: 	08 August 2019
# 
# bracketManipulations.py
# 
# A collection of utility functions for manipulating 
# representations of NCAA March Madness tournaments. 
# 
######################################################################

def applyRoundResults(seeds, results):
    """Returns the seeds that move on from a 
       round of the tournament with the given results. 

       Parameters
       ----------
       seeds : list of ints
           A list of seeds that competed in a round within a region, 
           listed from top to bottom in the official bracket format. 
       results : list of ints 
           A list of results, where a 1 (0) indicates the top (bottom) 
           team won. Should be half as long as seeds. 

       Returns
       -------
       newSeeds : list of ints
           A list of the seeds in the next round, i.e., 
           the winners of the given round. Same length as results.
    """
    nGames = len(results)
    # top gets the 1 else bot gets the 1
    newSeeds = [seeds[2 * i] * results[i] + seeds[2 * i + 1] * (1 - results[i]) for i in range(nGames)]
    return newSeeds


def bracketToSeeds(bracket):
    """Given a bracket vector (a list of 63 0s and/or 1s), 
       returns a list of the lists of seeds reaching 
       each round. 

       Parameters
       ----------
       bracket : list of ints
           a list of 63 0s and/or 1s representing the game outcomes in a bracket

       Returns
       -------
       per_round_seeds : defaultdict(list) of lists of ints
           A dictionary of the lists of seeds reaching each round. 
           The 0-index spot is the starting seeds so the round number can 
           be used directly to index into the list. 
    """
    f4_seeds = []
    ncg_seeds = []

    per_round_seeds = defaultdict(list)

    # Apply the bracket results from each round and region
    for region in range(4):
        round_num = 1
        region_vector = bracket[region * 15: (region + 1) * 15]
        seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
        per_round_seeds[0] += seeds

        # Stopping condition is only one seed, the regional champion, remains
        while len(seeds) > 1:
            n_games = int(len(seeds) / 2)
            results = region_vector[:n_games]
            new_seeds = applyRoundResults(seeds, results)
            region_vector = region_vector[n_games:]
            per_round_seeds[round_num] += new_seeds
            seeds = new_seeds
            round_num += 1
        f4_seeds.append(seeds[0])

    last_three_games = bracket[-3:]
    ncg_seeds = applyRoundResults(f4_seeds, last_three_games[:2])
    # nc_seeds should be a list of length 1
    nc_seeds = applyRoundResults(ncg_seeds, [last_three_games[2]])

    per_round_seeds[5] = ncg_seeds
    per_round_seeds[6] = nc_seeds

    return per_round_seeds


def aggregate(brackets):
    """Aggregates the win counts for different seed match-ups in each round.

       Parameters
       ----------
       brackets : list of lists
           A list of 63-integer lists of 0s and 1s representing brackets

       Returns
       -------
       counts : dictionary of dictionaries of integers
           Win counts keyed by round, matchup, and winner
    """
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for bracket in brackets:
        seeds = bracketToSeeds(bracket)
        # iterate thru each round. 1 being the results after first round
        for r in range(1, 7):
            round_seeds = seeds[r]
            for game_id, winner in enumerate(round_seeds):
                firstSeed = seeds[r - 1][game_id * 2]
                secondSeed = seeds[r - 1][game_id * 2 + 1]
                s1 = min(firstSeed, secondSeed)
                s2 = max(firstSeed, secondSeed)
                matchup = (s1, s2)
                counts[r][matchup][winner] += 1
    return counts


def vectorToString(bracketVector):
    """Converts bracket from list of 63 0s/1s to string of 63 0s/1s."""
    return ''.join([str(bracketVector[i]) for i in range(len(bracketVector))])


def stringToVector(bracketString):
    """Converts bracket from string of 0s/1s to list of 0s/1s."""
    return [int(bracketString[i]) for i in range(len(bracketString))]


def stringToHex(bracketString):
    """Converts a bracket/region string (63/15 0s and/or 1s) 
       to a 16-/4-digit hex representation
       by adding a leading 0.
    """
    bracketString = '0' + bracketString
    nHexDigits = len(bracketString) // 4
    hexString = ''
    for i in range(nHexDigits):
        nextFourBits = bracketString[4 * i : 4 * i + 4]
        hexString += '{0:1x}'.format(int(nextFourBits, 2))
        # 0 references the value, 1 is 1 lengthed x (hex)

    return hexString


def hexToString(bracketHex):
    """Converts a bracket/region hex representation 
       to a 63-/15-bit string.
    """
    if len(bracketHex) == 16: # Full bracket
        return bin(int(bracketHex, 16))[2:].zfill(63)
    else: # Just a region
        return bin(int(bracketHex, 16))[2:].zfill(15)


def prettifyRegionVector(regionHex):
    """Returns a more descriptive string for the 
       given 4-digit hex representation of a region vector. 
    """
    regionVector = stringToVector(hexToString(regionHex))
    seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    r1Winners = applyRoundResults(seeds, regionVector[:8])
    r2Winners = applyRoundResults(r1Winners, regionVector[8:12])
    r3Winners = applyRoundResults(r2Winners, regionVector[12:14])
    r4Winner = applyRoundResults(r3Winners, regionVector[14:])
    return '{0} {1} {2} {3}'.format(r1Winners, r2Winners, r3Winners, r4Winner)





# the findBracketWithinR algorithms generates brackets in a manner that encompases the idea of puting two and two together iteratively. 

def binaryCombineToRound(bracketVector):
    roundBits = [list() for i in range(6)]
    seen = 0

    roundCounts = [0] * 6
    
    while seen != 63:
        for one in range(2):
            roundCounts[0]+=1
            roundBits[0].append(bracketVector[seen])
            seen+=1

        for round in range(1,6):
            if roundCounts[round-1] % 2 == 0 and roundCounts[round-1] > 0:
                roundCounts[round-1] = 0

                roundCounts[round]+= 1
                roundBits[round].append(bracketVector[seen])
                seen+=1
                #print(test)
                #print(roundCounts)

    actualVector = []
    for i in roundBits:
        actualVector+= i
    return actualVector


# [32][16][8] . . [1] format , as opposed to the [15][15][15][15][3] 
def roundToRegion(bracketVector):
    actualBracket = [0] * 63
    actualBracket[-3:] = bracketVector[-3:]
    # final 3 games ok
    for region in range(4):
        regionVector = [0] * 15
        for round in range(4):
            games = 2**(3-round)
            
            for game in range(games):
                if round == 3:
                    regionVector[-1] = bracketVector[-7+region]
                else:
                    regionVector[-(2**(4-round)-1)+game] = bracketVector[    (-(2**(6-round)-1))  +  (region*games)   +game]
        actualBracket[15*region:15*(region+1)] = regionVector
            

    return actualBracket

def regionToRound(bracketVector):
    actualBracket = [0]*63
    actualBracket[-3:] = bracketVector[-3:]
    for region in range(4):
        regionVector = bracketVector[region*15:(region+1)*15]
        for round in range(4):

            games = 2**(3-round)
            for game in range(games):
                actualBracket[-(2**(6-round)-1)+region*games+game] = regionVector[-(2**(4-round) -1) + game ] 
    
    return actualBracket


# bracketVector is of round form

# [0]  . .. . [5]
# [g1]        [ncg]
# [g2]
# .
# .
# [g32]
def roundToRoundBits(bracketVector):
    bracketRoundBits = [list() for i in range(6)]
    seen = 0
    for i in range(6):
        for game in range(2**(5-i)):
            bracketRoundBits[i].append(bracketVector[seen])
            seen+=1
    return bracketRoundBits


class triBracket:
    def __init__(self, bit):
        self.top = None # top child
        self.bot = None # bot child
        # if both none we are at end
        self.bit = bit

    def print(self):
        print(self.bit)

class tournamentTriBrackets:

    def __init__(self,roundBits,originalBits):
        self.roundTriBrackets = [list() for i in range(6)] # stores triBrackets
        self.origOrder = queue.Queue()
        self.root = None
        
        for round in range(6):
            for game in range(2**(5-round)):
                self.roundTriBrackets[round].append(triBracket(roundBits[round][game]))

        for round in range(6):
            round = 5 - round
            for game in range(2**(5-round)):
                self.origOrder.put(originalBits[round][game])
                
        for round in range(6):
            for idx,aTriBracket in enumerate(self.roundTriBrackets[round]):
                if round == 0:
                    aTriBracket.top = None
                    aTriBracket.bot = None
                else:
                    aTriBracket.top = self.roundTriBrackets[round-1][2*idx]
                    aTriBracket.bot = self.roundTriBrackets[round-1][2*idx + 1]
                    
        self.root = self.roundTriBrackets[5][0] # ncg game
       # [ [(i).print() for i in self.roundTriBrackets[j]] for j in range(6)]
        
    # does the neccesary swapping , such that the end result tree and roundTriBrackets reflect the correct bracket relative to the given bracket for findBracketsWithinR, returns in round form. 
    def performRotations(self):
        # now do a bfs
        toVisit = queue.Queue()
        toVisit.put(self.root)

        rotated = []
        count = 0
        while not toVisit.empty():
            current = toVisit.get()
            orig = self.origOrder.get()
            if(orig == 1):
                rotated.append(current.bit)
                if(current.top is not None):
                    toVisit.put(current.top)
                if(current.bot is not None):
                    toVisit.put(current.bot)
            else:
                rotated.append(int(not current.bit)) # flip bit and rotate 
                if(current.bot is not None):
                    toVisit.put(current.bot)
                if(current.top is not None):
                    toVisit.put(current.top)

        # round 6 game 1, round 5 game 1, round 5 game 2 . . . .
        #print("rotated",rotated)
        #print(roundToRoundBits(rotated))
        final = []
        seen = 0
        for round in range(6):
            games = 2**(round)
            oneRound = []
            for game in range(games):
                oneRound.append(rotated[seen+game])
            oneRound.reverse()
            final+=oneRound
            seen+=games
        final.reverse()
        #print("final",final)
        #print(roundToRoundBits(final))
        return final
    

        

# bracketVector,orignal both in round format
def pfToOriginal(bracketVector,original):

    bracketRoundBits = roundToRoundBits(bracketVector)

    # original comes in as region format. 
    originalRoundBits = roundToRoundBits(original)
    #print("bracketB",bracketRoundBits)
    #print("origB",originalRoundBits)
    
    bracketTriRep = tournamentTriBrackets(bracketRoundBits,originalRoundBits)

    rotated = bracketTriRep.performRotations()

    return rotated

    

    
        
        
  
    
