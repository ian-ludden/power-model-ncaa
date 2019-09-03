from collections import defaultdict

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
