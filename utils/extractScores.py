__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import json


def extract_scores(filepath, output):
    scores = []
    with open(filepath) as f:
        data = json.load(f)
        for bracket in data['brackets']:
            scores.append(bracket['score'][0])
        result = {'scores': scores, 'actualBracket': data['actualBracket']}
    with open(output, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 3:
        modelFilename = sys.argv[3]
    else:
        modelFilename = 'models.json'
    with open(modelFilename, 'r') as modelFile:
        modelsDataJson = modelFile.read().replace('\n', '')

    modelsDict = json.loads(modelsDataJson)
    modelsList = modelsDict['models']

    numTrials = int(sys.argv[1])
    numBatches = int(sys.argv[2])

    for modelDict in modelsList:
        modelName = modelDict['modelName']

        for year in range(2013, 2019):
            for batchNumber in range(numBatches):
                if numTrials < 1000:
                    folderName = 'Experiments/{0}Trials'.format(numTrials)
                else:
                    folderName = 'Experiments/{0}kTrials'.format(int(numTrials / 1000))
                batchFolderName = '{0}/Batch{1:02d}'.format(folderName, batchNumber)
                rawDataFilepath = '{2}/generatedBrackets_{0}_{1}.json'.format(modelName, year, batchFolderName)
                scoresFilepath = '{2}/generatedScores_{0}_{1}.json'.format(modelName, year, batchFolderName)

                extract_scores(rawDataFilepath, scoresFilepath)
