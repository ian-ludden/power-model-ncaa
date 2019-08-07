__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import json
import numpy as np
import os


def calculate_tail_p(score, num_trials, year, batch_number, models_path, summary_path):
    with open(models_path) as f:
        models = json.load(f)
    n_models = len(models['models'])
    p = []
    for model in models['models']:
        name = model['modelName']
        if model.get('ref') is not None:
            summary_path = 'Summary_{}_models'.format(model['ref'])

        if not os.path.exists('{}/stats'.format(summary_path)):
            raise Exception(
                'Model stats must be calculated before. Please execute scoresHistogram.py')
        with open('{}/stats/{}/{}_batch{}.json'.format(summary_path, year, name, batch_number)) as f:
            stats = json.load(f)
        percentiles = [0] + stats['percentiles'][1:] + [stats['max'], 1920]
        for i in range(100, -1, -1):
            if score > percentiles[i] and score <= percentiles[i+1]:
                p.append((100 - i) / 100.)
                break
    p_all = 1. / n_models * np.sum(p)
    return p_all


if __name__ == '__main__':
    import sys

    num_trials = int(sys.argv[1])
    num_batches = int(sys.argv[2])
    models_path = sys.argv[3]
    summary_path = sys.argv[4]
    score = int(sys.argv[5])

    if not os.path.exists('{}/stats'.format(summary_path)):
        print('Model stats must be calculated before. Please execute scoresHistogram.py')
        exit(1)

    for year in range(2013, 2019):
        for batch_number in range(num_batches):
            p = calculate_tail_p(score, num_trials, year, batch_number, models_path, summary_path)
            print(year, batch_number, p)
