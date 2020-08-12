import numpy as np
from utils.preprocessForBradleyTerry import preprocess

######################################################################
# Authors:  Nestor Bermudez, Ian Ludden
# Date:     08 August 2019
# 
# fitBradleyTerry.py
# 
# Fits the Bradley-Terry model to historical tournament data and outputs 
# the parameters to be used to estimate win probabilities. 
# 
######################################################################

def fit(limit_year):
    W = np.zeros((16, 16))
    beta = np.random.uniform(0, 1, 16)

    data = preprocess(limit_year)

    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            if i < j:
                row = data[(data['player1'] == 's{}'.format(i + 1)) & (data['player2'] == 's{}'.format(j + 1))]
                if len(row) == 0:
                    continue
                W[i, j] = row['item1wins'].values[0]
            else:
                row = data[(data['player1'] == 's{}'.format(j + 1)) & (data['player2'] == 's{}'.format(i + 1))]
                if len(row) == 0:
                    continue
                W[i, j] = row['item2wins'].values[0]

    def log_likelihood(W, beta):
        ll = 0.
        for i in range(16):
            for j in range(16):
                if W[i, j] == 0:
                    continue
                ll += W[i, j] * np.log(beta[i]) - W[i, j] * np.log(beta[i] + beta[j])
        return ll

    def optimize(W, beta):
        beta_prime = np.zeros(16)
        for i in range(16):
            factor = 0.
            for j in range(16):
                if i == j:
                    continue
                factor += (W[i, j] + W[j, i]) / (beta[i] + beta[j])
            beta_prime[i] = W[i, :].sum() * 1 / factor
        return beta_prime / beta_prime.sum()

    prev = 0
    iteration = 0
    while True:
        beta = optimize(W, beta)
        ll = log_likelihood(W, beta)
        iteration += 1
        if np.abs(prev - ll) < 1e-10:
            break
        prev = ll

    p = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            p[i, j] = beta[i] / (beta[i] + beta[j])

    print(beta)

    return p


if __name__ == '__main__':
    import os
    if not os.path.exists('bradleyTerry'):
        os.makedirs('bradleyTerry')

    # with open('btParams.csv', 'w') as f:
    #     for year in range(2013, 2021):
    #         f.write('{0}\n'.format(year))
    #         p = fit(limit_year=year)
    #         for i in range(16):
    #             for j in range(16):
    #                 if i == j:
    #                     f.write('0.5000,')
    #                 else:
    #                     f.write('{0:.4f},'.format(p[i, j]))
    #             f.write('\n')
    for year in range(2013, 2021):
        p = fit(limit_year=year)