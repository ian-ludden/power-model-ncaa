__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import numpy as np
from utils.preprocessForBradleyTerry import preprocess


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
        # print('iter={}, ll={}'.format(iteration, ll))
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
    np.set_printoptions(linewidth=300)
    print(p)
    return p


if __name__ == '__main__':
    import os
    if not os.path.exists('bradleyTerry'):
        os.makedirs('bradleyTerry')

    for year in range(2020, 2021):
        p = fit(limit_year=year)
        with open('bradleyTerry/probs-{}.csv'.format(year), 'w') as f:
            f.write('"","component","player1","player2","prob1wins","prob2wins"' + '\n')
            for i in range(16):
                for j in range(16):
                    f.write('"1","full_dataset","s{}","s{}",{},{}'.format(
                        i + 1, j + 1, p[i, j], 1 - p[i, j]
                    ) + '\n')
