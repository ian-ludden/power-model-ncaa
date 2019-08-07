__author__ = "Nestor Bermudez"
__license__ = "MIT"
__version__ = "1.0.0"
__email__ = "nab6@illinois.edu"
__status__ = "Development"


import matplotlib.pyplot as plt
from triplets.priors.PriorDistributions import read_data
from scipy.stats import entropy
from scipy.stats import wasserstein_distance


FIRST_ROUND_SELECTOR = [0, 7, 5, 3, 2, 4, 6, 1]
LEGENDS = ['1 vs 16', '2 vs 15', '3 vs 14', '4 vs 13', '5 vs 12', '6 vs 11', '7 vs 10', '8 vs 9']


def main(_):
    for year in range(2012, 2019):
        _, pooled = read_data('TTT', year + 1)
        data = pooled.values[:, FIRST_ROUND_SELECTOR].astype(int) == 1
        counts = data.sum(axis=0)
        fig, ax = plt.subplots()
        ax.bar(LEGENDS, counts)
        plt.title('Number of upsets in the first round until {}'.format(year))
        plt.xlabel('Game')
        plt.ylabel('Count')

        for i in ax.patches:
            ax.text(i.get_x() + 0.27, i.get_height() + .5, str(i.get_height()), fontsize=9, color='black')

        plt.savefig('upsetDist/until_{}.png'.format(year))

    for year_a in range(2012, 2019):
        _, pooled_a = read_data('TTT', year_a + 1)
        p_a = pooled_a.values[:, FIRST_ROUND_SELECTOR].astype(int) == 0
        p_a = 1. * p_a.sum(axis=0) / p_a.sum()
        for year_b in range(year_a + 1, 2019):
            _, pooled_b = read_data('TTT', year_b + 1)
            p_b = pooled_b.values[:, FIRST_ROUND_SELECTOR].astype(int) == 0
            p_b = 1. * p_b.sum(axis=0) / p_b.sum()

            kl = entropy(p_a, p_b)
            print('Entropy between {} and {}: {}'.format(year_a, year_b, kl))

            wd = wasserstein_distance(p_a, p_b)
            print('Wasserstein between {} and {}: {}'.format(year_a, year_b, wd))
        print('=' * 120)


if __name__ == '__main__':
    main(0)
