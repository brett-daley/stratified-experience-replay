import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from plot import save


env_name = 'Stargunner'  # Change this line to generate a different plot
env_name_lower = env_name.lower().replace(' ', '')

PRE = 250_000  # Pre-population
C = 1_000_000  # Capacity


def format_plot(width_factor=1.0):
    ax = plt.gca()
    ax.set_aspect(1.0 / ax.get_data_ratio())
    plt.gcf().set_size_inches(width_factor * 6.4, 6.4)
    plt.tight_layout(pad=0.05)


def plot_histogram():
    plt.figure()

    counts = np.loadtxt(f'{env_name_lower}_unique_frequency.txt')

    # Plot histogram of the state-action pair frequency
    counts = np.asarray(list(reversed(sorted(counts))))
    ranks = 1+np.arange(len(counts))
    # Plot any values that occurred more than once (repeated values)
    plt.bar(ranks[counts > 1], counts[counts > 1], bottom=1, width=1.0, color='r', alpha=0.5)
    # Plot all values that occured at least once (unique values)
    plt.bar(ranks[counts >= 1], np.ones_like(counts), bottom=0, width=1.0, color='b', alpha=0.5)

    plt.xlim([1, counts.sum()])
    plt.xscale('log')
    plt.grid(b=True, which='major', axis='x')

    plt.title(fr"{env_name}: Frequency of $(s,a)$-Pairs")
    plt.xlabel('Rank')
    plt.ylabel('Count')

    format_plot(2.0)
    save(f'{env_name_lower}_unique_frequency', directory='.', pdf=False)


def plot_timeseries():
    plt.figure()

    Y = np.loadtxt(f'{env_name_lower}_n_unique_over_time.txt')
    X = 1 + np.arange(len(Y)) - PRE  # Shift relative to training start

    # Plot the number of unique state-action pairs
    plt.plot(X, Y, label='unique')

    # Plot the replay memory's size
    plt.plot(X, np.minimum(X-1+PRE, C), 'k--', label='total')

    # Shade between the lines
    plt.fill_between(X, 0.0 * Y, Y, color='b', alpha=0.25)
    plt.fill_between(X, Y, np.minimum(X-1+PRE, C), color='r', alpha=0.25)

    plt.xlim([0, X.max()])
    plt.ylim([0, 1.1 * C])
    plt.grid(b=True, which='both', axis='both')

    f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(g))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(g))

    plt.title(rf'{env_name}: Number of $(s,a)$-Pairs')
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    plt.legend(loc='upper left', framealpha=1.0, fontsize=14)

    format_plot()
    save(f'{env_name_lower}_n_unique_over_time', directory='.', pdf=False)


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 16})
    plot_histogram()
    plot_timeseries()
