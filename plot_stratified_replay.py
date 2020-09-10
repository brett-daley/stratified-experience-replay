import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from plot import save


def format_plot():
    ax = plt.gca()
    ax.set_aspect(1.0 / ax.get_data_ratio())
    plt.gcf().set_size_inches(6.4, 6.4)
    plt.tight_layout(pad=0.05)


def plot_histogram():
    plt.figure()

    counts = np.loadtxt('unique_frequency.txt')

    # Plot histogram of the state-action pair frequency
    counts = np.asarray(list(reversed(sorted(counts))))
    plt.bar(1+np.arange(len(counts)), height=counts, width=1.0, color='g')

    plt.xlim([1, counts.sum()])
    plt.xscale('log')
    plt.grid(b=True, which='major', axis='x')

    plt.title("Frequency of Unique State-Action Pairs")
    plt.xlabel('Rank')
    plt.ylabel('Count')

    format_plot()
    save('unique_frequency', directory='.', pdf=False)


def plot_timeseries():
    plt.figure()

    Y = np.loadtxt('n_unique_over_time.txt')
    X = 1 + np.arange(len(Y))

    # Plot the number of unique state-action pairs
    plt.plot(X, Y, label='unique')

    # Plot the replay memory's size
    plt.plot(X, X-1, 'k--', label='total')

    # Shade between the lines
    plt.fill_between(X, 0.0 * Y, Y, color='b', alpha=0.25)
    plt.fill_between(X, Y, X-1, color='r', alpha=0.25)

    plt.xlim([0, X.max()])
    plt.ylim(plt.xlim())
    plt.grid(b=True, which='both', axis='both')

    f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(g))
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(g))

    plt.title('Number of Unique State-Action Pairs')
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    plt.legend(loc='upper left', framealpha=1.0, fontsize=14)

    format_plot()
    save('n_unique_over_time', directory='.', pdf=False)


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 16})
    plot_histogram()
    plot_timeseries()