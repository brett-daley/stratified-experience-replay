import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
import os
import re
from plot import save, grab
from report_results import parse_all


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='results')
    parser.add_argument('--output_dir', type=str, default='plots')
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    bar_dict = {}

    summary = parse_all(args.input_dir)
    for exp_name, report in summary.items():
        assert '_seed-' not in exp_name

        game = grab('env-()', exp_name)
        y = report['return']

        if game not in bar_dict.keys():
            bar_dict[game] = []

        # Set baseline score as 0th value in dict list for each game
        if "Stratified" not in exp_name:
            uer_score = np.mean(y)
            bar_dict[game].insert(0, uer_score)
        # Set modified DQN score as last value in dict list for each game
        if "Stratified" in exp_name:
            ser_score = np.mean(y)
            bar_dict[game].append(ser_score)
        # Raise error if trying to plot PER
        if "Prioritized" in exp_name:
            raise NotImplementedError('Have not implemented Prioritized ER plotting')

    #  Check that 2 (and only 2) values (one type from new method and the baseline) are being compared per game
    for key in bar_dict.keys():
        assert len(bar_dict[key]) == 2, """
        Invalid number of comparisons. Check input directory to ensure comparison is between baseline
        and one (and only one) modified version.
        """

    bar_titles = []
    relative_pcts = []
    for key in bar_dict.keys():
        # Calculate relative score
        uer_score = bar_dict[key][0]
        ser_score = bar_dict[key][1]
        relative_score_as_pct = (ser_score / uer_score) * 100
        # Store for plotting
        relative_pcts.append(relative_score_as_pct)
        bar_titles.append(key)

    # Plot
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots()
    plt.title("SER Score as % of Uniform ER Score")
    bars = ax.barh(bar_titles, relative_pcts)
    for i, relative_pct in enumerate(relative_pcts):
        ax.text(i, relative_pct, '{:0.2f}%'.format(relative_pct), color='black', ha='center', va='bottom')

    plt.tight_layout(pad=0.2)

    save("bar_plot", args.output_dir, args.pdf)
    plt.close()


if __name__ == '__main__':
    main()
