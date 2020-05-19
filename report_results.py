import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
import re
import os
import yaml
from glob import glob
import pandas as pd

from plot import grab, parse_one


def parse_all(directory):
    keys = set()
    for f in os.listdir(directory):
        k = re.sub('_seed-[0-9]+.txt', '', f)
        keys.add(k)

    summary = {}
    for k in keys:
        report = parse_one(directory, k)
        if report is not None:
            summary.update({k: report})
    return summary


def build_df(game_list, score_list, experiment_list):
    df = pd.DataFrame(zip(game_list, score_list), index=experiment_list, columns=['game', 'avg_return'])
    return df


def output_df(dataframe, avg_return_flag=False, median_flag=False, mean_flag=False):
    if avg_return_flag is False and median_flag is False and mean_flag is False:
        print_yellow(
            """
        No flag specified. To view results, add --all_avg_returns and/or --game_medians and/or --game_means
            """
        )

    if avg_return_flag:
        df_sorted = dataframe.sort_values(by=['game', 'avg_return'], ascending=[True, False])
        with pd.option_context('display.max_rows', None):
            print(df_sorted)

    if median_flag:
        target = pd.concat([
            dataframe['game'],
            dataframe.groupby('game').transform(lambda x: (x / x.max()))
        ], axis=1)
        target_med = target.groupby('game').median()
        target_med.columns = ['median normalized score']
        print(target_med)

    if mean_flag:
        target = pd.concat([
            dataframe['game'],
            dataframe.groupby('game').transform(lambda x: (x / x.max()))
        ], axis=1)
        target_mean = target.groupby('game').mean()
        target_mean.columns = ['mean normalized score']
        print(target_mean)


def main():
    parser = ArgumentParser()
    parser.add_argument('--all_avg_returns', action='store_true',
                        help='(flag) Displays average return for each results file. Default disabled.')
    parser.add_argument('--game_medians', action='store_true',
                        help='(flag) Displays median of all normalized scores for a given game, across all games. \
                        Default disabled')
    parser.add_argument('--game_means', action='store_true',
                        help='(flag) Displays mean of all normalized scores for a given game, across all games. \
                                           Default disabled')
    parser.add_argument('--input_dir', type=str, default='results')
    parser.add_argument('--output-dir', type=str, default='plots')
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    game_list = []
    exp_list = []
    score_list = []

    summary = parse_all(args.input_dir)
    for exp_name, report in summary.items():
        assert '_seed-' not in exp_name

        game = grab('env-()', exp_name)
        y = report['avg_return']

        game_list.append(game)
        exp_list.append(exp_name)
        score_list.append(np.nanmean(y))

    df = build_df(game_list, score_list, exp_list)

    output_df(df, avg_return_flag=args.all_avg_returns, median_flag=args.game_medians, mean_flag=args.game_means)


if __name__ == '__main__':
    main()
