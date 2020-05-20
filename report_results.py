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
    parser.add_argument('--perf_by_params', action='store_true',
                        help='(flag) Displays mean of all normalized scores for parameter set, across all games. \
                        Default disabled')
    parser.add_argument('--input_dir', type=str, default='results')
    parser.add_argument('--output_dir', type=str, default='plots')
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    game_list = []
    exp_list = []
    score_list = []
    param_list = []

    summary = parse_all(args.input_dir)
    for exp_name, report in summary.items():
        assert '_seed-' not in exp_name

        game = grab('env-()', exp_name)
        y = report['avg_return']
        params = re.findall('n-[0-9]_m-[0-9]', exp_name)[0]

        game_list.append(game)
        exp_list.append(exp_name)
        score_list.append(np.nanmean(y))
        param_list.append(params)

    if args.all_avg_returns is False \
            and args.game_medians is False \
            and args.game_means is False \
            and args.perf_by_params is False:
        print(
            """
        No flag specified. To view results, add --all_avg_returns and/or --game_medians and/or --game_means and/or --perf_by_params
            """
        )

    if args.perf_by_params is True:

        df_base = pd.DataFrame(zip(game_list, score_list), index=param_list, columns=['game', 'avg_return'])
        df_reformat = pd.DataFrame(index=list(set(param_list)), columns=list(set(game_list)))

        for index, row in df_base.iterrows():
            df_reformat.at[index, row['game']] = row['avg_return']

        df3_adjusted = df_reformat / df_reformat.max()
        df3_total = df3_adjusted.copy()
        df3_total['mean'] = df3_adjusted.mean(axis=1)
        df3_total['median'] = df3_adjusted.median(axis=1)

        df3_finalsort = df3_total.sort_values(by=['mean'], ascending=[False])

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            # print(df3_mean)
            print(df3_finalsort)
            df3_finalsort.to_csv(args.output_dir + '/' + 'results_tables.csv')

    if args.all_avg_returns is True or args.game_medians is True or args.game_means is True:
        df = build_df(game_list, score_list, exp_list)
        output_df(df, avg_return_flag=args.all_avg_returns, median_flag=args.game_medians, mean_flag=args.game_means)


if __name__ == '__main__':
    main()
