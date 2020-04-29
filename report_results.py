import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
import re
import os
import yaml
from glob import glob
import pandas as pd

METRICS = ('timestep', 'episode', 'avg_return', 'epsilon', 'hours')


def new_report():
    report = dict(seeds=0)
    for m in METRICS:
        report[m] = []
    return report


def get_data_lines(path):
    pattern = re.compile(r'^[0-9]+\s{2}')
    lines = []
    with open(path, 'r') as fh:
        for l in fh.readlines():
            if re.match(pattern, l):
                lines.append(l)
    return lines


def get_config():
    with open('plot_config.yaml', 'r') as fh:
        return yaml.safe_load(fh.read())


def load_data_from_file(path):
    lines = get_data_lines(path)
    data = [np.fromstring(l, dtype=float, sep=' ') for l in lines]
    return np.asarray(data)


def parse_one(directory, key):
    pattern = key + '_seed-*.txt'
    files = glob(os.path.join(directory, pattern))
    if not files:
        print(f'Warning: skipping {key} because no files in {directory} match {pattern}')
        return None

    # {metric0: [seed0, seed1, ...], metric1: [seed0, seed1, ...], ...}
    report = new_report()

    for f in files:
        data = load_data_from_file(f)
        for i, m in enumerate(METRICS):
            try:
                report[m].append(data[:,i])
            except IndexError:
                print(f"Error: metric '{m}' could not be parsed from {f}")
                print( '       (file is probably corrupted)')
                raise
        report['seeds'] += 1

    # Average over seeds
    for m in METRICS:
        data = report.pop(m)

        try:
            mean = np.mean(data, axis=0)
            if report['seeds'] > 1:
                std = np.std(data, axis=0, ddof=1)
            else:
                std = np.zeros_like(mean)
        except ValueError as e:
            print(f'Warning: skipping {key} due to missing data')
            print(f'         {str(e)}')
            return None

        report[m] = mean
        report[m + '.std'] = std

    return report


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='results')
    parser.add_argument('--output-dir', type=str, default='plots')
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    config = get_config()
    game_list = []
    e_list = []
    score_list = []

    for plot_name, experiments in config.items():

        for e in experiments:
            assert '_seed-' not in e

            report = parse_one(args.input_dir, e)
            y = report['avg_return']

            game_list.append(plot_name)
            e_list.append(e)
            score_list.append(np.nanmean(y))

    df = pd.DataFrame(zip(game_list, score_list), index=e_list, columns=['game', 'exp_mean_score'])
    df_sorted = df.sort_values(by=['game', 'exp_mean_score'], ascending=[True, False])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_sorted)


if __name__ == '__main__':
    main()
