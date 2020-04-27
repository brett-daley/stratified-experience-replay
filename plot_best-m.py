import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
import re
import os
import yaml
from glob import glob

METRICS = ('timestep', 'episode', 'avg_return', 'epsilon', 'hours')


def new_report():
    report = dict(seeds=0)
    for m in METRICS:
        report[m] = []
    return report


def grab(pattern, string):
    assert '()' in pattern
    pattern = pattern.replace('()', '(.*?)(?=_|$)')
    return re.search(pattern, string).group(1)


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


def save(name, directory, pdf):
    path = os.path.join(directory, name)
    if pdf:
        path += '.pdf'
        plt.savefig(path, format='pdf')
    else:
        path += '.png'
        plt.savefig(path, format='png')
    print(f'Saved plot as {path}', flush=True)


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='results')
    parser.add_argument('--output-dir', type=str, default='plots')
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    easy_save = lambda name: save(name, args.output_dir, args.pdf)

    config = get_config()

    for plot_name, experiments in config.items():
        fig = plt.figure(figsize=(17, 7))
        ax = plt.subplot(111)
        y_best = -np.inf
        y_best_label = None
        y_best_color = None
        new_n_flag = False
        old_n = None

        for e in experiments:
            assert '_seed-' not in e

            n_val = re.search('n-(\d)', e)
            found_n = n_val.group(1)

            if old_n is not None and old_n != found_n:
                new_n_flag = True
            old_n = found_n

            report = parse_one(args.input_dir, e)
            x = report['timestep']
            y = report['avg_return']
            color = 'red' if 'n-1' in e \
                else 'green' if 'n-3' in e \
                else 'blue' if 'n-5' in e \
                else 'magenta' if 'n-7' in e \
                else 'cyan'
            x, y = map(np.array, [x, y])

            # Determine which m-trace to plot
            if 'True' in e:
                ax.plot(x, y, label=e, color=color, linestyle='-')
            elif 'False' in e:
                if new_n_flag:
                    ax.plot(x, y_best, label="best-m_"+y_best_label, color=y_best_color, linestyle=':')
                    y_best = -np.inf
                    y_best_label = None
                    y_best_color = None
                    new_n_flag = False
                if np.mean(y) > np.mean(y_best):
                    y_best = y
                    y_best_label = e
                    y_best_color = color

        ax.plot(x, y_best, label="best-m_"+y_best_label, color=y_best_color, linestyle=':')

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        easy_save(plot_name+"_best-m")
        plt.close()


if __name__ == '__main__':
    main()
