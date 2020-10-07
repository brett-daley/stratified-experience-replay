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
    pattern = key + '_seed-*'
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


def format_plot():
    ax = plt.gca()
    ax.set_aspect(1.0 / ax.get_data_ratio())
    plt.gcf().set_size_inches(6.4, 6.4)
    plt.tight_layout(pad=0.05)

    plt.legend(loc='best', framealpha=1.0, fontsize=12)
    plt.grid(b=True, which='both', axis='both')


def set_plot_attributes(params):
    if 'title' in params:
        plt.title(params['title'])
    if 'xlabel' in params:
        plt.xlabel(params['xlabel'])
    if 'ylabel' in params:
        plt.ylabel(params['ylabel'])
    if 'xlim' in params:
        plt.xlim(params['xlim'])
    if 'ylim' in params:
        plt.ylim(params['ylim'])
    if ('ylog' in params) and params['ylog']:
        plt.yscale('log')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='results')
    parser.add_argument('--output-dir', type=str, default='plots')
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    config = get_config()

    for plot_group, group_params in config.items():
        for plot_name, plot_params in group_params['plots'].items():
            plt.figure()
            plt.rc('xtick', labelsize=16)
            plt.rc('ytick', labelsize=16)
            plt.rcParams.update({'font.size': 22})
            set_plot_attributes(group_params)

            for e, label, color in zip(plot_params['experiments'],
                                       group_params['labels'],
                                       group_params['colors']):
                set_plot_attributes(plot_params)  # Overrides group parameters
                assert '_seed-' not in e
                assert '.txt' not in e
                report = parse_one(args.input_dir, e)
                xmetric = plot_params['xmetric'] if 'xmetric' in plot_params else group_params['xmetric']
                ymetric = plot_params['ymetric'] if 'ymetric' in plot_params else group_params['ymetric']
                try:
                    x, y, error = map(np.array, [report[xmetric], report[ymetric], report[ymetric + '.std']])
                except:
                    print(report.keys())
                plt.plot(x, y, color, label=label)
                plt.fill_between(x, (y - error), (y + error), color=color, alpha=0.25, linewidth=0)

            format_plot()
            save(plot_name, args.output_dir, args.pdf)
            plt.close()


if __name__ == '__main__':
    main()
