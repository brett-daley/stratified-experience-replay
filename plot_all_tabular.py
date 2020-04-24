import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os
from glob import glob
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

RESULTS_DIR = 'tabular_results'
PLOTS_DIR = 'tabular_plots'

SEEDS = range(100)
# SHOW_STD = False
SAVE_AS_PDF = False
MAX_ITERATIONS = 50

ENVS = ('gridworld', 'whirlpool')
N_VALUES = (1, 3, 5)
M_VALUES = (True, False)
MAX_SAMPLES = 100
COLORS = ('r', 'g', 'b', 'm')
assert len(COLORS) >= len(N_VALUES)


_print_lock = Lock()

def run_if_not_exists(cmd, filename):
    cmd += f' --iterations={MAX_ITERATIONS}'
    path = os.path.join(RESULTS_DIR, filename)
    with _print_lock:
        print(cmd)
        print('>', path)
        if os.path.exists(path):
            print('(already exists)')
        print(flush=True)
    if not os.path.exists(path):
        with open(path, 'w') as f:
            subprocess.call(cmd.split(' '), stdout=f)


class StatsParser:
    '''Automatically parses results and computes means, standard deviations.
    Previous requests are cached in memory to reduce runtime.'''
    def __init__(self):
        self.cache = {}  # Saves previous requests so we don't have to re-parse them

    def __getitem__(self, pattern):
        if pattern in self.cache:
            return self.cache[pattern]
        stats = self._glob_and_compute_stats(pattern)
        self.cache[pattern] = stats
        return stats

    def _glob_and_compute_stats(self, pattern):
        metrics = ('iteration', 'normalized_samples', 'rms', 'undisc_return', 'disc_return')
        stats = {m: [] for m in metrics}

        # Collect results from all files that match the pattern
        paths = glob(os.path.join(RESULTS_DIR, pattern))
        if not paths:
            raise RuntimeError(f'{pattern} did not match any files in {RESULTS_DIR}')
        for p in paths:
            data = np.loadtxt(p)
            data = data.reshape(-1, len(metrics))
            for i, m in enumerate(metrics):
                stats[m].append(data[:, i])

        # Compute means and standard deviations
        for m in metrics:
            if len(paths) > 1:
                stats[m + '.std'] = np.std(stats[m], axis=0, ddof=1)
            else:
                stats[m + '.std'] = np.zeros_like(stats[m][0])
            stats[m] = np.mean(stats[m], axis=0)
        return stats


def save_plot(name):
    path = os.path.join(PLOTS_DIR, name)
    if SAVE_AS_PDF:
        path += '.pdf'
        plt.savefig(path, format='pdf')
    else:
        path += '.png'
        plt.savefig(path, format='png')
    print('Saved', path)


def fair_samples(n, multibatch):
    return MAX_SAMPLES if multibatch else (n * MAX_SAMPLES)


def make_label(n, multibatch):
    if n == 1:
        return f'Both, n={n}'
    return f"{('n-Strap' if multibatch else 'n-Step')}, n={n}"


_parser = StatsParser()

def generate_plot(env, metric, xlim=MAX_ITERATIONS, title=''):
    plt.figure()
    for m in M_VALUES:
        for n, color in zip(N_VALUES, COLORS):
            if not m and (n == 1):
                # n=1 is identical with/without our method, so only plot it once
                continue
            s = fair_samples(n, m)
            stats = _parser[f'{env}_nstep-{n}_multibatch-{m}_samples-{s}_seed-*']
            plt.plot(stats['iteration'], stats[metric], ('-' if m else '--') + color,
                     label=make_label(n, m))
    plt.title(title)
    plt.legend(loc='best')
    plt.xlim([0, xlim])
    save_plot(name=f'{env}_{metric}')
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Run all experiments
    with ThreadPoolExecutor(max_workers=8) as executor:
        for env in ENVS:
            for n in N_VALUES:
                for m in M_VALUES:
                    for seed in SEEDS:
                        s = fair_samples(n, m)
                        cmd = f'python value_iteration.py {env} --nstep={n} --multibatch={m} --samples={s} --seed={seed}'
                        filename = f'{env}_nstep-{n}_multibatch-{m}_samples-{s}_seed-{seed}'
                        executor.submit(run_if_not_exists, cmd, filename)

    # Generate all plots
    generate_plot('gridworld', 'rms',
                  title='Gridworld - RMS Error', xlim=20)
    generate_plot('gridworld', 'undisc_return',
                  title='Gridworld - Undiscounted Return', xlim=20)

    generate_plot('whirlpool', 'rms',
                  title='Whirlpool - RMS Error')
    generate_plot('whirlpool', 'undisc_return',
                  title='Whirlpool - Undiscounted Return')


if __name__ == '__main__':
    main()
