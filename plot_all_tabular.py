import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os
from glob import glob

RESULTS_DIR = 'tabular_results'
PLOTS_DIR = 'tabular_plots'

ENVS = ('gridworld', 'whirlpool')
MAX_N = 10
MAX_ITERATIONS = 20
MAX_SEEDS = 10
# SHOW_STD = False
SAVE_AS_PDF = False


def run_if_not_exists(cmd, filename):
    cmd += f' --iterations={MAX_ITERATIONS}'
    path = os.path.join(RESULTS_DIR, filename)
    print(cmd)
    print('>', path)
    if os.path.exists(path):
        print('(already exists)')
    else:
        with open(path, 'w') as f:
            subprocess.call(cmd.split(' '), stdout=f)
    print(flush=True)


class StatsParser:
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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Run all experiments
    for env in ENVS:
        for n in range(1, MAX_N+1):
            for m in [False, True]:
                for s in range(MAX_SEEDS):
                    cmd = f'python value_iteration.py {env} --nstep={n} --multibatch={m} --samples=10 --seed={s}'
                    filename = f'{env}_nstep-{n}_multibatch-{m}_samples-10_seed-{s}'
                    run_if_not_exists(cmd, filename)

    # This object automatically parses results and computes means, standard devs.
    parser = StatsParser()

    for env in ENVS:
        # Discounted return vs. iteration
        for m in [False, True]:
            plt.figure()
            for n in range(1, 5+1):
                stats = parser[f'{env}_nstep-{n}_multibatch-{m}_samples-10_seed-*']
                plt.plot(stats['iteration'], stats['rms'], label=f'{n}_{m}')
            plt.legend(loc='best')
            plt.xlim([0, 5])
            save_plot(name=f'{env}_rms_multibatch-{m}')
            plt.close()

        # RMS error vs. iteration
        for m in [False, True]:
            plt.figure()
            for n in range(1, 5+1):
                stats = parser[f'{env}_nstep-{n}_multibatch-{m}_samples-10_seed-*']
                plt.plot(stats['iteration'], stats['disc_return'], label=f'{n}_{m}')
            plt.legend(loc='best')
            plt.xlim([0, 5])
            save_plot(name=f'{env}_discreturn_multibatch-{m}')
            plt.close()


if __name__ == '__main__':
    main()
