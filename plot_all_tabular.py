import numpy as np
import subprocess
import matplotlib.pyplot as plt
import os
from glob import glob

RESULTS_DIR = 'tabular_results'
PLOTS_DIR = 'tabular_plots'

MAX_N = 5
MAX_ITERATIONS = 10
MAX_SEEDS = 10


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


def glob_and_compute_stats(pattern):
    metrics = ('iteration', 'normalized_samples', 'rms', 'undisc_return', 'disc_return')
    stats = {m: [] for m in metrics}

    # Collect results from all files that match the pattern
    paths = glob(os.path.join(RESULTS_DIR, pattern))
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


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Run all experiments
    for env in ['gridworld']:
        for n in range(1, MAX_N):
            for m in [False, True]:
                if (n == 1) and m:
                    continue  # Multibatch does nothing when n=1

                for s in range(MAX_SEEDS):
                    cmd = f'python value_iteration.py {env} --nstep={n} --multibatch={m} --samples=10 --seed={s}'
                    filename = f'{env}_nstep-{n}_multibatch-{m}_samples-10_seed-{s}'
                    run_if_not_exists(cmd, filename)

    # Test the stats
    stats = glob_and_compute_stats('gridworld_nstep-1_multibatch-False_samples-10_seed-*')
    print(stats)


if __name__ == '__main__':
    main()
