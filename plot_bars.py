import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # Avoid type 3 fonts
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import numpy as np
import os
import re
from collections import defaultdict
from glob import glob

from plot import save, grab, load_data_from_file
import dqn_utils


def compute_avg_return(key, directory):
    pattern = key + '_seed-*'
    files = glob(os.path.join(directory, pattern))
    if not files:
        print(f'Warning: skipping {key} because no files in {directory} match {pattern}')
        raise ValueError

    values = []
    for f in files:
        data = load_data_from_file(f)
        v = np.mean(data[:, 2])  # 3rd row is episode return
        values.append(v)

    # Average over seeds
    mean = np.mean(values)
    if len(values) > 1:
        std = np.std(values, ddof=1)
    else:
        std = 0.0
    return mean, std


def std_divide(A_mean, A_std, B_mean, B_std):
    """Computes standard deviation of A/B."""
    return np.abs(A_mean / B_mean) * np.sqrt(np.square(A_std / A_mean) + np.square(B_std / B_mean))


def random_baseline(game, n):
    env = dqn_utils.make_env(game, seed=0)
    state = env.reset()
    while True:
        returns = env.get_episode_rewards()
        if len(returns) >= n:
            break

        action = np.random.randint(env.action_space.n)
        state, _, done, _ = env.step(action)
        if done:
            state = env.reset()

    return np.mean(returns), np.std(returns)


def fix_env_name(env):
    try:
        return {
            'beamrider': 'beam_rider',
            'stargunner': 'star_gunner',
            'spaceinvaders': 'space_invaders',
        }[env]
    except KeyError:
        return env


def main():
    np.random.seed(0)

    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='results')
    parser.add_argument('--output_dir', type=str, default='plots')
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Get unique runs (ignoring seeds)
    keys = set()
    for f in os.listdir(args.input_dir):
        k = re.sub('_seed-[0-9]+(.txt)?', '', f)
        keys.add(k)

    # Compute average performance and store in dictionary
    bar_dict = defaultdict(dict)
    for k in sorted(keys):
        mean, std = compute_avg_return(k, args.input_dir)
        # print(k, mean, std)
        env = fix_env_name(grab('env-()', k))
        rmem_type = grab('rmem-()', k)
        bar_dict[env].update({rmem_type: (mean, std)})

    envs_and_scores_and_errors = []
    for env in bar_dict.keys():
        print(env)
        random_mean, _ = random_baseline(env, n=100)
        uniform_mean = bar_dict[env]['ReplayMemory'][0]
        uniform_std = bar_dict[env]['ReplayMemory'][1]
        stratified_mean = bar_dict[env]['StratifiedReplayMemory'][0]
        stratified_std = bar_dict[env]['StratifiedReplayMemory'][1]

        # print('A', stratified_mean, stratified_std)
        # print('B', uniform_mean, uniform_std)
        # print(random_mean)

        # Assumes that the random baseline is a deterministic quantity
        relative_perf = 100.0 * (stratified_mean - random_mean) / (uniform_mean - random_mean)
        std_dev = 100.0 * std_divide(stratified_mean - random_mean, stratified_std,
                                     uniform_mean - random_mean, uniform_std)
        envs_and_scores_and_errors.append((env, relative_perf, std_dev))

    # Plot
    plt.style.use('seaborn-darkgrid')
    plt.figure()
    ax = plt.gca()

    envs_and_scores_and_errors = sorted(envs_and_scores_and_errors, key=lambda x: x[1])
    envs, scores, errors = zip(*envs_and_scores_and_errors)
    bars = ax.barh(envs, scores, zorder=3)
    error_bars = ax.errorbar(scores, envs, xerr=errors, ls='none', color='black', capsize=3, zorder=2)
    for i, s in enumerate(scores):
        ax.text(s * 0.82, i - 0.125, '{:0.2f}%'.format(s), color='white', zorder=4)

    ylim = [-0.75, 10.75]
    plt.ylim(ylim)
    plt.vlines(100., ymin=ylim[0], ymax=ylim[1], linestyles='dashed', color='slategray', zorder=0)
    save('bar_plot', args.output_dir, args.pdf, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
