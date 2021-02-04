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
        v = np.mean(data[:,2])  # 3rd row is episode return
        values.append(v)

    # Average over seeds
    mean = np.mean(values)
    if len(values) > 1:
        std = np.std(values, ddof=1)
    else:
        std = 0.0
    return mean, std


def random_baseline(game, n):
    env = dqn_utils.make_env(game, seed=0)
    for _ in range(n):
        state = env.reset()
        done = False
        while not done:
            state, _, done, _ = env.step(env.action_space.sample())
    returns = env.get_episode_rewards()
    return np.mean(returns)


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
    for k in keys:
        mean, std = compute_avg_return(k, args.input_dir)
        # print(k, mean, std)
        env = fix_env_name(grab('env-()', k))
        rmem_type = grab('rmem-()', k)
        bar_dict[env].update({rmem_type: (mean, std)})

    envs_and_scores = []
    for env in bar_dict.keys():
        print(env)
        random = random_baseline(env, n=100)
        uniform = bar_dict[env]['ReplayMemory'][0]
        stratified = bar_dict[env]['StratifiedReplayMemory'][0]
        relative_perf = 100 * (stratified - random) / (uniform - random)
        envs_and_scores.append((env, relative_perf))

    # Plot
    plt.style.use('seaborn-darkgrid')
    plt.figure()
    ax = plt.gca()

    envs_and_scores = sorted(envs_and_scores, key=lambda x: x[1])
    envs, scores = zip(*envs_and_scores)
    bars = ax.barh(envs, scores)
    for i, s in enumerate(scores):
        ax.text(s * 0.87, i - 0.125, '{:0.2f}%'.format(s), color='white')

    save('bar_plot', args.output_dir, args.pdf, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
