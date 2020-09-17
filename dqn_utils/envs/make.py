import gym
from gym.wrappers import Monitor
from datetime import datetime
import os

from dqn_utils.envs import atari_env, toytext_env


def make_env(env_name, seed):
    if env_name in atari_env.ALL_GAMES:
        env = atari_env.make(env_name)
    elif env_name in toytext_env.ALL_ENVS:
        env = toytext_env.make(env_name)
    else:
        env = gym.make(env_name)
        env = monitor(env, env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    return env


def monitor(env, name):
    monitor_dir = os.path.join('monitor', name, datetime.now().strftime(r'%Y.%m.%d_%H.%M.%S.%f'))
    return Monitor(env, directory=monitor_dir, video_callable=lambda e: False)
