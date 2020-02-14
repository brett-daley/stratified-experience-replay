import gym
from gym.wrappers import AtariPreprocessing
import numpy as np


def make(env_name):
    env = gym.make(env_name)
    env = AtariPreprocessing(env, terminal_on_life_loss=True, scale_obs=True, grayscale_obs=False)
    env = ClippedRewardWrapper(env)
    return env


class ClippedRewardWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)
