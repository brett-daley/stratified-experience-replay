import gym
from gym.envs.atari.atari_env import AtariEnv
import numpy as np
from collections import deque
import cv2

from dqn_utils import envs

def make(game, size=84, grayscale=True, history_len=4):
    env = AtariEnv(game, frameskip=4, obs_type='image')
    env = envs.make.monitor(env, game)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetWrapper(env)
    env = NoopResetWrapper(env)
    env = EpisodicLifeWrapper(env)
    env = ClippedRewardWrapper(env)
    env = PreprocessedImageWrapper(env, size, grayscale)
    if history_len > 1:
        env = HistoryWrapper(env, history_len)
    return env

# How we implement the FrozenLakeObsWrapper from further below, in addition to other useful wrappers
def make_frozenlake(game):
    if '8x8' in game:
        env = gym.make(game, map_name='8x8', is_slippery=True)
    else:
        env = gym.make(game, is_slippery=True)
    env = envs.make.monitor(env, game)
    env = FrozenLakeObsWrapper(env)
    return env

def make_taxi(game):
    env = gym.make(game)
    env = envs.make.monitor(env, game)
    env = TaxiObsWrapper(env)
    return env

class ClippedRewardWrapper(gym.RewardWrapper):
    '''Clips rewards to be in {-1, 0, +1} based on their signs.'''
    def reward(self, reward):
        return np.sign(reward)

class EpisodicLifeWrapper(gym.Wrapper):
    '''Signals done when a life is lost, but only resets when the game ends.'''
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        self.observation, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # We lost a life, but force a reset only if it's not game over.
            # Otherwise, the environment just handles it automatically.
            done = True
        self.lives = lives
        return self.observation, reward, done, info

    def reset(self):
        if self.was_real_done:
            self.observation = self.env.reset()
        self.lives = self.env.unwrapped.ale.lives()
        return self.observation

class FireResetWrapper(gym.Wrapper):
    '''Take action on reset for environments that are fixed until firing.'''
    def __init__(self, env):
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        super().__init__(env)

    def reset(self):
        self.env.reset()
        observation, _, _, _ = self.step(1)
        return observation

class HistoryWrapper(gym.Wrapper):
    '''Stacks the previous `history_len` observations along their last axis.
    Pads observations with zeros at the beginning of an episode.'''
    def __init__(self, env, history_len=4):
        assert history_len > 1
        super().__init__(env)
        self.history_len = history_len
        self.deque = deque(maxlen=history_len)

        self.shape = self.observation_space.shape
        self.dtype = self.observation_space.dtype
        self.observation_space.shape = (*self.shape[:-1], history_len * self.shape[-1])

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.deque.append(observation)
        return self._history(), reward, done, info

    def reset(self):
        observation = self.env.reset()
        self._clear()
        self.deque.append(observation)
        return self._history()

    def _history(self):
        return np.concatenate(list(self.deque), axis=-1)

    def _clear(self):
        for _ in range(self.history_len):
            self.deque.append(np.zeros(self.shape, dtype=self.dtype))

class NoopResetWrapper(gym.Wrapper):
    '''Sample initial states by taking a random number of no-ops on reset.
    The number is sampled uniformly from [0, `noop_max`].'''
    def __init__(self, env, noop_max=30):
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self):
        observation = self.env.reset()
        n = np.random.randint(self.noop_max + 1)
        for _ in range(n):
            observation, _, _, _ = self.step(0)
        return observation

class PreprocessedImageWrapper(gym.ObservationWrapper):
    '''Resizes image observations and optionally converts them to grayscale.'''
    def __init__(self, env, size=84, grayscale=True):
        super().__init__(env)
        self.size = size
        self.grayscale = grayscale
        self.shape = (size, size, 1 if grayscale else 3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    def observation(self, observation):
        if self.grayscale:
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        observation = cv2.resize(observation, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return observation.reshape(self.shape).astype(np.uint8)

class FrozenLakeObsWrapper(gym.ObservationWrapper):
    '''Converts FrozenLake observation (an integer) to a normalized coordinate in [0.,0.] - [1.,1.]
    For example, position 14 in FrozenLake 4x4 corresponds to position [x,y] = [2,3] (in the 3rd column and 4th row),
    which normalizes to [0.66666667, 1.].
    Compatible with 4x4 and 8x8 FrozenLake.'''
    def __init__(self, env, size=(2,)):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=size, dtype=np.float32)

    def observation(self, observation):
        row_range = self.env.desc.shape[0]
        col_range = self.env.desc.shape[1]

        row_coordinate = observation // row_range
        col_coordinate = observation % col_range

        normalized_row_coord = row_coordinate / (row_range - 1)
        normalized_col_coord = col_coordinate / (col_range - 1)

        normed_coordinate = np.empty(shape=self.size)
        normed_coordinate[0] = normalized_col_coord
        normed_coordinate[1] = normalized_row_coord
        normed_coordinate = normed_coordinate.astype(np.float32)

        # Rough check to ensure coordinate was indeed normalized
        assert np.linalg.norm(normed_coordinate) < 1.42, "Check norming; norm is greater than that of [[1],[1]] matrix"

        return normed_coordinate

class TaxiObsWrapper(gym.ObservationWrapper):
    '''Converts Taxi-v3 observation (an int up to 500) into a normalized 7-element vector:
    [taxi_x, taxi_y, passenger_pickup_x, passenger_pickup_y, destination_x, destination_y, passenger_in_car]
    where passenger_in_car = 0 if the passenger IS NOT in the taxi, and passenger_in_car = 1 if pass. IS in the taxi.
    NOTE: passnger_pickup_x and passnger_pickup_y correspond only to the pickup location, and do not change if the
    passenger is in the taxi. Of course, they do update if the passenger is dropped off at a new spot,
    since that is the new pickup location.'''
    def __init__(self, env, size=(7,)):
        super().__init__(env)
        self.size = size
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=size, dtype=np.float32)
        self.pass_pickup_x = None
        self.pass_pickup_y = None

    def observation(self, observation):
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(observation)

        normed_taxi_x = taxi_col / 4
        normed_taxi_y = taxi_row / 4

        if pass_idx == 0:
            self.pass_pickup_x = 0.
            self.pass_pickup_y = 0.
            pass_in_car = 0.
        elif pass_idx == 1:
            self.pass_pickup_x = 1.
            self.pass_pickup_y = 0.
            pass_in_car = 0.
        elif pass_idx == 2:
            self.pass_pickup_x = 0.
            self.pass_pickup_y = 1.
            pass_in_car = 0.
        elif pass_idx == 3:
            self.pass_pickup_x = 0.75
            self.pass_pickup_y = 1.
            pass_in_car = 0.
        elif pass_idx == 4:
            pass_in_car = 1.
        else:
            raise ValueError(f"Passenger's index {pass_idx} is invalid")

        if dest_idx == 0:
            dest_x = 0.
            dest_y = 0.
        elif dest_idx == 1:
            dest_x = 1.
            dest_y = 0.
        elif dest_idx == 2:
            dest_x = 0.
            dest_y = 1.
        elif dest_idx == 3:
            dest_x = 0.75
            dest_y = 1.
        else:
            raise ValueError(f"Destination's index {dest_idx} is invalid")

        normed_obs = np.empty(shape=self.size)
        normed_obs[0] = normed_taxi_x
        normed_obs[1] = normed_taxi_y
        normed_obs[2] = self.pass_pickup_x
        normed_obs[3] = self.pass_pickup_y
        normed_obs[4] = dest_x
        normed_obs[5] = dest_y
        normed_obs[6] = pass_in_car

        return normed_obs


ALL_GAMES = (
    # Classic 57 from ALE
    'alien',
    'amidar',
    'assault',
    'asterix',
    'asteroids',
    'atlantis',
    'bank_heist',
    'battle_zone',
    'beam_rider',
    'berzerk',
    'bowling',
    'boxing',
    'breakout',
    'centipede',
    'chopper_command',
    'crazy_climber',
    # 'defender',
    'demon_attack',
    'double_dunk',
    'enduro',
    'fishing_derby',
    'freeway',
    'frostbite',
    'gopher',
    'gravitar',
    'hero',
    'ice_hockey',
    'jamesbond',
    'kangaroo',
    'krull',
    'kung_fu_master',
    'montezuma_revenge',
    'ms_pacman',
    'name_this_game',
    'phoenix',
    'pitfall',
    'pong',
    'private_eye',
    'qbert',
    'riverraid',
    'road_runner',
    'robotank',
    'seaquest',
    'skiing',
    'solaris',
    'space_invaders',
    'star_gunner',
    # 'surround',
    'tennis',
    'time_pilot',
    'tutankham',
    'up_n_down',
    'venture',
    'video_pinball',
    'wizard_of_wor',
    'yars_revenge',
    'zaxxon',

    # Extra games in OpenAI Gym
    'adventure',
    'air_raid',
    'carnival',
    'elevator_action',
    'journey_escape',
    'kaboom',
    'pooyan',
)
