import gym
import numpy as np

from dqn_utils import envs


def make(env_name):
    if 'FrozenLake' in env_name:
        if '8x8' in env_name:
            env = gym.make(env_name, map_name='8x8', is_slippery=True)
        else:
            env = gym.make(env_name, is_slippery=True)
    else:
        env = gym.make(env_name)

    env = envs.make.monitor(env, env_name)

    if 'FrozenLake' in env_name:
        env = FrozenLakeObsWrapper(env)
    elif 'Taxi' in env_name:
        env = TaxiObsWrapper(env)
    return env


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
    [taxi_x, taxi_y, passenger_loc_x, passenger_loc_y, destination_x, destination_y, passenger_in_car]
    where passenger_in_car = 0 if the passenger IS NOT in the taxi, and passenger_in_car = 1 if pass. IS in the taxi.
    Passenger location is continuously updated when passenger_in_car==1.'''
    def __init__(self, env):
        super().__init__(env)
        self.size = (7,)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.size, dtype=np.float32)

    def observation(self, observation):
        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(observation)

        normed_taxi_x = taxi_col / 4
        normed_taxi_y = taxi_row / 4

        if pass_idx == 0:
            pass_pickup_x = 0.
            pass_pickup_y = 0.
            pass_in_car = 0.
        elif pass_idx == 1:
            pass_pickup_x = 1.
            pass_pickup_y = 0.
            pass_in_car = 0.
        elif pass_idx == 2:
            pass_pickup_x = 0.
            pass_pickup_y = 1.
            pass_in_car = 0.
        elif pass_idx == 3:
            pass_pickup_x = 0.75
            pass_pickup_y = 1.
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

        # Passenger location is taxi's location if in the car, else it's pickup location
        if pass_in_car:
            pass_loc_x, pass_loc_y = normed_taxi_x, normed_taxi_y
        else:
            pass_loc_x, pass_loc_y = pass_pickup_x, pass_pickup_y

        normed_obs = np.empty(shape=self.size)
        normed_obs[0] = normed_taxi_x
        normed_obs[1] = normed_taxi_y
        normed_obs[2] = pass_loc_x
        normed_obs[3] = pass_loc_y
        normed_obs[4] = dest_x
        normed_obs[5] = dest_y
        normed_obs[6] = pass_in_car

        return normed_obs


ALL_ENVS = (
    'Blackjack-v0',
    'FrozenLake-v0',
    'FrozenLake8x8-v0',
    'GuessingGame-v0',
    'HotterColder-v0',
    'NChain-v0',
    'Roulette-v0',
    'Taxi-v3',
)
