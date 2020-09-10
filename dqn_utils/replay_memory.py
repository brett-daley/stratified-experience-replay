import numpy as np
import random
from collections import deque

from dqn_utils.random_dict import RandomDict


class StratifiedReplayMemory:
    def __init__(self, env, batch_size=32, capacity=1_000_000):
        self.batch_size = batch_size
        self.capacity = capacity
        self.size_now = 0
        self.pointer = 0

        self.observations = np.empty(shape=[capacity, *env.observation_space.shape],
                                     dtype=env.observation_space.dtype)
        self.actions = np.empty(capacity, dtype=np.int32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)

        self.pair_to_indices_dict = RandomDict()

    def save(self, observation, action, reward, done, new_observation):
        p = self.pointer

        # self._update_histogram_data()

        # If we're full, we need to delete the oldest entry first
        if self._is_full():
            old_pair = make_pair(self.observations[p], self.actions[p])
            old_index_deque = self.pair_to_indices_dict[old_pair]
            old_index_deque.popleft()
            if not old_index_deque:
                self.pair_to_indices_dict.pop(old_pair)

        # Save the transition
        self.observations[p], self.actions[p], self.rewards[p], self.dones[p] = observation, action, reward, done

        # Update the index for the new entry
        new_pair = make_pair(observation, action)
        if new_pair not in self.pair_to_indices_dict:
            self.pair_to_indices_dict[new_pair] = deque()
        self.pair_to_indices_dict[new_pair].append(p)

        # Increment size and pointer
        self.size_now = min(self.size_now + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self.capacity

    def _is_full(self):
        return self.size_now == self.capacity

    def sample(self, discount, nsteps):
        if nsteps != 1:
            raise NotImplementedError('StratifiedReplayMemory supports only 1-step returns')
            # TODO: Can we generalize this to n-step returns?

        # Sample indices for the minibatch
        i = np.asarray([self._sample_index(nsteps) for _ in range(self.batch_size)])

        # Get the transitions
        observations = self.observations[i]
        actions = self.actions[i]
        rewards = self.rewards[i]
        done_mask = (1.0 - self.dones[i])
        bootstrap_observations = self.observations[(i + nsteps) % self.size_now]

        return observations, actions, rewards, done_mask, bootstrap_observations

    def _sample_index(self, nsteps):
        index_deque = self.pair_to_indices_dict.random_value()
        x = random.choice(index_deque)
        # Make sure the sampled index has room to bootstrap
        if (x - self.pointer) % self.size_now >= self.capacity - nsteps:
            # It's too close to the pointer; recurse and try again
            return self._sample_index(nsteps)
        return x

    def _update_histogram_data(self):
        if not self._is_full():
            if not hasattr(self, '_n_unique_over_time'):
                self._n_unique_over_time = []
            self._n_unique_over_time.append( len(self.pair_to_indices_dict.values) )

        if self._is_full():
            # Save data for unique vs time
            np.savetxt('n_unique_over_time.txt', self._n_unique_over_time, fmt='%d')

            # Save data for histogram
            count_list = []
            for _, index_deque in self.pair_to_indices_dict.values:
                count_list.append( len(index_deque) )
            np.savetxt('unique_frequency.txt', count_list, fmt='%d')
            import sys; sys.exit()


def make_pair(observation, action):
    return (observation.tostring(), action)


class ReplayMemory:
    def __init__(self, env, batch_size=32, capacity=1_000_000):
        self.batch_size = batch_size
        self.capacity = capacity
        self.size_now = 0
        self.pointer = 0

        self.observations = np.empty(shape=[capacity, *env.observation_space.shape],
                                     dtype=env.observation_space.dtype)
        self.actions = np.empty(capacity, dtype=np.int32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)

    def save(self, observation, action, reward, done, new_observation):
        p = self.pointer
        self.observations[p], self.actions[p], self.rewards[p], self.dones[p] = observation, action, reward, done
        self.size_now = min(self.size_now + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self.capacity

    def sample(self, discount, nsteps):
        i = np.random.randint(self.size_now - nsteps, size=self.batch_size)
        i = (self.pointer + i) % self.size_now

        observations = self.observations[i]
        actions = self.actions[i]
        dones = self.dones[i]
        bootstrap_observations = self.observations[(i + nsteps) % self.size_now]

        for k in range(nsteps):
            if k == 0:
                nstep_rewards = self.rewards[i]
                done_mask = (1.0 - self.dones[i])
            else:
                x = (i+k) % self.size_now
                nstep_rewards += done_mask * pow(discount, k) * self.rewards[x]
                done_mask *= (1.0 - self.dones[x])

        return observations, actions, nstep_rewards, done_mask, bootstrap_observations
