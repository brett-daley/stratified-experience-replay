import numpy as np
import random

from dqn_utils.random_dict import RandomDict


class PickyReplayMemory:
    def __init__(self, env, batch_size=32, capacity=1_000_000):
        self.is_using_picky_memory = True

        self.batch_size = batch_size
        self.capacity = capacity
        self.size_now = 0
        self.pointer = 0

        self.observations = np.empty(shape=[capacity, *env.observation_space.shape],
                                     dtype=env.observation_space.dtype)
        self.actions = np.empty(capacity, dtype=np.int32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)

        self.pair_to_index_dict = RandomDict()

    def save(self, observation, action, reward, done, new_observation):
        p = self.pointer

        # If we're full, we need to delete the oldest entry first
        if self._is_full():
            old_pair = make_pair(self.observations[p], self.actions[p])
            old_index_list = self.pair_to_index_dict[old_pair]
            old_index_list.pop(0)
            if not old_index_list:
                del self.pair_to_index_dict[old_pair]

        # Save the transition
        self.observations[p], self.actions[p], self.rewards[p], self.dones[p] = observation, action, reward, done

        # Update the index for the new entry
        new_pair = make_pair(observation, action)
        if new_pair not in self.pair_to_index_dict:
            # TODO: We might want to use a deque instead of a list
            self.pair_to_index_dict[new_pair] = []
        index_list = self.pair_to_index_dict[new_pair]
        index_list.append(p)

        # Increment size and pointer
        self.size_now = min(self.size_now + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self.capacity

    def _is_full(self):
        return self.size_now == self.capacity

    def sample(self, discount, nsteps):
        if nsteps != 1:
            raise NotImplementedError('PickyReplayMemory supports only 1-step returns')
            # TODO: Can we generalize this to n-step returns?

        # Sample indices for the minibatch
        indices = []
        for _ in range(self.batch_size):
            index_list = self.pair_to_index_dict.random_value()
            i = random.choice(index_list)
            indices.append(i)
        i = np.asarray(indices)
        i = (self.pointer + i) % self.size_now

        # Get the transitions
        observations = self.observations[i]
        actions = self.actions[i]
        rewards = self.rewards[i]
        done_mask = (1.0 - self.dones[i])
        bootstrap_observations = self.observations[(i + nsteps) % self.size_now]

        return observations, actions, rewards, done_mask, bootstrap_observations


def make_pair(observation, action):
    return (observation.tostring(), action)


class ReplayMemory:
    def __init__(self, env, batch_size=32, capacity=1_000_000):
        self.is_using_picky_memory = False
        self.batch_size = batch_size
        self.capacity = capacity
        self.size_now = 0
        self.pointer = 0

        self.observations = np.empty(shape=[capacity, *env.observation_space.shape],
                                     dtype=env.observation_space.dtype)
        self.actions = np.empty(capacity, dtype=np.int32)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)

    def save(self, observation, action, reward, done):
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
