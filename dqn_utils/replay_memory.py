import numpy as np
import random
from collections import deque

from dqn_utils.random_dict import RandomDict


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

    def _is_full(self):
        return self.size_now == self.capacity

    def sample(self, discount, nsteps, train_frac):
        # Sample indices for the minibatch
        i = np.asarray([self._sample_index(nsteps) for _ in range(self.batch_size)])

        observations = self.observations[i]
        actions = self.actions[i]
        dones = self.dones[i]
        bootstrap_observations = self.observations[(i + nsteps) % self.size_now]

        # Compute n-step rewards and get n-step bootstraps
        for k in range(nsteps):
            if k == 0:
                nstep_rewards = self.rewards[i]
                done_mask = (1.0 - self.dones[i])
            else:
                x = (i+k) % self.size_now
                nstep_rewards += done_mask * pow(discount, k) * self.rewards[x]
                done_mask *= (1.0 - self.dones[x])

        weights = np.ones_like(nstep_rewards)
        return (observations, actions, nstep_rewards, done_mask, bootstrap_observations, weights), i

    def _sample_index(self, nsteps):
        x = np.random.randint(self.size_now - nsteps)
        return (self.pointer + x) % self.size_now


class StratifiedReplayMemory(ReplayMemory):
    def __init__(self, env, batch_size=32, capacity=1_000_000):
        super().__init__(env, batch_size, capacity)
        self.pair_to_indices_dict = RandomDict()
        self.t = 0

    def save(self, observation, action, reward, done, new_observation):
        self.t += 1

        p = self.pointer

        # If we're full, we need to delete the oldest entry first
        if self._is_full():
            old_pair = make_pair(self.observations[p], self.actions[p])
            old_index_deque = self.pair_to_indices_dict[old_pair]
            old_index_deque.popleft()
            if not old_index_deque:
                self.pair_to_indices_dict.pop(old_pair)

        # Update the index for the new entry
        new_pair = make_pair(observation, action)
        if new_pair not in self.pair_to_indices_dict:
            self.pair_to_indices_dict[new_pair] = deque()
        self.pair_to_indices_dict[new_pair].append(p)

        # Save the transition
        super().save(observation, action, reward, done, new_observation)

    def sample(self, discount, nsteps, train_frac):
        if nsteps != 1:
            raise NotImplementedError('StratifiedReplayMemory supports only 1-step returns')
        return super().sample(discount, nsteps, train_frac)

    def _sample_index(self, nsteps):
        index_deque = self.pair_to_indices_dict.random_value()
        x = random.choice(index_deque)
        # Make sure the sampled index has room to bootstrap
        if (x - self.pointer) % self.size_now >= self.capacity - nsteps:
            # It's too close to the pointer; recurse and try again
            return self._sample_index(nsteps)
        return x


def make_pair(observation, action):
    return (hash(observation.tostring()), action)
