import numpy as np
import random
from collections import deque

from dqn_utils.prioritization import PrioritizedReplayBuffer
from dqn_utils.random_dict import RandomDict


class ReplayMemory:
    def __init__(self, env, batch_size=32, capacity=1_000_000, history_len=1):
        self.batch_size = batch_size
        self.capacity = capacity
        self.history_len = history_len
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

        observations = self._make_observation_batch(i)
        actions = self.actions[i]
        dones = self.dones[i]
        bootstrap_observations = self._make_observation_batch(i + nsteps)

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

    def _make_observation_batch(self, indices):
        observations = [self._get_history(i) for i in indices]
        return np.stack(observations)

    def _get_history(self, i):
        history = []
        for j in range(self.history_len):
            # Count backwards: x = i, i-1, i-2, ...
            x = (i - j) % self.capacity
            # Stop if we hit a previous episode
            if (j > 0) and self.dones[x]:
                break
            # Add this observation to the history
            history.append(self.observations[x])
            # Stop if the memory isn't full and we're trying to wrap around
            if not self._is_full() and (x == 0):
                break
            # Stop if the memory is full and this was the oldest experience
            if self._is_full() and (x == self.pointer):
                break

        # If we stopped early, then we need to pad with zeros
        zero_pad = np.zeros_like(self.observations[0])
        while len(history) < self.history_len:
            history.append(zero_pad)

        # Our history is backwards; reverse it, then stack along the last axis
        history = list(reversed(history))
        return np.concatenate(history, axis=-1)


class StratifiedReplayMemory(ReplayMemory):
    def __init__(self, env, batch_size=32, capacity=1_000_000):
        super().__init__(env, batch_size, capacity)
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
            # TODO: Can we generalize this to n-step returns?
        return super().sample(discount, nsteps, train_frac)

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


class PrioritizedReplayMemory:
    def __init__(self, env, batch_size=32, capacity=1_000_000):
        # Just hardcode the prioritization hyperparameters here
        # These are the default from the original paper, and OpenAI uses them too
        self.alpha = 0.6
        self.beta_schedule = lambda train_frac: 0.4 + 0.6 * train_frac
        self.epsilon = 1e-6

        # Now make the buffer
        self.buffer = PrioritizedReplayBuffer(env, capacity, self.alpha)
        self.batch_size = batch_size
        self.capacity = capacity

    def save(self, observation, action, reward, done, new_observation):
        # Note that the argument order changes! This is intentional.
        self.buffer.add(observation, action, reward, new_observation, done)

    def sample(self, discount, nsteps, train_frac):
        if nsteps != 1:
            raise NotImplementedError('PrioritizedReplayMemory supports only 1-step returns')

        beta = self.beta_schedule(train_frac)
        observations, actions, rewards, next_observations,\
            dones, weights, indices = self.buffer.sample(self.batch_size, beta)

        rewards = rewards.astype(np.float32)
        done_mask = 1.0 - dones.astype(np.float32)
        weights = weights.astype(np.float32)

        return (observations, actions, rewards, done_mask, next_observations, weights), indices

    def update_td_errors(self, indices, td_errors):
        new_priorities = np.abs(td_errors) + self.epsilon
        self.buffer.update_priorities(indices, new_priorities)


def make_pair(observation, action):
    return (observation.tostring(), action)
