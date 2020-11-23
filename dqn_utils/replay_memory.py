import numpy as np
import random
from collections import deque
import heapq

from dqn_utils.prioritization import PrioritizedReplayBuffer
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
        self.key_to_indices_dict = RandomDict()
        self.t = 0

    def save(self, observation, action, reward, done, new_observation):
        self.t += 1
        # self._update_histogram_data()

        # If we're full, we need to delete the oldest entry first
        if self._is_full():
            i = self._erase_old_index()
        else:
            i = self.pointer

        self._write_new_index(i, observation, action)

        # Temporarily adjust the pointer so we can write to an arbitrary index
        p = self.pointer
        self.pointer = i
        super().save(observation, action, reward, done, new_observation)
        self.pointer = (p + 1) % self.capacity
        return i

    def _write_new_index(self, index, observation, action):
        key = self._make_key(observation, action)
        if key not in self.key_to_indices_dict:
            self.key_to_indices_dict[key] = deque()
        self.key_to_indices_dict[key].append(index)

    def _erase_old_index(self):
        p = self.pointer
        key = self._make_key(self.observations[p], self.actions[p])
        index_deque = self.key_to_indices_dict[key]
        index_deque.popleft()
        if not index_deque:
            self.key_to_indices_dict.pop(key)
        return p

    def _make_key(self, observation, action):
        return (hash(observation.tostring()), action)

    def sample(self, discount, nsteps, train_frac):
        if nsteps != 1:
            raise NotImplementedError('StratifiedReplayMemory supports only 1-step returns')
            # TODO: Can we generalize this to n-step returns?
        return super().sample(discount, nsteps, train_frac)

    def _sample_index(self, nsteps):
        index_deque = self.key_to_indices_dict.random_value()
        x = random.choice(index_deque)
        # Make sure the sampled index has room to bootstrap
        if (x - self.pointer) % self.size_now >= self.capacity - nsteps:
            # It's too close to the pointer; recurse and try again
            return self._sample_index(nsteps)
        return x

    def _update_histogram_data(self):
        x = int(250_000 + 2_000_000)

        if not hasattr(self, '_n_unique_over_time'):
            self._n_unique_over_time = []
        self._n_unique_over_time.append( len(self.key_to_indices_dict.values) )

        if self.t == x:
            env_name = 'stargunner'  # Edit this to change the output filenames

            # Save data for unique vs time
            np.savetxt('{}_n_unique_over_time.txt'.format(env_name), self._n_unique_over_time, fmt='%d')

            # Save data for histogram
            count_list = []
            for _, index_deque in self.key_to_indices_dict.values:
                count_list.append( len(index_deque) )
            np.savetxt('{}_unique_frequency.txt'.format(env_name), count_list, fmt='%d')
            import sys; sys.exit()


class StatesOnlyStratifiedMemory(StratifiedReplayMemory):
    """Samples over unique states only (no actions), thereby ignoring policy corrections."""

    def _make_key(self, observation, action):
        return hash(observation.tostring())


class AnnealingStratifiedMemory(StratifiedReplayMemory):
    """Probabilistically interpolates between stratified and uniform sampling. The
    probability p is annealed to 0 over the course of training."""

    def sample(self, discount, nsteps, train_frac):
        self.train_frac = train_frac  # HACK: pass train_frac to _sample_index()
        return super().sample(discount, nsteps, train_frac)

    def _sample_index(self, nsteps):
        p = np.random.rand()
        if p < (1.0 - self.train_frac):
            # Sample with stratification
            return super()._sample_index(nsteps)
        else:
            # Sample uniformly
            x = np.random.randint(self.size_now - nsteps)
            return (self.pointer + x) % self.size_now


class ScaledLRMemory(StratifiedReplayMemory):
    """Samples uniformly but divides the effective learning rate by the (s,a) count."""

    def sample(self, discount, nsteps, train_frac):
        """Sample as usual, but then adjust the weights by the frequency count."""
        (observations, actions, nstep_rewards, done_mask,
            bootstrap_observations, weights), i = super().sample(discount, nsteps, train_frac)

        # Get the count for each (s,a) pair and divide the weight by it
        for x, (o, a) in enumerate(zip(observations, actions)):
            key = self._make_key(o, a)
            count = len(self.key_to_indices_dict[key])
            weights[x] /= count

        # Scale the weights so the average is 1
        weights /= weights.mean()

        return (observations, actions, nstep_rewards, done_mask, bootstrap_observations, weights), i

    def _sample_index(self, nsteps):
        """We override the parent method to force sampling to be uniform."""
        x = np.random.randint(self.size_now - nsteps)
        return (self.pointer + x) % self.size_now


class RedundantEjectMemory(StratifiedReplayMemory):
    """Discards the oldest transition from the (s,a) pair with the most transitions.
    If tied, discards the oldest pair's oldest transition."""

    def __init__(self, env, batch_size=32, capacity=1_000_000):
        super().__init__(env, batch_size, capacity)
        self.timesteps = np.empty(capacity, dtype=np.int32)
        self.heap = RedundancyHeap()

    def save(self, observation, action, reward, done, new_observation):
        i = super().save(observation, action, reward, done, new_observation)
        self.timesteps[i] = self.t

    def _erase_old_index(self):
        # Find the (s,a) pair with the most transitions
        key, index_deque = self.heap.pop()

        if len(index_deque) == 1:
            # All (s,a) pairs have 1 transition, so we remove the oldest (approximately)
            i, index_deque = self._get_approximate_oldest()
        else:
            # This (s,a) pair has multiple transitions, so we discard its oldest one
            i = index_deque.popleft()

        if len(index_deque) == 0:
            # Delete this (s,a) pair since it's empty
            self.key_to_indices_dict.pop(key)
        else:
            # This (s,a) pair is not empty, so put it back in the heap
            self.heap.push(key, index_deque)
        return i

    def _get_approximate_oldest(self):
        """Sample-based search for the index of the oldest experience. Returns the index
        of the oldest experience in the minibatch.
        """
        t = float('inf')
        oldest_index = None
        oldest_deque = None
        for _ in range(100):
            # Sample an (s,a) pair randomly and then get its oldest index
            index_deque = self.key_to_indices_dict.random_value()
            i = index_deque[0]
            # Save it if it's older (i.e. was added earlier)
            if self.timesteps[i] < t:
                t = self.timesteps[i]
                oldest_index = i
                oldest_deque = index_deque
        return oldest_index, oldest_deque

    def _write_new_index(self, index, observation, action):
        key = self._make_key(observation, action)
        new = (key not in self.key_to_indices_dict)
        if new:
            self.key_to_indices_dict[key] = deque()
        self.key_to_indices_dict[key].append(index)
        if new:
            self.heap.push(key, self.key_to_indices_dict[key])


class RedundancyHeap:
    def __init__(self):
        self.heap = []

    def push(self, key, index_deque):
        # heapq uses a min heap but we want a max heap, so multiply length by -1
        priority = -len(index_deque)
        return heapq.heappush(self.heap, (priority, key, index_deque))

    def pop(self):
        priority, key, index_deque = heapq.heappop(self.heap)
        return key, index_deque


class PrioritizedReplayMemory(ReplayMemory):
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


