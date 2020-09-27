import numpy as np
import random
from collections import deque

from dqn_utils.random_dict import RandomDict
from dqn_utils.segment_tree import MinSegmentTree, SumSegmentTree


class ReplayMemory:
    def __init__(self, env, batch_size=32, capacity=1_000_000, history_len=1):
        self.batch_size = batch_size
        self.capacity = capacity                      # Nominal size
        self._buff_size = capacity + (history_len-1)  # Actual size bigger to save history
        self.history_len = history_len
        self.size_now = 0
        self.pointer = 0
        self.base = None  # Another pointer that's used to shift our sampled indices

        # NOTE: Must initialize observations with zeros to implicitly pad early observations
        self.observations = np.zeros(shape=[self._buff_size, *env.observation_space.shape],
                                     dtype=env.observation_space.dtype)
        self.actions = np.empty(self._buff_size, dtype=np.int32)
        self.rewards = np.empty(self._buff_size, dtype=np.float32)
        self.dones = np.empty(self._buff_size, dtype=np.float32)

    def save(self, observation, action, reward, done, new_observation, history):
        p = self.pointer
        self.observations[p], self.actions[p], self.rewards[p], self.dones[p] = observation, action, reward, done
        self.size_now = min(self.size_now + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self._buff_size
        self.base = (self.pointer - self.size_now) % self._buff_size

    def _is_full(self):
        return self.size_now == self.capacity

    def sample(self, discount, nsteps, train_frac):
        if nsteps != 1:
            raise NotImplementedError('ReplayMemory supports only 1-step returns')

        # Sample indices for the minibatch
        i = np.asarray([self._sample_index(nsteps) for _ in range(self.batch_size)])

        observations = self._get_batch(self.observations, i)
        actions = self._get_batch(self.actions, i)
        dones = self._get_batch(self.dones, i)
        bootstrap_observations = self._get_batch(self.observations, i + nsteps)

        nstep_rewards = self._get_batch(self.rewards, i)
        done_mask = (1.0 - dones)

        weights = np.ones_like(nstep_rewards)
        return (observations, actions, nstep_rewards, done_mask, bootstrap_observations, weights), i

    def _sample_index(self, nsteps):
        # Subtract nsteps so we have room to bootstrap at the end
        return np.random.randint(self.size_now - nsteps)

    def _get_batch(self, array, indices):
        # Indices should be in [0, size_now - nsteps]
        assert indices.max() <= self.size_now - 1
        # Shift by the base pointer
        indices = (self.base + indices) % self._buff_size

        # Now make the batch
        if array is self.observations:
            return self._make_observation_batch(indices)
        return array[indices]

    def _make_observation_batch(self, indices):
        observations = [self._get_history(i) for i in indices]
        return np.stack(observations)

    def _get_history(self, i):
        history = []
        for j in range(self.history_len):
            # Count backwards: x = i, i-1, i-2, ...
            x = (i - j) % self._buff_size
            # Stop if we hit a previous episode
            if (j > 0) and self.dones[x]:
                break
            # Add this observation to the history
            history.append(self.observations[x])

        # If we stopped early, then we need to pad with zeros
        zero_pad = np.zeros_like(self.observations[0])
        while len(history) < self.history_len:
            history.append(zero_pad)

        # Our history is backwards; reverse it, then stack along the last axis
        history = list(reversed(history))
        return np.concatenate(history, axis=-1)


class StratifiedReplayMemory(ReplayMemory):
    def __init__(self, env, batch_size=32, capacity=1_000_000, history_len=1):
        super().__init__(env, batch_size, capacity, history_len)
        self.hashes = np.empty(self._buff_size, dtype=np.longlong)
        self.pair_to_indices_dict = RandomDict()

    def save(self, observation, action, reward, done, new_observation, history):
        # self._update_histogram_data()

        # If we're full, we need to delete the oldest entry first
        if self._is_full():
            # Oldest experience is actually (history_len-1) ahead of the pointer
            x = (self.pointer + self.history_len - 1) % self._buff_size
            old_key = (self.hashes[x], self.actions[x])
            old_index_deque = self.pair_to_indices_dict[old_key]
            old_index_deque.popleft()
            if not old_index_deque:
                self.pair_to_indices_dict.pop(old_key)

        # Update the index for the new entry
        history_hash = hash(history.tostring())
        new_key = (history_hash, action)
        if new_key not in self.pair_to_indices_dict:
            self.pair_to_indices_dict[new_key] = deque()
        self.pair_to_indices_dict[new_key].append(self.pointer)

        # Save the transition
        self.hashes[self.pointer] = history_hash
        super().save(observation, action, reward, done, new_observation, history)

    def sample(self, discount, nsteps, train_frac):
        if nsteps != 1:
            raise NotImplementedError('StratifiedReplayMemory supports only 1-step returns')
            # TODO: Can we generalize this to n-step returns?
        return super().sample(discount, nsteps, train_frac)

    def _sample_index(self, nsteps):
        # Keep trying until we don't sample too close to the end of the buffer
        while True:
            index_deque = self.pair_to_indices_dict.random_value()
            x = random.choice(index_deque)  # Physical address
            i = (x - self.base) % self._buff_size  # Relative to base
            if i < self.size_now - nsteps:
                return i

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


class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, env, batch_size=32, capacity=1_000_000, history_len=1):
        super().__init__(env, batch_size, capacity, history_len)

        # Just hardcode the prioritization hyperparameters here
        # These are the default from the original paper, and OpenAI uses them too
        self.alpha = 0.6
        self.beta_schedule = lambda train_frac: 0.4 + 0.6 * train_frac
        self.epsilon = 1e-6

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def __len__(self):
        return self.size_now

    def save(self, observation, action, reward, done, new_observation, history):
        p = self.pointer
        super().save(observation, action, reward, done, new_observation, history)
        self._it_sum[p] = self._max_priority ** self.alpha
        self._it_min[p] = self._max_priority ** self.alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, discount, nsteps, train_frac):
        if nsteps != 1:
            raise NotImplementedError('PrioritizedReplayMemory supports only 1-step returns')

        # HACK: lets us pass these indices to self._sample_index one by one
        self.t = 0
        self.idxes = self._sample_proportional(self.batch_size)

        # If we sample an invalid index, we resample randomly from a new batch to avoid bias
        for i in range(self.batch_size):
            # An index is invalid if it's too far from the base
            # The relative distance must be in [0, size_now - nsteps]
            while (self.idxes[i] - self.base) % self._buff_size > self.size_now - 1:
                # Keep resampling until we get a good index
                new_idxes = self._sample_proportional(self.batch_size)
                self.idxes[i] = random.choice(new_idxes)

        (observations, actions, nstep_rewards, done_mask, bootstrap_observations, _), idxes = super().sample(discount, nsteps, train_frac)

        beta = self.beta_schedule(train_frac)
        assert beta > 0

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)

        return (observations, actions, nstep_rewards, done_mask, bootstrap_observations, weights), idxes

    def _sample_index(self, nsteps):
        i = self.idxes[self.t]
        self.t += 1
        return i

    def update_td_errors(self, indices, td_errors):
        new_priorities = np.abs(td_errors) + self.epsilon
        self._update_priorities(indices, new_priorities)

    def _update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self.alpha
            self._it_min[idx] = priority ** self.alpha

            self._max_priority = max(self._max_priority, priority)
