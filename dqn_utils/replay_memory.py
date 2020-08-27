import numpy as np
import random


class PickyReplayMemory:
    def __init__(self, env, batch_size=32):
        # "env" here is just used to align PickyReplayMemory with ReplayMemory's instantiation in dqn.py
        self.is_using_picky_memory = True
        self.batch_size = batch_size
        self.transitions = set()
        self.size_now = 0

    def save(self, observation, action, reward, done, new_observation):

        # Must convert to tuple to pass into set (set needs hash, which array doesn't have)
        # In the sample() method below we'll convert it back
        tuple_observation = tuple(observation)
        tuple_new_observation = tuple(new_observation)

        transition = (tuple_observation, action, reward, done, tuple_new_observation)

        self.transitions.add(transition)
        self.size_now = len(self.transitions)


    def sample(self, discount, nsteps):
        # Dummy inputs in sample() method call are just to align PickyReplayMemory with ReplayMemory's "API" in dqn.py
        assert self.size_now > self.batch_size, "Error: batch size > current replay memory size"

        # We must convert back to an array, and ensure all are in the same dtype and formats as in the
        # regular replay memory
        sampled_transitions = random.sample(self.transitions, self.batch_size)
        sampled_transitions_array = np.array(sampled_transitions)

        observation_tuples = sampled_transitions_array[:, 0]
        observations = np.array([*observation_tuples], dtype=np.float32)

        actions = sampled_transitions_array[:, 1].astype(np.int32)
        rewards = sampled_transitions_array[:, 2].astype(np.float32)
        dones = sampled_transitions_array[:, 3].astype(np.float32)

        new_observation_tuples = sampled_transitions_array[:, 4]
        new_observations = np.array([*new_observation_tuples], dtype=np.float32)

        done_mask = (1.0 - dones)

        return observations, actions, rewards, done_mask, new_observations


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
