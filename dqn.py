import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer
import numpy as np
import argparse
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import atari_env


def make_q_function(input_shape, n_actions):
    layers = [InputLayer(input_shape),
              Conv2D(32, kernel_size=8, strides=4, activation='relu'),
              Conv2D(64, kernel_size=4, strides=2, activation='relu'),
              Conv2D(64, kernel_size=3, strides=1, activation='relu'),
              Flatten(),
              Dense(512, activation='relu'),
              Dense(n_actions)]
    return tf.keras.models.Sequential(layers)


class DQNAgent:
    def __init__(self, env, discount=0.99):
        self.env = env
        self.discount = discount
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-4)

        input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.q_function = make_q_function(input_shape, self.n_actions)
        self.target_q_function = make_q_function(input_shape, self.n_actions)
        print(self.q_function.summary())

    def policy(self, observation, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        return self._greedy_action(observation).numpy()

    @tf.function
    def _greedy_action(self, observation):
        observation = tf.cast(observation, tf.float32) / 255.0

        q_values = self.q_function(observation[None])
        return tf.argmax(q_values, axis=1)[0]

    @tf.function
    def train(self, observations, actions, rewards, dones, next_observations):
        observations = tf.cast(observations, tf.float32) / 255.0
        next_observations = tf.cast(next_observations, tf.float32) / 255.0

        with tf.GradientTape() as tape:
            q_values = self.q_function(observations)
            action_mask = tf.one_hot(actions, depth=self.n_actions)
            q_values = tf.reduce_sum(action_mask * q_values, axis=1)

            target_q_values = self.target_q_function(next_observations)
            done_mask = (1.0 - dones)
            returns = rewards + done_mask * self.discount * tf.reduce_max(target_q_values, axis=1)
            loss = tf.reduce_mean(tf.square(returns - q_values))

        gradients = tape.gradient(loss, self.q_function.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_function.trainable_variables))

    @tf.function
    def copy_target_network(self):
        for var, target_var in zip(self.q_function.trainable_variables, self.target_q_function.trainable_variables):
            target_var.assign(var)


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

    def save(self, observation, action, reward, done):
        p = self.pointer
        self.observations[p], self.actions[p], self.rewards[p], self.dones[p] = observation, action, reward, done
        self.size_now = min(self.size_now + 1, self.capacity)
        self.pointer = (self.pointer + 1) % self.capacity

    def sample(self):
        i = np.random.randint(self.size_now - 1, size=self.batch_size)
        i = (self.pointer + i) % self.size_now
        return self.observations[i], self.actions[i], self.rewards[i], self.dones[i], self.observations[(i+1) % self.size_now]


def epsilon_schedule(t, timeframe=1_000_000, min_epsilon=0.1):
    return np.clip(1.0 - (1.0 - min_epsilon) * (t / timeframe), min_epsilon, 1.0)


def train(env, timesteps):
    agent = DQNAgent(env)
    replay_memory = ReplayMemory(env)
    observation = env.reset()

    print('timestep', 'episode', 'avg_return', 'epsilon', sep='  ', flush=True)
    for t in range(-50_000, timesteps+1):  # Relative to training start
        epsilon = epsilon_schedule(t)

        if t >= 0:
            if (t % 5_000) == 0:
                rewards = env.get_episode_rewards()
                print(f'{t}  {len(rewards)}  {np.mean(rewards[-100:])}  {epsilon:.3f}', flush=True)

            if (t % 10_000) == 0:
                agent.copy_target_network()

            if (t % 4) == 0:
                minibatch = replay_memory.sample()
                agent.train(*minibatch)

        action = agent.policy(observation, epsilon)
        new_observation, reward, done, _ = env.step(action)
        replay_memory.save(observation, action, reward, done)
        observation = env.reset() if done else new_observation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='pong', help='(str) Name of Atari game. Default: pong')
    parser.add_argument('--timesteps', type=int, default=5_000_000, help='(int) Training duration. Default: 5_000_000')
    parser.add_argument('--seed', type=int, default=0, help='(int) Seed for random number generation. Default: 0')
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    env = atari_env.make(args.env, seed=args.seed)
    train(env, args.timesteps)
