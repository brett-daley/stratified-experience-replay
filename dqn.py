import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer
import numpy as np
import argparse
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
from distutils.util import strtobool
import time
import math

import atari_env
from dqn_utils import ReplayMemory


def make_q_function(input_shape, n_actions, h_size=512):
    layers = [InputLayer(input_shape),
              Dense(h_size, activation='tanh'),
              Dense(h_size, activation='tanh'),
              Dense(n_actions)]
    return tf.keras.models.Sequential(layers)


class DQNAgent:
    def __init__(self, env, nsteps, mstraps, minibatches, discount=0.99):
        self.env = env
        assert nsteps >= 1
        self.nsteps = nsteps
        self.mstraps = mstraps  # Only used by BatchmodeDQNAgent
        assert minibatches >= 1
        self.minibatches = minibatches
        self.discount = discount
        self.replay_memory = ReplayMemory(env)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)

        input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        self.q_function = make_q_function(input_shape, self.n_actions)
        self.target_q_function = make_q_function(input_shape, self.n_actions)
        print(self.q_function.summary())

    def policy(self, observation, epsilon):
        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        return self._greedy_action(observation).numpy()

    def save(self, *transition):
        self.replay_memory.save(*transition)

    def _sample(self):
        return self.replay_memory.sample(self.discount, self.nsteps)

    def _preprocess(self, observation):
        return tf.cast(observation, tf.float32)

    def _q_values(self, observation):
        return self.q_function(observation)

    def _target_q_values(self, next_observation):
        return self.target_q_function(next_observation)

    @tf.function
    def _greedy_action(self, observation):
        observation = self._preprocess(observation)
        q_values = self._q_values(observation[None])
        return tf.argmax(q_values, axis=1)[0]

    def update(self, t):
        update_freq = 10_000
        if (t % update_freq) == 0:
            self.copy_target_network()

        # Compute fractional training frequency
        train_freq_frac, train_freq_int = math.modf(self.minibatches / update_freq)
        # The integer portion tells us the minimum number of minibatches we do each timestep
        for _ in range(int(train_freq_int)):
            minibatch = self._sample()
            self._train(*minibatch)
        # The fractional portion tells us how often to add an extra minibatch
        if train_freq_frac != 0.0:
            extra_train_freq = round(1.0 / train_freq_frac)
            if (t % extra_train_freq) == 0:
                minibatch = self._sample()
                self._train(*minibatch)

    @tf.function
    def _train(self, observations, actions, nstep_rewards, done_mask, bootstrap_observations):
        observations = self._preprocess(observations)
        bootstrap_observations = self._preprocess(bootstrap_observations)

        target_q_values = self._target_q_values(bootstrap_observations)
        nstep_discount = pow(self.discount, self.nsteps)
        bootstraps = nstep_discount * tf.reduce_max(target_q_values, axis=1)

        with tf.GradientTape() as tape:
            q_values = self._q_values(observations)
            action_mask = tf.one_hot(actions, depth=self.n_actions)
            q_values = tf.reduce_sum(action_mask * q_values, axis=1)

            returns = nstep_rewards + (done_mask * bootstraps)
            loss = tf.reduce_mean(tf.square(returns - q_values))

        gradients = tape.gradient(loss, self.q_function.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_function.trainable_variables))

    @tf.function
    def copy_target_network(self):
        for var, target_var in zip(self.q_function.trainable_variables, self.target_q_function.trainable_variables):
            target_var.assign(var)


class BatchmodeDQNAgent(DQNAgent):
    def update(self, t):
        if (t % 10_000) == 0:
            for _ in range(self.mstraps):
                self.copy_target_network()

                for _ in range(self.minibatches // self.nsteps):
                    minibatch = self._sample()
                    self._train(*minibatch)


def epsilon_schedule(t, timeframe=300_000, min_epsilon=0.1):
    return np.clip(1.0 - (1.0 - min_epsilon) * (t / timeframe), min_epsilon, 1.0)
    # epsilon set constant at 0.1
    # return 0.1


def train(env, agent, timesteps, seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    observation = env.reset()

    print(f'Training {type(agent).__name__} (n={agent.nsteps}, m={agent.mstraps}, k={agent.minibatches}) on {env.spec.id} for {timesteps} timesteps with seed={seed}')
    print('timestep', 'episode', 'avg_return', 'epsilon', 'hours', sep='  ', flush=True)
    for t in range(-50_000, timesteps+1):  # Relative to training start
        epsilon = epsilon_schedule(t)

        if t == 0:
            start_time = time.time()

        if t >= 0:
            if (t % 5_000) == 0:
                rewards = env.get_episode_rewards()
                hours = (time.time() - start_time) / 3600
                print(f'{t}  {len(rewards)}  {np.mean(rewards[-100:])}  {epsilon:.3f}  {hours:.3f}', flush=True)

            agent.update(t)

        action = agent.policy(observation, epsilon)
        new_observation, reward, done, _ = env.step(action)
        agent.save(observation, action, reward, done)
        observation = env.reset() if done else new_observation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='pong',
                        help='(str) Name of Atari game. Default: pong')
    parser.add_argument('-n', '--nsteps', type=int, default=1,
                        help='(int) Number of rewards to use before bootstrapping. Default: 1')
    parser.add_argument('-m', '--mstraps', type=int, default=1,
                        help='(int) Number of bootstrapping. Default: 1')
    parser.add_argument('-k', '--minibatches', type=int, default=2500,
                        help='(int) Number of minibatches per training epoch. Default: 2500')
    parser.add_argument('--timesteps', type=int, default=3_000_000,
                        help='(int) Training duration. Default: 3_000_000')
    parser.add_argument('--seed', type=int, default=0,
                        help='(int) Seed for random number generation. Default: 0')
    args = parser.parse_args()

    env = atari_env.make(args.env, args.seed)

    agent_cls = BatchmodeDQNAgent if (args.mstraps > 0) else DQNAgent
    agent = agent_cls(env, args.nsteps, args.mstraps, args.minibatches)

    train(env, agent, args.timesteps, args.seed)
