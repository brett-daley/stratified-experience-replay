import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer
import numpy as np
import argparse
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
from distutils.util import strtobool
import time
import math

import dqn_utils


class DQNAgent:
    def __init__(self, env, nsteps, minibatches, **kwargs):
        self.env = env
        assert nsteps >= 1
        self.nsteps = nsteps
        assert minibatches >= 1
        self.minibatches = minibatches
        self.discount = kwargs['discount']
        self.replay_memory = kwargs['rmem_constructor'](env)
        self.optimizer = kwargs['optimizer']
        self.scale_obs = kwargs['scale_obs']
        self.update_freq = kwargs['update_freq']

        input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n
        model_fn = kwargs['model_fn']

        def make_q_function():
            return tf.keras.models.Sequential([InputLayer(input_shape),
                                               *model_fn(),
                                               Dense(self.n_actions)])

        self.q_function = make_q_function()
        self.target_q_function = make_q_function()
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
        return self.scale_obs * tf.cast(observation, tf.float32)

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
        if (t % self.update_freq) == 0:
            self.copy_target_network()

        # Compute fractional training frequency
        train_freq_frac, train_freq_int = math.modf(self.minibatches / self.update_freq)
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
    def __init__(self, env, nsteps, minibatches, **kwargs):
        super().__init__(env, nsteps, minibatches, **kwargs)
        self.mstraps = kwargs['mstraps']

    def update(self, t):
        if (t % self.update_freq) == 0:
            for _ in range(self.mstraps):
                self.copy_target_network()

                for _ in range(self.minibatches // self.nsteps):
                    minibatch = self._sample()
                    self._train(*minibatch)


def train(env, agent, prepopulate, epsilon_schedule, timesteps, seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    observation = env.reset()

    print('timestep', 'episode', 'avg_return', 'epsilon', 'hours', sep='  ', flush=True)
    for t in range(-prepopulate, timesteps+1):  # Relative to training start
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
                        help='(int) Number of target network updates per training epoch. Default: 1')
    parser.add_argument('-k', '--minibatches', type=int, default=2500,
                        help='(int) Number of minibatches per training epoch. Default: 2500')
    parser.add_argument('--timesteps', type=int, default=3_000_000,
                        help='(int) Training duration. Default: 3_000_000')
    parser.add_argument('--seed', type=int, default=0,
                        help='(int) Seed for random number generation. Default: 0')
    args = parser.parse_args()

    env = dqn_utils.make_env(args.env, args.seed)
    hparams = dqn_utils.get_hparams(args.env)

    if args.mstraps > 0:
        agent_cls = BatchmodeDQNAgent
        hparams['mstraps'] = args.mstraps
    else:
        agent_cls = DQNAgent

    print(hparams)
    agent = agent_cls(env, args.nsteps, args.minibatches, **hparams)

    print(f'Training {type(agent).__name__} (n={args.nsteps}, m={args.mstraps}, k={args.minibatches}) on {args.env} for {args.timesteps} timesteps with seed={args.seed}')
    train(env, agent, hparams['prepopulate'], hparams['epsilon_schedule'], args.timesteps, args.seed)