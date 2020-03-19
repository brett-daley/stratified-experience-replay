import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, clone_model
import numpy as np
import argparse
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import atari_env
from dqn_original import DQNAgent, ReplayMemory, epsilon_schedule
from autoencoder_conv import encoder_layers, decoder_layers
from pretrained_models.model_editor import make_untrainable


class AutoencoderAgent(DQNAgent):
    def __init__(self, env, discount=0.99):
        self.env = env
        self.discount = discount
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-4)

        input_shape = env.observation_space.shape
        self.n_actions = env.action_space.n

        self.encoder = Sequential(encoder_layers(), name='Encoder')
        self.encoder.build([None, 128, 128, 3])
        print(self.encoder.summary())

        self.decoder = Sequential(decoder_layers(), name='Decoder')
        self.decoder.build([None, 128, 128, 3])
        print(self.decoder.summary())

        self.q_function = Sequential([Dense(self.n_actions)], name='Q-Function')
        self.q_function.build([None, 2048])
        print(self.q_function.summary())

        self.target_q_function = clone_model(self.q_function)
        self.target_q_function._name = 'Target Q-Function'
        make_untrainable(self.target_q_function)
        self.target_q_function.build([None, 2048])
        print(self.target_q_function.summary())

    def _q_values(self, observation):
        return self.q_function(self.encoder(observation))

    def _target_q_values(self, next_observation):
        return self.q_function(self.encoder(next_observation))

    @tf.function
    def train_autoencoder(self, observations):
        observations = self._preprocess(observations)

        with tf.GradientTape() as tape:
            predictions = self.decoder(self.encoder(observations))
            loss = tf.reduce_mean(tf.square(predictions - observations))

        trainable_variables = (self.encoder.trainable_variables + self.decoder.trainable_variables)
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))


def train(env, timesteps):
    agent = AutoencoderAgent(env)
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

                for _ in range(2500):
                    observations, _, _, _, _ = replay_memory.sample()
                    agent.train_autoencoder(observations)

                for _ in range(2500):
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
