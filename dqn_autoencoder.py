import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, clone_model
import numpy as np
import argparse
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import atari_env
from dqn_original import DQNAgent, ReplayMemory, add_common_args, epsilon_schedule, train
from autoencoder_conv import encoder_layers, decoder_layers
from pretrained_models.model_editor import make_untrainable


class AutoencoderAgent(DQNAgent):
    def __init__(self, env, nsteps, discount=0.99):
        self.env = env
        assert nsteps >= 1
        self.nsteps = nsteps
        self.discount = discount
        self.replay_memory = ReplayMemory(env)
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

    def update(self, t):
        if (t % 10_000) == 0:
            for _ in range(2500):
                observations, _, _, _, _ = self._sample(nsteps=1)
                self._train_autoencoder(observations)

            for _ in range(2500):
                minibatch = self._sample(self.nsteps)
                self._train(*minibatch)

    @tf.function
    def _train_autoencoder(self, observations):
        observations = self._preprocess(observations)

        with tf.GradientTape() as tape:
            predictions = self.decoder(self.encoder(observations))
            loss = tf.reduce_mean(tf.square(predictions - observations))

        trainable_variables = (self.encoder.trainable_variables + self.decoder.trainable_variables)
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    env = atari_env.make(args.env, args.seed, size=128, grayscale=False, history_len=1)
    agent = AutoencoderAgent(env, nsteps=args.nsteps)
    train(env, agent, args.timesteps, args.seed)
