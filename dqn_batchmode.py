import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer
import numpy as np
import argparse
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import atari_env
from dqn_original import DQNAgent, ReplayMemory, epsilon_schedule


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
