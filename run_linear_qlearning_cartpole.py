import tensorflow as tf
import gym

import linear_qlearning
from pretrained_models import load_custom



def main():
    env = gym.make('CartPole-v0')
    model = load_custom('cartpole_mlp', num_actions=env.action_space.n)
    linear_qlearning.train(env, model)


if __name__ == '__main__':
    main()
