import tensorflow as tf
import gym

import linear_qlearning
import pretrained_models



def main():
    env = gym.make('CartPole-v0')
    model = pretrained_models.load('cartpole_mlp', unlearn_last_layer=True)
    linear_qlearning.train(env, model)


if __name__ == '__main__':
    main()
