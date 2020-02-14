import tensorflow as tf

import atari_env
import linear_qlearning
from pretrained_models import load_keras


def main():
    env = atari_env.make('PongNoFrameskip-v4')
    model = load_keras('vgg16', input_shape=env.observation_space.shape,
                       num_actions=env.action_space.n)
    linear_qlearning.train(env, model)


if __name__ == '__main__':
    main()
