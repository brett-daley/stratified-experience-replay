import gym
import linear_qlearning


def main():
    env = gym.make('CartPole-v0')
    linear_qlearning.train(env)


if __name__ == '__main__':
    main()
