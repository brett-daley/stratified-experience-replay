import atari_env


def main():
    env = atari_env.make('BreakoutNoFrameskip-v4')
    env.seed(0); env.action_space.seed(0)  # For reproducibility
    env.reset()

    for t in range(1000):
        action = env.action_space.sample()     # Take a random action
        _, reward, done, _ = env.step(action)  # Returns (obs, reward, done, info)

        # TODO: do something useful here
        print(t, reward, done)

        if done:
            env.reset()

    env.close()


if __name__ == '__main__':
    main()
