import atari_env
import os


def main():
    # Check if images folder exists; if not, create one
    if not os.path.exists('pong_images'):
        os.mkdir('pong_images')

    # Create environment and use wrappers in atari_env.py
    env = atari_env.make('PongNoFrameskip-v4')
    env.reset()

    num_images = 1_000_000

    for i in range(num_images):
        # Take random action
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        
        # Save image to file
        name_string = 'pong_images/img_' + str(i+1).zfill(7) + '.png'
        env.env.ale.saveScreenPNG(bytes(name_string, 'utf-8'))

        # If game ends, reset
        if done:
            env.reset()


if __name__ == '__main__':
    main()
