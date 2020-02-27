import atari_env
import os
from PIL import Image


def take_pics(num_images, env_name, folder_path='game_images/'):
    # Check if images folder exists; if not, create one
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # Create environment and use wrappers in atari_env.py
    env = atari_env.make(env_name)
    observation = env.reset()

    for i in range(num_images):
        # Save image to file
        name_string = folder_path + 'img_' + str(i+1) + '.png'
        Image.fromarray((observation * 255).astype('uint8'), mode='RGB').save(name_string)

        # Take random action
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)

        # If game ends, reset
        if done:
            env.reset()


if __name__ == '__main__':
    take_pics(num_images=1_000_000, env_name='PongNoFrameskip-v4', folder_path='game_images/')
