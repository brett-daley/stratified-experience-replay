import atari_env
import os
import cv2
import argparse


def take_pics(num_images, env_name, folder_path):
    # Check if images folder exists; if not, create one
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    # Create environment and use wrappers in atari_env.py
    env = atari_env.make(env_name, seed=0)
    observation = env.reset()

    for i in range(num_images):
        # Save image to file
        name_string = folder_path + 'img_' + str(i + 1) + '.png'
        observation = cv2.cvtColor(observation, cv2.COLOR_BGR2RGB)
        cv2.imwrite(name_string, observation)

        # Take random action
        action = env.action_space.sample()
        observation, reward, done, _ = env.step(action)

        # If game ends, reset
        if done:
            env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produces desired number of atari environment screenshots.')
    parser.add_argument('num_images', type=int, help='number of screenshots to take')
    parser.add_argument('env_name', type=str, help='name of atari game to screenshot')
    parser.add_argument('folder_path', type=str, help='the folder to save the images to')

    args = parser.parse_args()

    take_pics(num_images=args.num_images, env_name=args.env_name, folder_path=args.folder_path)

    print("Screenshots complete!")
