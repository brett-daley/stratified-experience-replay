import numpy as np
import imageio
import glob
import argparse


def save_to_np(input_path, output_path):

    # Load images into numpy array
    single_pic_path = './pic_file/img_1.png'
    pic_folder_path = str(input_path) + '*.png'
    im = imageio.imread(single_pic_path)
    image_shape = im.shape

    num_images = len([file for file in glob.glob(pic_folder_path)])
    images_album = np.empty(shape=(num_images, *image_shape))

    for i, im_path in enumerate(glob.glob('./pic_file/*.png')):
        images_album[i, :, :, :] = imageio.imread(im_path)

    np.save(str(output_path), images_album)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts Atari screenshots to single numpy array')
    parser.add_argument('input_path', type=str,
                        help='name of file with images (ex: ./pic_file/)')
    parser.add_argument('output_path', type=str,
                        help='destination file for saved NumPy array (ex: pics_as_np)')

    args = parser.parse_args()

    save_to_np(input_path=args.input_path, output_path=args.output_path)

    print("Screenshots saved to file!")