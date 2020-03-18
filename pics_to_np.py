import numpy as np
import imageio
import glob
import argparse
import os


def save_to_np(input_dir, output_path):
    # Load images into numpy array
    first_pic_path = os.path.join(input_dir, 'img_1.png')
    image_shape = imageio.imread(first_pic_path).shape
    pic_paths = glob.glob(os.path.join(input_dir, 'img_*.png'))

    images_album = np.empty(shape=(len(pic_paths), *image_shape), dtype=np.uint8)

    for i, im_path in enumerate(pic_paths):
        if i % 10_000 == 0:
            print(f'Processed {i}/{len(pic_paths)} images ({100.0 * i / len(pic_paths):.1f}%)')
        images_album[i] = imageio.imread(im_path)
    print('Done.')

    np.save(output_path, images_album)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts Atari screenshots to single numpy array')
    parser.add_argument('input_dir', type=str,
                        help='name of directory with images (ex: ./pic_file)')
    parser.add_argument('output_path', type=str,
                        help='destination file for saved NumPy array (ex: pics_as_np)')

    args = parser.parse_args()

    save_to_np(input_dir=args.input_dir, output_path=args.output_path)

    print("Screenshots saved to file!")