import numpy as np
import imageio
import glob
import argparse
import os


def save_to_np(input_dir, output_path):
    first_pic_path = os.path.join(input_dir, 'img_1.png')
    if not os.path.exists(first_pic_path):
        print(f'Error: {first_pic_path} does not exist')
        return

    # Load images into numpy array
    image_shape = imageio.imread(first_pic_path).shape
    pic_paths = glob.glob(os.path.join(input_dir, 'img_*.png'))

    images_album = np.empty(shape=(len(pic_paths), *image_shape), dtype=np.uint8)

    for i, im_path in enumerate(pic_paths):
        if i % 10_000 == 0:
            print(f'Processed {i}/{len(pic_paths)} images ({100.0 * i / len(pic_paths):.1f}%)')
        images_album[i] = imageio.imread(im_path)
    print('Done.')

    np.save(output_path, images_album)
    print(f'Screenshots saved in {output_path}.npy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converts images to single numpy array')
    parser.add_argument('--input_dir', default='pic_file', type=str,
                        help='name of directory with images (default: pic_file)')
    parser.add_argument('--output_path', default='pics_as_np', type=str,
                        help='destination file for saved NumPy array (default: pics_as_np)')

    args = parser.parse_args()

    save_to_np(input_dir=args.input_dir, output_path=args.output_path)
