import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, InputLayer, Reshape
from tensorflow.keras.models import Sequential
import numpy as np
import imageio
import glob
import matplotlib.pyplot as plt
import argparse

# CH: Had to do this to prevent duplication error in libiomp5.dylib on my machine
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

encoder_layers = [
    Conv2D(32,  kernel_size=7, strides=4, activation='relu', padding='same'),
    Conv2D(128, kernel_size=7, strides=4, activation='relu', padding='same'),
    Conv2D(256, kernel_size=3, strides=2, activation='relu', padding='same'),
    Conv2D(512, kernel_size=3, strides=2, activation='relu', padding='same'),
    Flatten()
]

decoder_layers = [
    Reshape(target_shape=(2, 2, 512)),
    Conv2DTranspose(256, kernel_size=3, strides=2, activation='relu',   padding='same'),
    Conv2DTranspose(128, kernel_size=3, strides=2, activation='relu',   padding='same'),
    Conv2DTranspose(32,  kernel_size=7, strides=4, activation='relu',   padding='same'),
    Conv2DTranspose(3,   kernel_size=7, strides=4, activation='linear', padding='same'),
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_array', type=str, default='pics_as_np.npy',
                        help='Path to image array. Default: pics_as_np.npy')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Minibatch size. Default: 128')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of passes over the training set. Default: 50')
    args = parser.parse_args()

    # Load images from NumPy array
    images_album = np.load(args.path_to_array)
    num_images = len(images_album)
    image_shape = images_album[0].shape

    # rescale the image into a range of [0,1], and convert to np array
    images_album = images_album.astype(np.float32) / 255.

    # Split into train and test batches
    train_images = images_album[:int(np.floor(0.8 * num_images))]
    test_images = images_album[int(np.floor(0.8 * num_images)):]

    layers = [InputLayer(image_shape)] + encoder_layers + decoder_layers
    autoencoder = Sequential(layers)
    print(autoencoder.summary())
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    num_samples = len(train_images) + len(test_images)

    autoencoder.fit(train_images, train_images,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    shuffle=True,
                    validation_data=(test_images, test_images),
                    verbose=1)

    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for i, ind in enumerate([0, 10, 20, 30]):
        input_img = train_images[ind].reshape(image_shape)
        reconstructed_img = autoencoder.predict(train_images[ind:ind + 1])[0].reshape(image_shape)
        ax[0, i].imshow(input_img)
        ax[0, i].set_title('original')
        ax[1, i].imshow(reconstructed_img)
        ax[1, i].set_title('reconstructed')

    plt.show()

    # Save autoencoder
    autoencoder.save(f'savedautoencoder_samples{num_samples}_epochs{args.epochs}_batchsize{args.batch_size}.h5')
    print("autoencoder saved to", f'savedautoencoder_samples{num_samples}_epochs{args.epochs}_batchsize{args.batch_size}.h5')
