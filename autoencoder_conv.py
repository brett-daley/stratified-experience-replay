import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, InputLayer, Reshape
import numpy as np
import imageio
import glob
import matplotlib.pyplot as plt
keras = tf.keras

# CH: Had to do this to prevent duplication error in libiomp5.dylib on my machine
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load images from NumPy array
images_album = np.load('pics_as_array.npy')
num_images = len(images_album)
image_shape = images_album[0].shape

# rescale the image into a range of [0,1], and convert to np array
images_album /= 255.

# Split into train and test batches
train_images = images_album[:int(np.floor(0.8 * num_images))]
test_images = images_album[int(np.floor(0.8 * num_images)):]

encoder_layers = [
    InputLayer(image_shape),
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

autoencoder = keras.Sequential(encoder_layers + decoder_layers)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

num_epochs = 50
b_size = 128
num_samples = len(train_images) + len(test_images)

autoencoder.fit(train_images, train_images,
                epochs=num_epochs,
                batch_size=b_size,
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
autoencoder.save('savedautoencoder_samples{}_epochs{}_batchsize{}.h5'.format(num_samples, num_epochs, b_size))
print("autoencoder saved to", 'savedautoencoder_samples{}_epochs{}_batchsize{}.h5'.format(num_samples, num_epochs, b_size))
