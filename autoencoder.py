
import tensorflow as tf
import numpy as np
import imageio
import glob
import matplotlib.pyplot as plt

keras = tf.keras

# CH: Had to do this to prevent duplication error in libiomp5.dylib on my machine
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load images into images_album
single_pic_path = './pic_file/img_1.png'
pic_folder_path = './pic_file/*.png'
im = imageio.imread(single_pic_path)
image_shape = im.shape

num_images = len([file for file in glob.glob(pic_folder_path)])
images_album = np.empty(shape=(num_images, image_shape[0], image_shape[1], image_shape[2]))

i = 0
for im_path in glob.glob('./pic_file/*.png'):
     images_album[i, :, :, :] = imageio.imread(im_path)
     i += 1

# rescale the image into a range of [0,1]
images_album /= 255.

# turn images into a flattened vector
images_flat = images_album.reshape(len(images_album), -1)
train_images_flat = images_flat[0:int(np.floor(0.8 * num_images)), :]
test_images_flat = images_flat[int(np.floor(0.8 * num_images)):, :]

# Create autoencoder
encoder_layers = [
    keras.layers.InputLayer(train_images_flat.shape[1]),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(32, activation='relu')
]

decoder_layers = [
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(train_images_flat.shape[1], activation='sigmoid')
]


autoencoder = keras.Sequential(encoder_layers + decoder_layers)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(train_images_flat, train_images_flat,
                epochs=15,
                batch_size=23,
                shuffle=True,
                validation_data=(test_images_flat, test_images_flat),
                verbose=1)

fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for i, ind in enumerate([0, 10, 20, 30]):
    input_img = train_images_flat[ind].reshape(image_shape)
    reconstructed_img = autoencoder.predict(train_images_flat[ind:ind + 1])[0].reshape(image_shape)
    ax[0, i].imshow(input_img)
    ax[0, i].set_title('original')
    ax[1, i].imshow(reconstructed_img)
    ax[1, i].set_title('reconstructed')

plt.show()

