import tensorflow as tf
import numpy as np
import imageio
import glob
import matplotlib.pyplot as plt
keras = tf.keras

# CH: Had to do this to prevent duplication error in libiomp5.dylib on my machine
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

# rescale the image into a range of [0,1], and convert to np array
images_album /= 255.
np_images_album = np.array(images_album)

# Split into train and test batches
train_images = np_images_album[0:int(np.floor(0.8 * num_images)), :, :, :]
test_images = np_images_album[int(np.floor(0.8 * num_images)):, :, :, :]

image_shape_flat = image_shape[0] * image_shape[1] * image_shape[2]

encoder_layers = [
    keras.layers.InputLayer(image_shape),
    keras.layers.Conv2D(32, kernel_size=7, strides=4, activation='relu', padding='same'),
    keras.layers.Conv2D(128, kernel_size=7, strides=4, activation='relu', padding='same'),
    keras.layers.Conv2D(256, kernel_size=3, strides=2, activation='relu', padding='same'),
    keras.layers.Conv2D(512, kernel_size=3, strides=2, activation='relu', padding='same'),
    keras.layers.Flatten()
]

decoder_layers = [
    keras.layers.Reshape(target_shape=(2, 2, 512)),
    keras.layers.Conv2D(512, kernel_size=3, activation='relu', padding='same'),
    keras.layers.UpSampling2D(size=(2, 2)),
    keras.layers.Conv2D(256, kernel_size=3, activation='relu', padding='same'),
    keras.layers.UpSampling2D(size=(2, 2)),
    keras.layers.Conv2D(128, kernel_size=7, activation='relu', padding='same'),
    keras.layers.UpSampling2D(size=(4, 4)),
    keras.layers.Conv2D(32, kernel_size=7, activation='relu', padding='same'),
    keras.layers.UpSampling2D(size=(4, 4)),
    keras.layers.Conv2D(3, kernel_size=3, activation='sigmoid', padding='same'),
]

autoencoder = keras.Sequential(encoder_layers + decoder_layers)
autoencoder.summary()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

num_epochs = 50
b_size = 128
num_samples = len(train_images) + len(test_images)

autoencoder.fit(train_images, train_images,
                epochs=num_epochs,
                batch_size=b_size,
                shuffle=True,
                validation_data=(test_images, test_images),
                verbose=1)

# autoencoder.fit(train_images, train_images,
#                 epochs=50,
#                 batch_size=128,
#                 shuffle=True,
#                 validation_data=(test_images, test_images),
#                 verbose=1)

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
