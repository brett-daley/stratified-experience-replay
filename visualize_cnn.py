'''Saves images that approximately-maximally activate each
convolutional filter in the given neural network.
Modified from https://keras.io/examples/conv_filter_visualization'''
from __future__ import print_function

from argparse import ArgumentParser
import time
import os
import numpy as np
from PIL import Image as pil_image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import pretrained_models


def normalize(x):
    """utility function to normalize a tensor.

    # Arguments
        x: An input tensor.

    # Returns
        The normalized input tensor.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def deprocess_image(x):
    """utility function to convert a float array into a valid uint8 image.

    # Arguments
        x: A numpy-array representing the generated image.

    # Returns
        A processed numpy-array, which could be used in e.g. imshow.
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def process_image(x, former):
    """utility function to convert a valid uint8 image back into a float array.
       Reverses `deprocess_image`.

    # Arguments
        x: A numpy-array, which could be used in e.g. imshow.
        former: The former numpy-array.
                Need to determine the former mean and variance.

    # Returns
        A processed numpy-array representing the generated image.
    """
    if K.image_data_format() == 'channels_first':
        x = x.transpose((2, 0, 1))
    return (x / 255 - 0.5) * 4 * former.std() + former.mean()


def visualize_layer(model,
                    layer_name,
                    step=1.,
                    epochs=15,
                    upscaling_steps=9,
                    upscaling_factor=1.2,
                    output_dim=(412, 412),
                    filter_range=(0, None)):
    """Visualizes the most relevant filters of one conv-layer in a certain model.

    # Arguments
        model: The model containing layer_name.
        layer_name: The name of the layer to be visualized.
                    Has to be a part of model.
        step: step size for gradient ascent.
        epochs: Number of iterations for gradient ascent.
        upscaling_steps: Number of upscaling steps.
                         Starting image is in this case (80, 80).
        upscaling_factor: Factor to which to slowly upgrade
                          the image towards output_dim.
        output_dim: [img_width, img_height] The output image dimensions.
        filter_range: Tupel[lower, upper]
                      Determines the to be computed filter numbers.
                      If the second value is `None`,
                      the last filter will be inferred as the upper boundary.
    """

    def _generate_filter_image(input_img,
                               layer_output,
                               filter_index):
        """Generates image for one particular filter.

        # Arguments
            input_img: The input-image Tensor.
            layer_output: The output-image Tensor.
            filter_index: The to be processed filter number.
                          Assumed to be valid.

        #Returns
            Either None if no image could be generated.
            or a tuple of the image (array) itself and the last loss.
        """
        s_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # we start from a gray image with some random noise
        intermediate_dim = tuple(
            int(x / (upscaling_factor ** upscaling_steps)) for x in output_dim)
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random(
                (1, 3, intermediate_dim[0], intermediate_dim[1]))
        else:
            input_img_data = np.random.random(
                (1, intermediate_dim[0], intermediate_dim[1], 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # Slowly upscaling towards the original size prevents
        # a dominating high-frequency of the to visualized structure
        # as it would occur if we directly compute the 412d-image.
        # Behaves as a better starting point for each following dimension
        # and therefore avoids poor local minima
        for up in reversed(range(upscaling_steps)):
            # we run gradient ascent for e.g. 20 steps
            for _ in range(epochs):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                # some filters get stuck to 0, we can skip them
                if loss_value <= K.epsilon():
                    print(f'Activation of filter {filter_index:3}: {loss_value:5.0f} (appears to be dead; skipping)')
                    return None

            # Calculate upscaled dimension
            intermediate_dim = tuple(
                int(x / (upscaling_factor ** up)) for x in output_dim)
            # Upscale
            img = deprocess_image(input_img_data[0])
            img = np.array(pil_image.fromarray(img).resize(intermediate_dim,
                                                           pil_image.BICUBIC))
            input_img_data = np.expand_dims(
                process_image(img, input_img_data[0]), 0)

        # decode the resulting input image
        img = deprocess_image(input_img_data[0])
        e_time = time.time()
        print(f'Activation of filter {filter_index:3}: {loss_value:5.0f} (took {(e_time-s_time):4.2f}s)')
        return img, loss_value

    # this is the placeholder for the input images
    assert len(model.inputs) == 1
    input_img = model.inputs[0]

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    output_layer = layer_dict[layer_name]
    assert isinstance(output_layer, layers.Conv2D)

    # Compute to be processed filter range
    n = len(output_layer.get_weights()[1])
    filter_lower, filter_upper = filter_range
    if filter_upper is None:
        filter_upper = n
    else:
        filter_upper = min(filter_upper, n)
    assert(filter_lower >= 0
           and filter_upper > filter_lower)

    # iterate through each filter and generate its corresponding image
    processed_filters = []
    for f in range(filter_lower, filter_upper):
        img_loss = _generate_filter_image(input_img, output_layer.output, f)
        processed_filters.append(img_loss)
    return processed_filters


def save_images_from_layer(model, layer, max_filters, directory):
    name = layer.name
    print(f'Starting layer {name}')

    layer_dir = os.path.join(directory, name)
    if not os.path.exists(layer_dir):
        os.mkdir(layer_dir)

    image_loss_pairs = visualize_layer(model, name, filter_range=(0, max_filters))
    for i, pair in enumerate(image_loss_pairs):
        if pair is not None:
            image, _ = pair
            path = os.path.join(layer_dir, f'{name}_filter{i}.png')
            save_img(path, image)
    print(flush=True)


def main():
    parser = ArgumentParser()
    parser.add_argument('model', type=str, help="Name of custom model to visualize. Or try 'vgg16' to test an ImageNet-pretrained VGG16.")
    parser.add_argument('--all-layers', action='store_true', help='Visualize every layer. By default, only visualizes the last layer.')
    parser.add_argument('--max-filters', type=int, default=None, help='Maximum number of filters per layer to visualize. No limit by default.')
    parser.add_argument('--force', action='store_true', help='Permits overwriting images if they already exist.')
    args = parser.parse_args()

    tf.compat.v1.disable_eager_execution()

    if args.model == 'vgg16':
        # Example VGG16 network with ImageNet weights
        model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False)
    else:
        # Load a custom model
        model = pretrained_models.load_custom(args.model)
    print(model.summary())

    # Create a directory for the images
    model_dir = str(args.model)
    if os.path.exists(model_dir):
        if not args.force:
            print(f"Refusing to write to existing directory '{model_dir}'")
            print('Add --force to override, or remove the directory manually.')
            return
    else:
        os.mkdir(model_dir)

    conv_layers = [l for l in model.layers if isinstance(l, layers.Conv2D)]
    if not args.all_layers:
        # Just visualize the last convolutional layer
        conv_layers = [next(reversed(conv_layers))]

    for l in conv_layers:
        save_images_from_layer(model, l, args.max_filters, model_dir)


if __name__ == '__main__':
    main()
