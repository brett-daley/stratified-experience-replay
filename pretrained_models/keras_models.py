import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten

from pretrained_models import model_editor


def vgg16(input_shape, num_actions):
    model = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False)
    return model_editor.append_layers(model, new_layers=[Flatten(), Dense(num_actions)])
