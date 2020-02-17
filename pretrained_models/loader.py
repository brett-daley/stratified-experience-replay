import tensorflow as tf
import os

from pretrained_models import keras_models, model_editor


def load_custom(model_name, num_actions=None):
    model_path = os.path.join(__package__, model_name)
    model = tf.keras.models.load_model(model_path + '.h5')

    model_editor.make_untrainable(model)

    if not num_actions:
        # If the number of actions is not specified, attempt to use the model as-is.
        return model
    return model_editor.replace_last_layer(model, units=num_actions)


def load_keras(model_name, input_shape, num_actions):
    return getattr(keras_models, model_name)(input_shape, num_actions)
