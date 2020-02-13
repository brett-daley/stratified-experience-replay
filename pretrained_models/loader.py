import tensorflow as tf
from tensorflow.keras.layers import Dense
import os


def load(model_name, unlearn_last_layer):
    if '.h5' not in model_name:
        model_name += '.h5'
    model_path = os.path.join(__package__, model_name)
    model = tf.keras.models.load_model(model_path)

    make_untrainable(model)
    if unlearn_last_layer:
        model = replace_last_layer(model)
    return model


def make_untrainable(model):
    for l in model.layers:
        l.trainable = False


def replace_last_layer(model):
    assert isinstance(model, tf.keras.models.Sequential)
    layers = [l for l in model.layers]
    last_layer = layers.pop()
    assert isinstance(last_layer, tf.keras.layers.Dense)
    layers.append(Dense(last_layer.units, activation=last_layer.activation))
    return tf.keras.models.Sequential(layers)
