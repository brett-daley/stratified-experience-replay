import tensorflow as tf
from tensorflow.keras.layers import Dense


def append_layers(model, new_layers):
    layers = [l for l in model.layers]
    try:
        return tf.keras.models.Sequential(layers + new_layers)
    except TypeError:
        return tf.keras.models.Sequential(layers + [new_layers])


def make_untrainable(model):
    for l in model.layers:
        l.trainable = False


def replace_last_layer(model, units):
    layers = [l for l in model.layers]
    last_layer = layers.pop()
    assert isinstance(last_layer, Dense)
    layers.append(Dense(units))
    return tf.keras.models.Sequential(layers)
