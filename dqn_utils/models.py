import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer
from tensorflow.keras.models import Sequential


def get_model_fn_by_name(model_name):
    return globals()[model_name]


def atari_cnn():
    return [Conv2D(32, kernel_size=8, strides=4, activation='relu'),
            Conv2D(64, kernel_size=4, strides=2, activation='relu'),
            Conv2D(64, kernel_size=3, strides=1, activation='relu'),
            Flatten(),
            Dense(512, activation='relu')]


def atari_cnn_5mux():
    submodel_fn = lambda: Sequential(
                    [Conv2D(32, kernel_size=8, strides=4, activation='relu'),
                     Conv2D(64, kernel_size=4, strides=2, activation='relu'),
                     Conv2D(64, kernel_size=3, strides=1, activation='relu'),
                     Flatten(),
                     Dense(83, activation='relu')])
    return [Multiplexer(submodel_fn, n=5)]


def cartpole_mlp():
    return [Dense(512, activation='tanh'),
            Dense(512, activation='tanh')]

def frozenlake_mlp():
    return [Dense(512, activation='tanh'),
            Dense(512, activation='tanh')]

def cartpole_mlp_2mux():
    submodel_fn = lambda: Sequential([Dense(361, activation='tanh') for _ in range(2)])
    return [Multiplexer(submodel_fn, n=2)]


def cartpole_mlp_3mux():
    submodel_fn = lambda: Sequential([Dense(295, activation='tanh') for _ in range(2)])
    return [Multiplexer(submodel_fn, n=3)]


def cartpole_mlp_10mux():
    submodel_fn = lambda: Sequential([Dense(160, activation='tanh') for _ in range(2)])
    return [Multiplexer(submodel_fn, n=10)]


class Multiplexer(tf.keras.Model):
    def __init__(self, submodel_fn, n):
        super().__init__()
        self.submodels = [submodel_fn() for _ in range(n)]
        self.n = n

    def call(self, inputs):
        outputs = tf.stack([m(inputs) for m in self.submodels])
        return multiplex(outputs)


def multiplex(all_outputs):
    stacked_outputs = tf.stack(all_outputs)
    activations = tf.reduce_sum(tf.abs(stacked_outputs), axis=-1)
    selection_mask = tf.one_hot(tf.argmax(activations, axis=0), depth=len(all_outputs))
    selection_mask = tf.transpose(selection_mask)[..., None]
    return tf.reduce_sum(selection_mask * stacked_outputs, axis=0)
