import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer
from tensorflow.keras.models import Sequential
import numpy as np
from dqn_utils import schedules
import wandb

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
        self.multiplexer_epsilon_schedule = schedules.LinearAnnealSchedule(start_value=1.0, end_value=0.1, timeframe=300_000)
        self.timestep = -250_000

    def call(self, inputs):
        outputs = tf.stack([m(inputs) for m in self.submodels])
        multiplexer_epsilon = self.multiplexer_epsilon_schedule(self.timestep) if self.timestep >= 0 else 1.0
        self.timestep += 1
        # if self.timestep % 300_000 == 0:
        print(f'Multiplexer Epsilon timestep: \t{self.timestep}')
        print(f'Multiplexer Epsilon value: \t{multiplexer_epsilon}')
        return multiplex(outputs, multiplexer_epsilon=multiplexer_epsilon)


def make_selection_mask(stacked_outputs, all_outputs, multiplexer_epsilon):
    if np.random.rand() < multiplexer_epsilon:
        selection_mask = tf.one_hot(tf.random.uniform(shape=(32,), minval=0, maxval=5, dtype=tf.dtypes.int64),
                                    depth=len(all_outputs))
    else:
        activations = tf.reduce_sum(tf.abs(stacked_outputs), axis=-1)  # result is 5 x 32
        selection_mask = tf.one_hot(tf.argmax(activations, axis=0), depth=len(all_outputs))  # sub. argmax -> tf.randint
    return tf.transpose(selection_mask)[..., None]   # reformatting for matmul


def multiplex(all_outputs, multiplexer_epsilon):
    stacked_outputs = tf.stack(all_outputs)
    selection_mask = make_selection_mask(stacked_outputs, all_outputs, multiplexer_epsilon)
    return tf.reduce_sum(selection_mask * stacked_outputs, axis=0) # eliminates the now-0s

# def multiplex(all_outputs):
#     stacked_outputs = tf.stack(all_outputs)
#     # activations = tf.reduce_sum(tf.abs(stacked_outputs), axis=-1)  # result is 5 x 32
#     # selection_mask = tf.one_hot(tf.argmax(activations, axis=0), depth=len(all_outputs))  # sub. argmax -> tf.randint
#     selection_mask = tf.one_hot(tf.random.uniform(shape=(32,), minval=0, maxval=5, dtype=tf.dtypes.int64),
#                                 depth=len(all_outputs))
#     selection_mask = tf.transpose(selection_mask)[..., None]   # reformatting for matmul
#     return tf.reduce_sum(selection_mask * stacked_outputs, axis=0) # eliminates the now-0s
