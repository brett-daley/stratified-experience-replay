from tensorflow.keras.layers import Conv2D, Dense, Flatten


def get_model_fn_by_name(model_name):
    return globals()[model_name]


def atari_cnn():
    return [Conv2D(32, kernel_size=8, strides=4, activation='relu'),
            Conv2D(64, kernel_size=4, strides=2, activation='relu'),
            Conv2D(64, kernel_size=3, strides=1, activation='relu'),
            Flatten(),
            Dense(512, activation='relu')]


def cartpole_mlp():
    return [Dense(512, activation='tanh'),
            Dense(512, activation='tanh')]
