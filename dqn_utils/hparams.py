from tensorflow.keras import optimizers

from dqn_utils.envs import atari_env, toytext_env
from dqn_utils.models import get_model_fn_by_name
from dqn_utils.replay_memory import ReplayMemory
from dqn_utils import schedules


def get_hparams(env_name):
    if env_name in atari_env.ALL_GAMES:
        return atari_hparams()
    if env_name in toytext_env.ALL_ENVS:
        return toytext_hparams()
    if env_name == 'CartPole-v0':
        return cartpole_hparams()
    raise ValueError(f'environment {env_name} has no hyperparameters assigned')


def atari_hparams():
    return {
        'discount': 0.99,
        'epsilon_schedule': schedules.ConstantSchedule(0.1),
        'model_fn': get_model_fn_by_name('atari_cnn'),
        'optimizer': optimizers.Adam(learning_rate=1e-4, epsilon=1e-4),
        'prepopulate': 250_000,
        'rmem_constructor': lambda env: ReplayMemory(env, batch_size=32, capacity=1_000_000, history_len=4),
        'scale_obs': 1.0 / 255.0,
        'update_freq': 10_000,
    }


def toytext_hparams():
    return {
        'discount': 0.99,
        'epsilon_schedule': schedules.LinearAnnealSchedule(start_value=1.0, end_value=0.1, timeframe=600_000),
        'model_fn': get_model_fn_by_name('cartpole_mlp'),
        'optimizer': optimizers.Adam(learning_rate=1e-4, epsilon=1e-4),
        'prepopulate': 50_000,
        'rmem_constructor': lambda env: ReplayMemory(env, batch_size=32, capacity=500_000),
        'scale_obs': 1.0,
        'update_freq': 10_000,
    }


def cartpole_hparams():
    return {
        'discount': 0.99,
        'epsilon_schedule': schedules.LinearAnnealSchedule(start_value=1.0, end_value=0.1, timeframe=300_000),
        'model_fn': get_model_fn_by_name('cartpole_mlp'),
        'optimizer': optimizers.Adam(learning_rate=1e-4, epsilon=1e-8),
        'prepopulate': 50_000,
        'rmem_constructor': lambda env: ReplayMemory(env, batch_size=32, capacity=500_000),
        'scale_obs': 1.0,
        'update_freq': 10_000,
    }
