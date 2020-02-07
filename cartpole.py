import gym
from gym.wrappers import Monitor
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
NOTE: Code root referenced from lecture from "Machine Learning with Phil":
https://www.youtube.com/channel/UC58v9cLitc8VaCjrcKyAbrw
"""

# Create CartPole instance
env = gym.make("CartPole-v0")
# Wrap env with a monitor. Always overwrite the last run, and don't save videos.
env = Monitor(env, directory='monitor', force=True, video_callable=lambda e: False)

# Set global variables
MAXSTATES = 10 ** 4
GAMMA = 0.9
ALPHA = 0.01


# Create dictionary to find max elements of Q array
def max_dict(d):
    max_v = float('-inf')
    for key, val in d.items():
        if val > max_v:
            max_v = val
            max_key = key
    return max_key, max_v


# Discretize state space
def create_bins():
    # obs[0]: cart position ranging from -4.8 to +4.8
    # obs[1]: cart velocity ranging from negative infinity to positive infinity
    # obs[2]: pole angle ranging from -41.8 to +41.8
    # obs[3]: pole velocity (at top) ranging from negative inf to positive inf

    bins = np.zeros((4, 10))
    bins[0] = np.linspace(-4.8, 4.8, 10)
    bins[1] = np.linspace(-5, 5, 10)
    bins[2] = np.linspace(-.418, .418, 10)
    bins[3] = np.linspace(-5, 5, 10)
    # bins = tf.zeros((4,10))
    # bins[0] = tf.linspace(-4.8, 4.8, 10)
    # bins[1] = tf.linspace(-5., 5., 10)
    # bins[2] = tf.linspace(-.418, .418, 10)
    # bins[3] = tf.linspace(-5., 5., 10)

    return bins


def assign_bins(observation, bins):
    state = np.zeros(4)
    # state = tf.zeros(4)
    for i in range(4):
        state[i] = np.digitize(observation[i], bins[i])
        # state[i] = tf.python.ops.math_ops._bucketize(observation[i], bins[i])
    return state


def get_state_as_string(state):
    string_state = ''.join(str(int(e)) for e in state)
    return string_state


def get_all_states_as_string():
    states = []
    for i in range(MAXSTATES):
        states.append(str(i).zfill(4))
    return states


def initialize_Q():
    Q = {}

    all_states = get_all_states_as_string()
    for state in all_states:
        Q[state] = {}
        for action in range(env.action_space.n):
            Q[state][action] = 0
    return Q


def play_one_game(bins, Q, eps=0.5):
    observation = env.reset()
    done = False
    count = 0  # number of moves in an episode
    state = get_state_as_string(assign_bins(observation, bins)) # initializes state randomly

    while not done:
        count += 1
        if np.random.uniform() < eps:
            act = env.action_space.sample()  # epsilon-greedy
        # if tf.random.uniform() < eps:
        #     act = env.action_space.sample() # epsilon-greedy
        else:
            act = max_dict(Q[state])[0]

        observation, reward, done, _ = env.step(act)

        if done and count < 200:
            reward = -400

        state_new = get_state_as_string(assign_bins(observation, bins))

        a1, max_q_s1a1 = max_dict(Q[state_new])
        Q[state][act] += ALPHA * (reward + GAMMA * max_q_s1a1 - Q[state][act])
        state, act = state_new, a1


def play_many_games(bins, N=6000):
    Q = initialize_Q()

    for n in range(N):
        # ensure that epsilon decreases as the game continues
        eps = 1.0 / np.sqrt(n+1)
        # eps = 1.0 / tf.sqrt(float(n) + 1)
        play_one_game(bins, Q, eps)

        if n % 100 == 0:
            episode_reward = env.get_episode_rewards()[-1]
            print(n, '%.4f' % eps, episode_reward, flush=True) # so we know roughly what game we are on


def plot_running_avg():
    rewards = env.get_episode_rewards()
    N = len(rewards)
    running_avg = np.empty(N)
    # running_avg = tf.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(rewards[max(0, t-100):(t+1)])
        # running_avg[t] = tf.mean(totalrewards[max(0, t - 100):(t + 1)])
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == '__main__':
    bins = create_bins()
    play_many_games(bins)
    plot_running_avg()
