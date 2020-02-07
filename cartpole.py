import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

"""
NOTE: Code root referenced from lecture from "Machine Learning with Phil":
https://www.youtube.com/channel/UC58v9cLitc8VaCjrcKyAbrw
"""

# Create CartPole instance
env = gym.make("CartPole-v0")

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
    total_reward = 0

    while not done:
        count += 1
        if np.random.uniform() < eps:
            act = env.action_space.sample()  # epsilon-greedy
        # if tf.random.uniform() < eps:
        #     act = env.action_space.sample() # epsilon-greedy
        else:
            act = max_dict(Q[state])[0]

        observation, reward, done, info = env.step(act)

        total_reward += reward

        if done and count < 200:
            reward = -400

        state_new = get_state_as_string(assign_bins(observation, bins))

        a1, max_q_s1a1 = max_dict(Q[state_new])
        Q[state][act] += ALPHA * (reward + GAMMA * max_q_s1a1 - Q[state][act])
        state, act = state_new, a1

    return total_reward, count


def play_many_games(bins, N=6000):
    Q = initialize_Q()

    length = []
    reward = []
    for n in range(N):
        # ensure that epsilon decreases as the game continues
        eps = 1.0 / np.sqrt(n+1)
        # eps = 1.0 / tf.sqrt(float(n) + 1)
        episode_reward, episode_length = play_one_game(bins, Q, eps)

        if n % 100 == 0:
            print(n, '%.4f' % eps, episode_reward) # so we know roughly what game we are on
        length.append(episode_length)
        reward.append(episode_reward)

    return length, reward


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)
    # running_avg = tf.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])
        # running_avg[t] = tf.mean(totalrewards[max(0, t - 100):(t + 1)])
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == '__main__':
    bins = create_bins()
    episode_lengths, episode_rewards = play_many_games(bins)
    plot_running_avg(episode_rewards)
