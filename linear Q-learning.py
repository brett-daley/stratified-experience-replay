import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


#Put everything in function "main" and run below.
def main():
    episode_rewards, episode_lengths = cartpole_playtolearn(eps=0.5, timesteps=600, alpha=0.01, gamma=0.9)
    plot_running_avg(episode_rewards)


# Create CartPole instance
env = gym.make("CartPole-v0")


def Qnet_forward(state_vector):
    # Input is 4x1 state vector [cart pos., cart vel., pole angle, pole vel.]
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(4, input_dim=1, activation='linear'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))

    # Output is q value vector for "left" and "right" given input state
    model_input = state_vector
    model_output = model.predict(model_input)

    return model_output, model


def Qnet_backprop(model, alpha, state_vector, target_output):

    sgd = tf.keras.optimizers.SGD(lr=alpha)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    model.fit(state_vector, target_output)

    return


def cartpole_playtolearn(eps=0.5, timesteps=60000, alpha=0.01, gamma=0.9):
    game_reward_list = []
    game_length_list = []

    env.reset()
    state = np.array([0, 0, 0, 0])         # sets initial state with all values at zero
    total_reward = 0
    count = 0

    for n in range(timesteps):

        Q_vals_for_state, model = Qnet_forward(state)

        if np.random.uniform() < eps:
            act = env.action_space.sample() # epsilon-greedy
        else:
            act = np.argmax(Q_vals_for_state)

        observation, reward, done, info = env.step(act)
        total_reward += reward
        count += 1

        if done: # restart game
            env.reset()
            state = np.array([0, 0, 0, 0])
            game_reward_list.append(total_reward)
            game_length_list.append(count)
            continue

        Q_vals_new_state, model_new = Qnet_forward(observation)
        Q_target_vector = Q_vals_for_state  # Keep Q vals the same for all actions agent didn't pick
        Q_target_vector[act,0] = reward + gamma * np.max(Q_vals_new_state) # Only update q-val for action taken

        Qnet_backprop(model, alpha, Q_vals_for_state, target_output=Q_target_vector)

        state = observation

        if n % 5000 == 0:                   # So we know what game we are on
            print("{} time steps have elapsed".format(n))
            print("Most recent cartpole score was {}".format(game_reward_list[-1]))
            print("Most recent game length was {}".format(game_length_list[-1]))

        if n % 100 == 0:
            eps = 1.0 / np.sqrt(n/100 + 1)  # Ensure that epsilon decreases as the training progresses

    return game_reward_list, game_length_list


def plot_running_avg(totalrewards):
    N = len(totalrewards)
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = np.mean(totalrewards[max(0, t-100):(t+1)])

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()


if __name__ == '__main__':
    main()
