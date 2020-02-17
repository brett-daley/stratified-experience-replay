from gym.wrappers import Monitor
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def exploration_schedule(t, timeframe=2_000, min_epsilon=0.1):
    return max(1.0 - (1.0 - min_epsilon) * (t / timeframe), min_epsilon)


def train(env, model, timesteps=5_000, gamma=0.99):
    env = Monitor(env, directory='monitor', force=True, video_callable=lambda e: False)
    state = env.reset()

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    model.build(input_shape=(None, *env.observation_space.shape))
    # TODO: technically, this shouldn't be mean squared error
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    print(model.summary())

    print('timestep', 'episode', 'avg_return', 'epsilon', sep='  ', flush=True)
    for t in range(timesteps+1):
        epsilon = exploration_schedule(t)

        if t % 50 == 0:
            rewards = env.get_episode_rewards()
            avg_return = np.mean(rewards[-100:])
            num_episodes = len(rewards)
            print(t, num_episodes, avg_return, epsilon, sep='  ', flush=True)

        if t == timesteps:
            break  # End training after logging the final timestep

        # Compute current Q-values
        qvalues = model.predict(state[None])

        # Select action according to epsilon-greedy policy
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qvalues)

        # Step the environment
        next_state, reward, done, _ = env.step(action)

        # Reset the environment if done; compute the target Q-value accordingly
        target = reward
        if done:
            next_state = env.reset()
        else:
            # If we're not done, bootstrap from the next state
            next_qvalues = model.predict(next_state[None])
            target += gamma * np.max(next_qvalues)
        qvalues[0, action] = target  # Only update Q-value for action taken

        # Take a gradient step
        model.train_on_batch(state[None], qvalues)
        state = next_state

    plot_running_avg(env)
    env.close()


def plot_running_avg(env):
    rewards = env.get_episode_rewards()
    N = len(rewards)
    running_avg = np.empty(N)

    for t in range(N):
        running_avg[t] = np.mean(rewards[max(0, t-100):(t+1)])

    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()
