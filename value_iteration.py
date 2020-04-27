from argparse import ArgumentParser
import numpy as np
import itertools
import os
from distutils.util import strtobool


class TabularEnv:
    def __init__(self, mdp_file, discount):
        mdp = np.loadtxt(mdp_file)
        self.discount = discount
        self.use_noisy_rewards = True

        # In case states are not consecutive numbers, we'll read them
        # one-by-one and then alias them to consecutive numbers
        self.state_lookup = state_lookup = {}
        next_available = 0
        for s in mdp[:, 0]:
            if s not in state_lookup:
                state_lookup[s] = next_available
                next_available += 1
        self.S = S = len(state_lookup)

        # We can assume actions are always consecutive
        self.A = A = int(mdp[:, 1].max()) + 1

        self._model = np.zeros(shape=(S, A, S))
        self._reward = np.zeros(shape=(S, A, S))
        for s1, a, s2, T, r in mdp:
            s1, a, s2 = state_lookup[s1], int(a), state_lookup[s2]
            self._model[s1, a, s2] = T
            self._reward[s1, a, s2] = r

        # Model sanity check
        for s in self.states():
            for a in self.actions():
                try:
                    assert self._model[s, a, :].sum() == 1.0
                except AssertionError:
                    raise AssertionError(f'T({s}, {a}, :) = {self._model[s, a, :]} does not sum to 1')

    def states(self):
        return range(self.S)

    def is_terminal(self, state):
        return state == (self.S - 1)

    def actions(self):
        return range(self.A)

    def model(self, s1, a):
        return self._model[s1, a]

    def sample_transition(self, s1, a):
        return np.random.choice(self.states(), p=self.model(s1, a))

    def reward(self, s1, a, s2):
        r = self._reward[s1, a, s2]
        if self.use_noisy_rewards and not self.is_terminal(s1):
            r += np.random.normal(scale=0.1)
        return r

    def step(self, s1, a):
        s2 = self.sample_transition(s1, a)
        r = self.reward(s1, a, s2)
        return s2, r


class ValueIterationAgent:
    def __init__(self, env, nstep=1, use_multibatch=False):
        self.env = env

        if use_multibatch:
            self.nstep = 1
            self.multibatch = nstep
        else:
            self.nstep = nstep
            self.multibatch = 1

        self.values = np.zeros(shape=[env.S], dtype=np.float32)
        self.policy = np.random.choice(env.actions(), size=env.S)
        self._copy()
        self.samples = 0  # To measure sample efficiency

    def _copy(self):
        self._old_values = self.values.copy()

    def delta(self):
        assert not (np.allclose(self.values, 0.0) and np.allclose(self._old_values, 0.0))
        return np.abs(self.values - self._old_values).max()

    def update_with_sweep(self):
        for _ in range(self.multibatch):
            self._copy()
            for s in self.env.states():
                returns = [self._full_backup(s, a) for a in self.env.actions()]
                self.values[s] = np.max(returns)
                self.policy[s] = argmax_random_tiebreak(returns)

    def _full_backup(self, s1, a):
        if self.nstep != 1:
            raise NotImplementedError

        returns = np.asarray([self.env.reward(s1, a, s2) + (self.env.discount * self._old_values[s2])
                              for s2 in self.env.states()])
        return (self.env.model(s1, a) * returns).sum()

    def update_with_samples(self, k):
        for _ in range(self.multibatch):
            self._copy()
            for s in self.env.states():
                returns = [self._sample_backup(s, a, k) for a in self.env.actions()]
                self.values[s] = np.max(returns)
                self.policy[s] = argmax_random_tiebreak(returns)

    def _sample_backup(self, s, a, k):
        total = 0.0
        for _ in range(k):
            total += self._sample_nstep_return(s, a)
            self.samples += 1
        return (total / k)

    def _sample_nstep_return(self, s1, a):
        s2 = self.env.sample_transition(s1, a)
        nstep_return = self.env.reward(s1, a, s2)

        for i in range(1, self.nstep):
            s1, a, s2 = self._next(s2)
            nstep_return += pow(self.env.discount, i) * self.env.reward(s1, a, s2)

        nstep_return += pow(self.env.discount, self.nstep) * self._old_values[s2]
        return nstep_return

    def _next(self, s2):
        a = self.policy[s2]
        return s2, a, self.env.sample_transition(s2, a)


def argmax_random_tiebreak(a):
    a = np.asarray(a)
    return np.random.choice(np.flatnonzero(a == a.max()))


def compute_optimal_values(env, precision=1e-9):
    agent = ValueIterationAgent(env)
    env.use_noisy_rewards = False  # Temporarily disable noise
    while True:
        agent.update_with_sweep()
        if agent.delta() < precision:
            break
    env.use_noisy_rewards = True   # Re-enable noise
    return agent.values.copy()


def rms(values1, values2, terminal_state):
    # Compute the root mean squared error
    return np.sqrt(np.mean(np.square(values1 - values2)[:-1]))


def performance(env, agent, start_state, terminal_state, n=100, H=1000):
    total_undisc_return = 0.0
    total_disc_return = 0.0

    for _ in range(n):
        discount = 1.0
        s = start_state
        undisc_return = 0.0
        disc_return = 0.0

        for t in itertools.count():
            if (s == terminal_state) or (t == H):
                break

            a = agent.policy[s]
            s, r = env.step(s, a)

            undisc_return += r
            disc_return += discount * r
            discount *= env.discount

        total_undisc_return += undisc_return
        total_disc_return += disc_return

    return (total_undisc_return / n), (total_disc_return / n)


def normalized_samples(env, agent):
    # Computes the number of samples taken per state-action pair
    SA = (env.S * env.A)
    assert (agent.samples % SA) == 0
    return (agent.samples // SA)


def run(env, start_state, terminal_state,
        nstep, use_multibatch, samples_per_iteration, max_iterations,
        verbose=True):
    optimal_values = compute_optimal_values(env)
    agent = ValueIterationAgent(env, nstep, use_multibatch)

    if verbose:
        print('iteration  normalized_samples  rms  undisc_return  disc_return')

    for i in itertools.count():
        v = agent.values
        e = rms(v, optimal_values, terminal_state)
        undisc_return, disc_return = performance(env, agent, start_state, terminal_state)
        samples = normalized_samples(env, agent)

        print(f'{i}  {samples}  {e:f}  {undisc_return:.2f}  {disc_return:f}')
        if verbose:
            print(list(np.around(v, 3)))
            print(list(agent.policy))
            print()

        if i == max_iterations:
            break

        agent.update_with_samples(k=samples_per_iteration)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('-n', '--nstep', type=int, default=1)
    parser.add_argument('-m', '--multibatch', type=strtobool, default='False')
    parser.add_argument('-s', '--samples', type=int, default=100)
    parser.add_argument('-i', '--iterations', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Metadata that's easiest to just hardcode here
    start_state, terminal_state, discount = {
        'classic_gridworld': (7, 11, 0.9),
        'windy_gridworld': (0, 97, 0.95),
        'windy_gridworld_stochastic': (0, 97, 0.95),
        'whirlpool': (0, 48, 0.99),
    }[args.env_name]

    # Load the MDP file
    mdp_dir = 'gridworlds'
    mdp_file = os.path.join(mdp_dir, args.env_name + '.mdp')
    env = TabularEnv(mdp_file, discount)

    # The env might have aliased the state names, so we must "translate" them here
    start_state = env.state_lookup[float(start_state)]
    terminal_state = env.state_lookup[float(terminal_state)]

    # Ensure that the terminal state is the last state in the array
    assert terminal_state == env.S - 1

    run(env, start_state, terminal_state,
        args.nstep, args.multibatch, args.samples, args.iterations,
        args.verbose)
