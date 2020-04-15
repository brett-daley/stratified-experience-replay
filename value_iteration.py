from argparse import ArgumentParser
import numpy as np
import itertools
import os

import gridworlds


class TabularEnv:
    def __init__(self, mdp_file):
        mdp = np.loadtxt(mdp_file)

        self.S = S = int(max(np.maximum(mdp[:, 0], mdp[:, 2]))) + 1
        self.A = A = int(mdp[:, 1].max()) + 1

        self._model = np.zeros(shape=(S, A, S))
        self._reward = np.zeros(shape=(S, A, S))
        for s1, a, s2, T, r in mdp:
            s1, a, s2 = int(s1), int(a), int(s2)
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

    def random_state(self):
        return np.random.randrange(self.S)

    def actions(self):
        return range(self.A)

    def model(self, s1, a):
        return self._model[s1, a]

    def sample_transition(self, s1, a):
        return np.random.choice(self.states(), p=self.model(s1, a))

    def reward(self, s1, a, s2):
        return self._reward[s1, a, s2]

    def step(self, s1, a):
        s2 = self.sample_transition(s1, a)
        r = self.reward(s1, a, s2)
        return s2, r


class ValueIterationAgent:
    def __init__(self, env, nstep, discount):
        self.env = env
        self.nstep = nstep
        self.discount = discount
        self.values = np.zeros(shape=[env.S], dtype=np.float32)
        self.policy = np.random.choice(env.actions(), size=env.S)
        self._copy()

    def _copy(self):
        self._old_values = self.values.copy()

    def delta(self):
        assert not (np.allclose(self.values, 0.0) and np.allclose(self._old_values, 0.0))
        return np.abs(self.values - self._old_values).max()

    def update_with_sweep(self):
        self._copy()
        for s in self.env.states():
            returns = [self._full_backup(s, a) for a in self.env.actions()]
            self.values[s] = np.max(returns)
            self.policy[s] = np.argmax(returns)

    def _full_backup(self, s1, a):
        if self.nstep != 1:
            raise NotImplementedError

        returns = np.asarray([self.env.reward(s1, a, s2) + (self.discount * self._old_values[s2])
                              for s2 in self.env.states()])
        return (self.env.model(s1, a) * returns).sum()

    def update_with_samples(self, k):
        self._copy()
        for s in self.env.states():
            returns = [self._sample_backup(s, a, k) for a in self.env.actions()]
            self.values[s] = np.max(returns)
            self.policy[s] = np.argmax(returns)

    def _sample_backup(self, s, a, k):
        total = 0.0
        for _ in range(k):
            total += self._sample_nstep_return(s, a)
        return (total / k)

    def _sample_nstep_return(self, s1, a):
        s2 = self.env.sample_transition(s1, a)
        nstep_return = self.env.reward(s1, a, s2)

        for i in range(1, self.nstep):
            s1, a, s2 = self._next(s2)
            nstep_return += pow(self.discount, i) * self.env.reward(s1, a, s2)

        nstep_return += pow(self.discount, self.nstep) * self._old_values[s2]
        return nstep_return

    def _next(self, s2):
        a = self.policy[s2]
        return s2, a, self.env.sample_transition(s2, a)


def compute_optimal_values(mdp_file, discount, precision=1e-9):
    env = TabularEnv(mdp_file)
    agent = ValueIterationAgent(env, nstep=1, discount=discount)

    while True:
        agent.update_with_sweep()
        if agent.delta() < precision:
            break
    return agent.values.copy()


def mse(values1, values2, terminal_state):
    # Ensure that the terminal state is the last state in the array
    assert terminal_state == (len(values1) - 1) == (len(values2) - 1)
    # Compute the mean squared error
    return np.mean(np.square(values1 - values2)[:-1])


def performance(env, agent, start_state, terminal_state, n=100, H=1000):
    total_return = 0.0

    for _ in range(n):
        discount = 1.0
        s = start_state
        discounted_return = 0.0

        for t in itertools.count():
            a = agent.policy[s]
            s, r = env.step(s, a)
            discounted_return += discount * r
            discount *= agent.discount
            if (s == terminal_state) or (t == H):
                break

        total_return += discounted_return

    return (total_return / n)


def run(mdp_file, start_state, terminal_state, discount,
        nstep, samples_per_iteration, max_iterations,
        verbose=True):
    optimal_values = compute_optimal_values(mdp_file, discount)

    env = TabularEnv(mdp_file)
    agent = ValueIterationAgent(env, nstep, discount)

    if verbose:
        print('iteration  mse  avg_return')

    for i in itertools.count():
        v = agent.values
        e = mse(v, optimal_values, terminal_state)
        p = performance(env, agent, start_state, terminal_state)

        print(f'{i}  {e:f}  {p:f}')
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
    parser.add_argument('-s', '--samples', type=int, default=100)
    parser.add_argument('-i', '--iterations', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Metadata that's easiest to just hardcode here
    start_state, terminal_state, discount = {
        'gridworld': (7, 11, 0.9),
        'whirlpool': (0, 48, 0.99),
    }[args.env_name]

    # Load the MDP file
    mdp_dir = 'gridworlds'
    mdp_file = os.path.join(mdp_dir, args.env_name + '.mdp')

    run(mdp_file, start_state, terminal_state, discount,
        args.nstep, args.samples, args.iterations,
        args.verbose)
