import numpy as np


class TabularEnv:
    def __init__(self, mdp_path):
        mdp = np.loadtxt(mdp_path)

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

    def model(self, s1, a, s2):
        return self._model[s1, a, s2]

    def model_distr(self, s1, a):
        return [self._model[s1, a, s2] for s2 in self.states()]

    def reward(self, s1, a, s2):
        return self._reward[s1, a, s2]


class ValueIterationAgent:
    def __init__(self, env, discount=0.9):
        self.env = env
        self.discount = discount
        self._values = np.zeros(shape=[env.S], dtype=np.float32)
        self._policy = np.zeros(shape=[env.S], dtype=np.int32)

    def _copy(self):
        self._old_values = self._values.copy()

    def update_with_sweep(self):
        self._copy()

        for s1 in self.env.states():
            best = -float('inf')
            for a in self.env.actions():
                avg = 0.0
                for s2 in self.env.states():
                    avg += self.env.model(s1, a, s2) * (self.env.reward(s1, a, s2) + self.discount * self._old_values[s2])
                best = max(avg, best)
            self._values[s1] = best

    def update_with_samples(self, n):
        self._copy()

        for s1 in self.env.states():
            best = -float('inf')
            for a in self.env.actions():
                avg = 0.0
                distr = self.env.model_distr(s1, a)
                for _ in range(n):
                    s2 = np.random.choice(np.arange(self.env.S), p=distr)
                    avg += self.env.reward(s1, a, s2) + self.discount * self._old_values[s2]
                avg /= n
                best = max(avg, best)
            self._values[s1] = best


def main():
    env = TabularEnv('gridworld.mdp')
    agent = ValueIterationAgent(env)

    for _ in range(100):
        for s in env.states():
            v = agent._values[s]
            print(f'{s:2}: {v:.3f}')
        print()

        # agent.update_with_sweep()
        agent.update_with_samples(100)


if __name__ == '__main__':
    main()
