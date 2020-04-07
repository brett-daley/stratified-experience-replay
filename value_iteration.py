import numpy as np


class GridworldEnv:
    def __init__(self):
        mdp = np.loadtxt('gridworld.mdp')

        self.S = S = int(max(mdp[:, 0].max(), mdp[:, 2].max())) + 1
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

    def actions(self):
        return range(self.A)

    def model(self, s1, a, s2):
        return self._model[s1, a, s2]

    def reward(self, s1, a, s2):
        return self._reward[s1, a, s2]


def main():
    env = GridworldEnv()
    discount = 0.9
    values = np.zeros(shape=[env.S])

    for _ in range(100):
        for s in env.states():
            v = values[s]
            print(f'{s:2}: {v:.3f}')
        print()

        old_values = values.copy()

        for s1 in env.states():
            best = -float('inf')
            for a in env.actions():
                avg = 0.0
                for s2 in env.states():
                    avg += env.model(s1, a, s2) * (env.reward(s1, a, s2) + discount * old_values[s2])
                best = max(avg, best)
            values[s1] = best


if __name__ == '__main__':
    main()
