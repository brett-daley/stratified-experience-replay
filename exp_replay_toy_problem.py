import numpy as np
import matplotlib.pyplot as plt
import itertools

DISCOUNT = 0.9
ITERATIONS = 100


def linreg(x, y, w):
    """Conducts of a linear regression of y vs x, weighted by w."""
    x, y, w = map(np.asarray, [x, y, w])

    sum_x = (w * x).sum()
    sum_y = (w * y).sum()
    sum_x2 = (w * np.square(x)).sum()
    sum_xy = np.sum(w * x * y)
    n = w.sum()

    # m: slope of line
    m = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - np.square(sum_x))
    # b: y-intercept of line
    b = (sum_y - m*sum_x) / n

    return m, b


def value(s, q_values):
    """Returns the value of state s: i.e. max_a Q(s,a)."""
    return max(q_values[s,0], q_values[s,1])


def get_optimal_values():
    """Manually compute the Q-values of the optimal policy."""
    optimal_values = np.zeros(shape=(3,2), dtype=np.float64)
    optimal_values[0,0] = DISCOUNT / (1.0 - DISCOUNT)
    optimal_values[0,1] = DISCOUNT**2 / (1.0 - DISCOUNT)
    optimal_values[1,0] = DISCOUNT / (1.0 - DISCOUNT)
    optimal_values[1,1] = DISCOUNT**2 / (1.0 - DISCOUNT)
    optimal_values[2,0] = 1.0 / (1.0 - DISCOUNT)
    optimal_values[2,1] = DISCOUNT**2 / (1.0 - DISCOUNT)
    return optimal_values


def backup(q_values):
    """Conducts a theoretical 1-step Bellman backup for our MDP."""
    new_values = q_values.copy()
    V = lambda s: value(s, q_values)

    new_values[0,0] = 0.0 + DISCOUNT * V(2)
    new_values[0,1] = 0.0 + DISCOUNT * V(0)
    new_values[1,0] = 0.0 + DISCOUNT * V(2)
    new_values[1,1] = 0.0 + DISCOUNT * V(1)
    new_values[2,0] = 1.0 + DISCOUNT * V(2)
    new_values[2,1] = 0.0 + DISCOUNT * (V(0) + V(1)) / 2.0
    return new_values


def q_learning_update(q_values):
    """Conducts a Q-Learning update asssuming an infinitely large replay memory."""
    return backup(q_values)


def rms(q_values):
    """Returns the root-mean-square error of Q with respect to Q*."""
    return np.sqrt(np.mean(np.square(q_values - get_optimal_values())))


def dqn_update(q_values, params, p):
    """Conducts a DQN update assuming an infinitely large replay memory."""
    new_values = q_values.copy()
    target_values = backup(q_values)

    # Now we need to perform a weighted LS regression
    weights = q_values.copy()
    weights[0,0] = 0.5*(1-p) * p
    weights[0,1] = 0.5*(1-p) * (1-p)
    weights[1,0] = 0.5*(1-p) * p
    weights[1,1] = 0.5*(1-p) * (1-p)
    weights[2,0] = p * p
    weights[2,1] = p * (1-p)
    assert np.isclose(weights.sum(), 1.0)

    for a in range(2):
        x = np.arange(3, dtype=np.float64)
        y = target_values[:, a]
        w = weights[:, a]
        # w = np.ones_like(y)

        m, b = linreg(x, y, w)
        params[a,0] = m
        params[a,1] = b

        for s in range(3):
            new_values[s,a] = m*s + b

    # WARNING: uncommenting this line overrides the linear function with tabular values
    # new_values = q_learning_update(q_values)

    return new_values, params


def test_p_value(update_rule, p):
    q_values = np.zeros(shape=(3,2), dtype=np.float64)
    params = np.zeros(shape=(2,2), dtype=np.float64)

    X, Y = [], []
    for i in itertools.count():
        error = rms(q_values)
        X.append(i)
        Y.append(error)
        if i == ITERATIONS:
            break
        q_values, params = update_rule(q_values, params, p)

    return params[0]


def main():
    dqn = lambda q_values, params, p: dqn_update(q_values, params, p)

    X = []
    Y = []
    for p in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        x = test_p_value(dqn, p)
        print(p, x)
        X.append(x[0])
        Y.append(x[1])

    plt.plot(X, Y)
    plt.plot(0.5, 8.8333, 'ro')
    plt.xlim([0,1])
    plt.ylim([0,10])
    plt.savefig('figure.png')


if __name__ == '__main__':
    main()
