import numpy as np
from ..base_policy import BasePolicy


def calculate_return(r, discount_factor):
    """return G for every time step given a sequence of rewards"""
    n = len(r)
    g = np.zeros(r.shape)
    current = 0
    for i in reversed(range(0, n)):
        current = r[i] + current * discount_factor
        g[i] = current
    return g


def first_sa(s, a, g):
    """deduplicate (s, a) keeping only the first occurrances while also matching the corresponding returns"""
    a = np.expand_dims(a, 1)
    sa = np.hstack((s, a))
    sa, i = np.unique(sa, return_index=True, axis=0)
    sa = (sa[:, 0], sa[:, 1], sa[:, 2])
    g = g[i]
    # return unique sa and g
    # sa = tuple(first dim of s, second dim of s, a)
    # this is for numpy indexing!
    # ex: sa = ([0, 0], [0, 1], [1, 1])
    # means: s = [(0, 0), (0, 1)]; a = [1, 1]
    return sa, g


class MonteCarloPolicy(BasePolicy):
    """firt visit monte carlo with true average"""
    def __init__(self, discount_factor, observation_space, n_action):
        self.discount_factor = discount_factor
        self.observation_space = observation_space
        self.n_action = n_action
        # value tables
        self.q = np.zeros(list(self.observation_space.high) +
                          [n_action])  # (s0, s1, a)
        self.cnt = np.zeros(self.q.shape, dtype=int)

    def step(self, state):
        return np.argmax(self.q[tuple(state)])  # greedy action selection

    def optimize_step(self, data):
        # evaluation (prediction)
        s = np.array(data['s'])
        a = np.array(data['a'])
        r = np.array(data['r'])
        g = calculate_return(r, self.discount_factor)

        # value error (VE)
        sa, g = first_sa(s, a, g)
        error = g - self.q[sa]

        # average
        self.cnt[sa] += 1
        # this is just a normal average
        self.q[sa] += 1 / self.cnt[sa] * error