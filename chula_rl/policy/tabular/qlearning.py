import numpy as np
from ..base_policy import BasePolicy


class QlearningPolicy(BasePolicy):
    """q learning"""
    def __init__(self, lr, discount_factor, observation_space, n_action):
        self.lr = lr
        self.discount_factor = discount_factor
        self.observation_space = observation_space
        self.n_action = n_action

        self.q = np.zeros(list(self.observation_space.high) +
                          [n_action])  # (s0, s1, a)

    def step(self, state):
        return np.argmax(self.q[tuple(state)])

    def optimize_step(self, data):
        s, a, r, ss, done = data['s'], data['a'], data['r'], data['ss'], data[
            'done']
        sa = tuple(s) + (a, )  # (s0, s1, a)
        td_error = r + (1.0 - done) * self.discount_factor * np.max(
            self.q[tuple(ss)]) - self.q[sa]
        self.q[sa] += self.lr * td_error
