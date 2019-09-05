import numpy as np
from ..base_policy import BasePolicy


class SARSAPolicy(BasePolicy):
    """one-step sarsa with policy improvement (immediately)"""
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
        # evaluation (prediction)
        s, a, r, ss, aa, done = data['s'], data['a'], data['r'], data[
            'ss'], data['aa'], data['done']
        sa = tuple(s) + (a, )  # (s0, s1, a)
        ssaa = tuple(ss) + (aa, )
        td_error = r + (
            1.0 - done) * self.discount_factor * self.q[ssaa] - self.q[sa]
        self.q[sa] += self.lr * td_error


class SARSATrueAvgPolicy(BasePolicy):
    """one-step sarsa with policy improvement (immediately), 
    with true average."""
    def __init__(self, discount_factor, observation_space, n_action):
        self.discount_factor = discount_factor
        self.observation_space = observation_space
        self.n_action = n_action

        self.q = np.zeros(list(self.observation_space.high) +
                          [n_action])  # (s0, s1, a)
        self.cnt = np.zeros(self.q.shape, dtype=int)

    def step(self, state):
        return np.argmax(self.q[tuple(state)])

    def optimize_step(self, data):
        # evaluation (prediction)
        s, a, r, ss, aa, done = data['s'], data['a'], data['r'], data[
            'ss'], data['aa'], data['done']
        sa = tuple(s) + (a, )  # (s0, s1, a)
        ssaa = tuple(ss) + (aa, )
        td_error = r + (
            1.0 - done) * self.discount_factor * self.q[ssaa] - self.q[sa]

        self.cnt[sa] += 1
        self.q[sa] += 1 / self.cnt[sa] * td_error


class ExpectedSARSAPolicy(BasePolicy):
    """one-step expected sarsa with policy improvement (immediately)"""
    def __init__(self, lr, discount_factor, observation_space, n_action):
        self.lr = lr
        self.discount_factor = discount_factor
        self.observation_space = observation_space
        self.n_action = n_action

        self.q = np.zeros(list(self.observation_space.high) +
                          [n_action])  # (s0, s1, a)

        self.policy = None

    def set_policy(self, policy: BasePolicy):
        """expected SARSA needs to know the true policy which might not be greedy
        i.e. an episilon greedy mask is applied."""
        self.policy = policy

    def step(self, state):
        return np.argmax(self.q[tuple(state)])

    def optimize_step(self, data):
        # evaluation (prediction)
        s, a, r, ss, done = data['s'], data['a'], data['r'], data['ss'], data[
            'done']
        sa = tuple(s) + (a, )  # (s0, s1, a)
        assert self.policy is not None, "please set_policy first"
        ssaa = tuple(ss) + (self.policy.step(ss), )
        td_error = r + (
            1.0 - done) * self.discount_factor * self.q[ssaa] - self.q[sa]
        self.q[sa] += self.lr * td_error
