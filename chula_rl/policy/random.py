import numpy as np

from .base_policy import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, action_fn):
        self.action_fn = action_fn

    def step(self, state):
        return self.action_fn()

    def optimize_step(self, data):
        pass


class VecRandomPolicy(BasePolicy):
    def __init__(self, action_fn):
        self.action_fn = action_fn

    def step(self, state):
        n_env = state.shape[0]
        return np.array([self.action_fn() for i in range(n_env)])

    def optimize_step(self, data):
        pass
