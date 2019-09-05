import random

from chula_rl.policy.base_policy import BasePolicyWrapper


class EpsilonGreedy(BasePolicyWrapper):
    def __init__(self, policy, eps: float, n_action: int):
        super().__init__(policy)
        self.eps = eps
        self.n_action = n_action

    def step(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.n_action - 1)
        else:
            return self.policy.step(state)