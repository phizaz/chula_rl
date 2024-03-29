from collections import defaultdict, deque

import numpy as np

import gym
from chula_rl.policy.base_policy import BasePolicy
from chula_rl.exception import *

from .base_explorer import BaseExplorer


class EpisodeExplorer(BaseExplorer):
    """explore for an episode"""
    def __init__(self, n_max_interaction: int, env: gym.Env):
        super().__init__(env)
        self.n_max_interaction = n_max_interaction

        self.last_s = self.env.reset()

        self.n_interaction = 0
        self.n_ep = 0

    def step(self, policy: BasePolicy):
        data = defaultdict(list)
        while True:
            if self.n_interaction >= self.n_max_interaction:
                raise InteractionExceeded()

            # explore
            a = policy.step(self.last_s)
            s, r, done, info = self.env.step(a)
            self.n_interaction += 1

            # collect data
            data['s'].append(self.last_s)
            data['a'].append(a)
            data['r'].append(r)
            data['done'].append(done)

            self.last_s = s

            # if done reset
            if done:
                self.last_s = self.env.reset()
                self.n_ep += 1
                self._update_stats(self.n_interaction,
                                   info['episode']['reward'])
                return data
