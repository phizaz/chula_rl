from chula_rl.policy.base_policy import BasePolicy
from typing import Dict
import gym

import numpy as np
from collections import defaultdict, deque


class BaseExplorer:
    def __init__(self, env: gym.Env):
        self.env = env

        self.hist = defaultdict(list)
        self.mean_reward = deque(maxlen=10)

    def get_stats(self):
        returns = np.array(self.mean_reward)
        return {
            'history': self.hist,
            'reward': returns.mean(),
            'reward:q1': np.quantile(returns, q=0.1),
            'reward:q9': np.quantile(returns, q=0.9),
        }

    def _update_stats(self, n_interaction, reward):
        """keep track of rewards and history"""
        self.hist['n_interaction'].append(n_interaction)
        self.hist['reward'].append(reward)
        self.mean_reward.append(reward)

    def step(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()
