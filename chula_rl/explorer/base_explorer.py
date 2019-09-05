from chula_rl.policy.base_policy import BasePolicy
from typing import Dict
import gym

import numpy as np
from collections import defaultdict, deque


class BaseExplorer:
    def __init__(self, env: gym.Env):
        self.env = env

        self.hist = defaultdict(list)
        self.mean_return = deque(maxlen=10)

    def get_stats(self):
        # reward means total reward of an episode
        rewards = np.array(self.mean_return)
        return {
            'reward': rewards.mean() if len(rewards) > 0 else 0.0,
            'reward:q1':
            np.quantile(rewards, q=0.1) if len(rewards) > 0 else 0.0,
            'reward:q9':
            np.quantile(rewards, q=0.9) if len(rewards) > 0 else 0.0,
        }

    def get_hist(self):
        return self.hist

    def _update_stats(self, n_interaction, ret):
        """keep track of rewards and history"""
        self.hist['n_interaction'].append(n_interaction)
        self.hist['reward'].append(ret)
        self.mean_return.append(ret)

    def step(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()
