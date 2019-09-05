from .base_replay import BaseReplay
import gym
import numpy as np


class OneStepUniformReplay(BaseReplay):
    """a wrapper for one-step explorer, this will put the experience into a database, then samples as a return"""
    def __init__(self, explorer, n_sample, n_max_size, obs_space: gym.Space,
                 act_space: gym.Space):
        super().__init__(explorer)
        self.n_sample = n_sample
        self.n_max_size = n_max_size
        self.obs_space = obs_space
        self.act_space = act_space

        # size
        self.n = 0
        # index
        self.i = 0
        # data
        s_shape = [self.n_max_size] + list(obs_space.shape)
        self.s = np.zeros(s_shape, dtype=obs_space.dtype)
        a_shape = [self.n_max_size] + list(act_space.shape)
        self.a = np.zeros(a_shape, dtype=act_space.dtype)
        self.r = np.zeros([self.n_max_size], dtype=np.float32)
        self.done = np.zeros([self.n_max_size], dtype=bool)
        self.extra = {}

    def get_stats(self):
        out = super().get_stats()
        out.update({})
        return out

    def sample(self, n_sample):
        """sample from replay for n_sample size"""
        # -1 to make sure you have the "next" state and action
        assert self.n > 0
        if self.n <= self.n_max_size:
            idx = np.random.choice(self.n - 1, n_sample, replace=True)
        else:
            offset = self.n - self.n_max_size
            idx = np.random.choice(self.n - offset - 1, n_sample,
                                   replace=True) + offset
        data = {
            's': self.take(self.s, idx),
            'a': self.take(self.a, idx),
            'r': self.take(self.r, idx),
            'ss': self.take(self.s, idx + 1),
            'aa': self.take(self.a, idx + 1),
            'done': self.take(self.done, idx),
        }
        return data

    def take(self, arr, idx):
        idx = idx % self.n_max_size
        return arr[idx]

    def put(self, exp):
        data = {
            's': self.s,
            'a': self.a,
            'r': self.r,
            'done': self.done,
        }
        ignore = set(['ss', 'aa'])
        for k, v in exp.items():
            if k in data:
                data[k][self.i] = v
            elif k not in ignore:
                raise NotImplementedError('not support extras')
        self.i += 1
        self.i %= self.n_max_size
        self.n += 1

    def step(self, policy):
        exp = self.explorer.step(policy)
        self.put(exp)
        if self.n > 1:
            data = self.sample(self.n_sample)
        else:
            data = None
        return data
