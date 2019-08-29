import gym
from gym.core import Wrapper
import time
import numpy as np


class EpisodeSummary(Wrapper):
    """allows to get env statistics like rewards.
    you will get episode statistics at the end of ep (as info)."""

    def __init__(self, env):
        super().__init__(env)
        self.rewards = None
        self.needs_reset = True
        self.total_steps = 0

    def reset(self, **kwargs):
        self.tstart = time.time()
        self.rewards = []
        self.needs_reset = False
        return super().reset(**kwargs)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        s, r, done, info = self.env.step(action)
        self.rewards.append(r)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {
                'reward': round(eprew, 6),
                'length': eplen,
                'time': round(time.time() - self.tstart, 6)
            }
            info['episode'] = epinfo
        self.total_steps += 1
        return (s, r, done, info)

    def close(self):
        return super().close()
