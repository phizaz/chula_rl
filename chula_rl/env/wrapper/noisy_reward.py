import gym
import numpy as np


class NoisyReward(gym.Wrapper):
    def __init__(self, env, sigma):
        super().__init__(env)
        self.sigma = sigma

    def step(self, action):
        s, r, done, info = self.env.step(action)
        r += np.random.randn()
        return s, r, done, info
