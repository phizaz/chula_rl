from pprint import pprint

from chula_rl.env.simple_gridworld import Gridworld
from chula_rl.explorer.one_step_uniform_replay import *
import numpy as np


class Explorer:
    def step(self, policy=None):
        data = {
            's': np.random.randint(0, 4, 2),
            'a': np.random.randint(4),
            'r': np.random.randn(),
            'done': False,
        }
        print('data:', data)
        return data


env = Gridworld()
explorer = Explorer()
replay = OneStepUniformReplay(explorer, 1, 3, env.observation_space,
                              env.action_space)

for i in range(4):
    replay.step(None)
pprint(replay.step(None))
