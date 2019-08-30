"""
taken from: https://github.com/Datatouille/rl-workshop
with some changes
"""

import numpy as np
from gym import Env, spaces
import seaborn as sns
import matplotlib.pyplot as plt


class Gridworld(Env):
    """
    Simple gridworld environment

    Args:
        shape: the size of the grid (row, col)
        start: starting position
        goal: target position
        traps: list of traps
        wind: chance of not getting into the desired position (shifted by +1, -1 on either row or col)
    """
    def __init__(self,
                 shape=(4, 3),
                 start=(2, 0),
                 goal=(1, 2),
                 traps=[(1, 1)],
                 goal_reward=5,
                 trap_reward=-5,
                 move_reward=-1,
                 wind_p=0.):
        self.shape = shape
        self.start = start
        self.goal = goal
        self.traps = traps
        self.move_reward = move_reward
        self.trap_reward = trap_reward
        self.goal_reward = goal_reward
        self.wind_p = wind_p

        self.action_space = spaces.Discrete(4)
        low = np.array([0, 0])
        high = np.array(shape)
        self.observation_space = spaces.Box(low, high, dtype=np.int64)

        self.action_text = np.array(['U', 'L', 'D', 'R'])
        self.state_space = [(i, j) for i in range(shape[0])
                            for j in range(shape[1])]

    def reset(self, start=None):
        if start is None:
            self.i = self.start[0]
            self.j = self.start[1]
        else:
            self.i, self.j = start
        self.traversed = [self.start]
        self.done = False
        #physical grid
        self.physical_grid = dict.fromkeys(self.state_space, ['F', 'x'])
        self.physical_grid[self.start] = ['F', 'o']
        self.physical_grid[self.goal] = ['G', 'x']
        for t in self.traps:
            self.physical_grid[t] = ['T', 'x']
        #reward grid
        self.reward_grid = dict.fromkeys(self.state_space, 0)
        self.reward_grid[self.goal] = self.goal_reward
        for t in self.traps:
            self.reward_grid[t] = self.trap_reward
        return np.array([self.i, self.j])

    def step(self, action):
        if self.done:
            raise NotImplementedError('an environment needs reset')

        reward = self.move_reward
        i, j = self.i, self.j
        action = self.action_text[action]
        if action == 'U':
            i -= 1
        elif action == 'L':
            j -= 1
        elif action == 'D':
            i += 1
        elif action == 'R':
            j += 1
        else:
            raise NotImplementedError()
        #check legality
        if (i, j) in self.state_space:
            #update position
            self.i, self.j = i, j
            #wind blows
            self._wind()
            #save traversed
            self.traversed.append((self.i, self.j))
            #update physical
            self._update_physical()
            #update reward
            reward += self.reward_grid[(self.i, self.j)]
        else:
            pass
        if (self.i, self.j) == self.goal: self.done = True
        #return s',r, done or not
        return (np.array((self.i, self.j)), reward, self.done, {})

    def render(self, mode='human', figsize=(4, 2)):
        assert mode == 'human'
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        self._print_physical(axes[0])
        self._print_reward(axes[1])

    def _print_reward(self, ax, visible_only=False):
        field = np.zeros(self.shape)
        ax.set_title('Reward')
        for (x, y), r in self.reward_grid.items():
            field[x, y] = r
        sns.heatmap(field, center=0, annot=True, ax=ax)

    def _print_physical(self, ax, visible_only=False):
        field = np.zeros(self.shape)
        ax.set_title('Map')
        for (x, y), (t, w) in self.physical_grid.items():
            if w == 'o':
                field[x, y] = 2
            elif t == 'T':
                field[x, y] = -1
            elif t == 'G':
                field[x, y] = 1
            else:
                field[x, y] = 0
        sns.heatmap(field, center=0, annot=True, cbar=False, ax=ax)

    def _update_physical(self):
        for key in self.state_space:
            self.physical_grid[key][1] = 'x'
        tile = self.physical_grid[(self.i, self.j)][0]
        self.physical_grid[(self.i, self.j)] = [tile, 'o']

    def _wind(self):
        """wind could alter the landing position"""
        offset = np.random.choice([-1, 1])
        if np.random.uniform() < self.wind_p:
            if np.random.uniform() < 0.5:
                pos = self.i + offset
                self.i = np.clip(pos, 0, self.shape[0] - 1)
            else:
                pos = self.j + offset
                self.j = np.clip(pos, 0, self.shape[1] - 1)
