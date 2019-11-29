from chula_rl.alphazero.santorini.santorinigo.env2 import Santorini as Santorini2, state_array
from chula_rl.alphazero.santorini.santorinigo.environment import Santorini
import random
import time
import numpy as np


class Verify:
    def __init__(self, env, env2):
        self.env = env
        self.env2 = env2

    def reset(self):
        e1 = self.env.reset()
        e2 = state_array(self.env2.reset())
        assert np.array_equal(e1, e2)
        return e1

    def legal_moves(self):
        e1 = self.env.legal_moves()
        e2 = self.env2.legal_moves()
        assert len(e1) == len(e2), f'len e1 {len(e1)} != len e2 {len(e2)}'
        assert all(a == b for a, b in zip(e1, e2))
        return e1

    def step(self, a):
        s1, r1, d1, i1 = self.env.step(a)
        s2, r2, d2, _ = self.env2.step(a)
        s2 = state_array(s2)
        assert np.array_equal(s1, s2)
        assert r1 == r2, f'r1 {r1} != r2 {r2}'
        assert d1 == d2, f'd1 {d1} != d2 {d2}'
        return s1, r1, d1, i1


env = Santorini()
env2 = Santorini2()

verify = Verify(env, env2)

state = verify.reset()
for i_act in range(1000):
    actions = verify.legal_moves()
    a = random.choice(actions)
    s, r, done, info = verify.step(a)
    if done:
        verify.reset()