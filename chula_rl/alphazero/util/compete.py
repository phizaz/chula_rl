import numpy as np

from ..arena import Arena
from ..mcts import MCTS


def make_arena(net1, net2, g):
    mcts1 = MCTS(g, net1, 25, 1.)
    mcts2 = MCTS(g, net2, 25, 1.)
    arena = Arena(lambda x: np.argmax(mcts1.getActionProb(x, temp=0)),
                  lambda x: np.argmax(mcts2.getActionProb(x, temp=0)), g)
    return arena
