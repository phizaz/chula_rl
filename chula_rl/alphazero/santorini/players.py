import numpy as np


class RandomPlayer:
    """a random player for the game of santorini"""
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a] != 1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanPlayer:
    """if you want to play it yourself"""
    def __init__(self, game):
        self.game = game

    def play(self, board):
        print(board[:2])  # print the buliding and the positions of the workers
        valid = self.game.getValidMoves(board, 1)

        while True:
            a = input()
            worker, walk, build, *_ = a.split(' ')
            worker = int(worker)
            ai = self.game.env.atoi[(worker, walk, build)]
            if valid[ai]:
                break
            else:
                print('Invalid')
        return ai