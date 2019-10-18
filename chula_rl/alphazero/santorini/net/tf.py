import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from chula_rl.alphazero.game import Game
from chula_rl.alphazero.net import Net


class OthelloNet(Net):
    def __init__(self, game: Game, args: NetArgs):
        self.args = args
        self.nnet = Backbone(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        # implement this function
        # look for inspiration from Othello which is provided in full!
        pass

    def predict(self, board):
        """
        board: np array with board
        """
        # implement this function
        pass

    def save_checkpoint(self,
                        folder='checkpoint',
                        filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".
                  format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        self.nnet.model.save_weights(filepath)

    def load_checkpoint(self,
                        folder='checkpoint',
                        filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        self.nnet.model.load_weights(filepath)


class Backbone:
    def __init__(self, game, args):
        # implement this function
        # look for inspiration from Othello which is provided in full!
        # self.model = some keras model
        pass