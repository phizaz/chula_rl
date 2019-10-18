import os
import time

import attr
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from chula_rl.alphazero.game import Game
from chula_rl.alphazero.net import Net


@attr.s(auto_attribs=True)
class NetArgs:
    lr: float = 0.001
    dropout: float = 0.3
    n_ep: int = 10
    n_bs: int = 64
    n_ch: int = 256


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
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.nnet.model.fit(x=input_boards,
                            y=[target_pis, target_vs],
                            batch_size=self.args.n_bs,
                            epochs=self.args.n_ep)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :].astype(np.float32)

        # run
        pi, v = self.nnet.model.predict(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time() - start))
        return pi[0], v[0]

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
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        # Neural Net
        self.input_boards = Input(
            shape=(self.board_x,
                   self.board_y))  # s: batch_size x board_x x board_y

        x_image = Reshape(
            (self.board_x, self.board_y,
             1))(self.input_boards)  # batch_size  x board_x x board_y x 1
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(
            args.n_ch, 3, padding='same', use_bias=False)(
                x_image)))  # batch_size  x board_x x board_y x num_channels
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(
            args.n_ch, 3, padding='same', use_bias=False)(
                h_conv1)))  # batch_size  x board_x x board_y x num_channels
        h_conv3 = Activation('relu')(
            BatchNormalization(axis=3)(Conv2D(args.n_ch,
                                              3,
                                              padding='valid',
                                              use_bias=False)(h_conv2))
        )  # batch_size  x (board_x-2) x (board_y-2) x num_channels
        h_conv4 = Activation('relu')(
            BatchNormalization(axis=3)(Conv2D(args.n_ch,
                                              3,
                                              padding='valid',
                                              use_bias=False)(h_conv3))
        )  # batch_size  x (board_x-4) x (board_y-4) x num_channels
        h_conv4_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(
            axis=1)(Dense(1024,
                          use_bias=False)(h_conv4_flat))))  # batch_size x 1024
        s_fc2 = Dropout(args.dropout)(Activation('relu')(BatchNormalization(
            axis=1)(Dense(512, use_bias=False)(s_fc1))))  # batch_size x 1024
        self.pi = Dense(self.action_size, activation='softmax',
                        name='pi')(s_fc2)  # batch_size x self.action_size
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)  # batch_size x 1

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'],
            optimizer=Adam(args.lr))
