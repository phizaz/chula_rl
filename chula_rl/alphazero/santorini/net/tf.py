import os
import time

import attr
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from tensorflow.keras import layers
from torch import nn, optim

from chula_rl.alphazero.net import Net
from chula_rl.alphazero.util.avg import AverageMeter
from chula_rl.alphazero.util.bar import Bar

from .features import CNNFeature


@attr.s(auto_attribs=True)
class NetArgs:
    lr: float = 0.001
    dropout: float = 0.3
    epochs: int = 10
    batch_size: int = 64
    num_channels: int = 256


class SantoriniNet(Net):
    def __init__(self, board_dim: tuple, n_action: int, args: NetArgs):
        self.args = args
        self.net = Backbone(CNNFeature.n_ch, board_dim, n_action, args)
        self.action_size = n_action

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        input_boards, target_pis, target_vs = list(zip(*examples))
        input_boards = [CNNFeature.extract(b) for b in input_boards]
        input_boards = np.asarray(input_boards)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        self.net.model.fit(x=input_boards,
                           y=[target_pis, target_vs],
                           batch_size=self.args.batch_size,
                           epochs=self.args.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        # feature extract
        board = CNNFeature.extract(board)
        board = board[np.newaxis, :, :, :].astype(np.float32)

        # run
        pi, v = self.net.model.predict(board)

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
        self.net.model.save_weights(filepath)

    def load_checkpoint(self,
                        folder='checkpoint',
                        filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        self.net.model.load_weights(filepath)


class Backbone:
    def __init__(self, in_ch: int, board_dim: tuple, n_action: int,
                 args: NetArgs):
        # game params
        self.board_x, self.board_y = board_dim
        self.action_size = n_action
        self.args = args

        n_ch = args.num_channels

        self.input = tf.keras.Input(shape=(self.board_x, self.board_y, in_ch))

        self.core = tf.keras.Sequential([
            layers.Conv2D(n_ch, 3, 1, 'same', data_format='channels_first'),
            layers.BatchNormalization(1),  # channels first
            layers.ReLU(),
            layers.Conv2D(n_ch, 3, 1, 'same', data_format='channels_first'),
            layers.BatchNormalization(1),
            layers.ReLU(),
            layers.Conv2D(n_ch, 3, 1, 'same', data_format='channels_first'),
            layers.BatchNormalization(1),
            layers.ReLU(),
            layers.Conv2D(n_ch, 3, 1, 'same', data_format='channels_first'),
            layers.BatchNormalization(1),
            layers.ReLU(),
            layers.Flatten(data_format='channels_first'),
            layers.Dense(1024),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(args.dropout),
            layers.Dense(512),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(args.dropout),
        ])

        self.action_head = layers.Dense(n_action, activation='softmax')
        self.value_head = layers.Dense(1, activation='tanh')

        core = self.core(self.input)
        pi = self.action_head(core)
        v = self.value_head(core)

        self.model = tf.keras.Model(inputs=self.input, outputs=[pi, v])
        self.model.compile(
            loss=['categorical_crossentropy', 'mean_squared_error'],
            optimizer=tf.keras.optimizers.Adam(args.lr))
