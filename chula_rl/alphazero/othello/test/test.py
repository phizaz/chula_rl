import tensorflow as tf

import chula_rl as rl
from chula_rl.alphazero.coach import *
from chula_rl.alphazero.othello.game import *
from chula_rl.alphazero.othello.net.tf import *
tf.compat.v1.disable_eager_execution()
# from chula_rl.alphazero.othello.net.pytorch import *

project = 'othello'


def make_name(args: Args, nargs: NetArgs):
    return f'{project}-itr{args.n_itr}-ep{args.n_ep}-sims{args.n_sims}-compete{args.n_compete}_ep{nargs.n_ep}-ch{nargs.n_ch}'


def run(args, net_args):
    def make_net():
        return OthelloNet(g, net_args)

    name = make_name(args, net_args)
    print('name:', name)
    args.checkpoint = f'./test/{project}/{name}'

    g = OthelloGame(6)

    writer = tf.summary.create_file_writer(f'tensorboard/{project}/{name}')
    with writer.as_default():
        c = Coach(g, make_net, args)
        c.learn()


if __name__ == "__main__":
    args = Args(
        n_itr=10,
        n_ep=5,
        n_sims=25,
        n_compete=5,
    )

    net_args = NetArgs(
        dropout=0.3,
        n_ep=10,
        n_bs=64,
        n_ch=256,
    )
    run(args, net_args)
