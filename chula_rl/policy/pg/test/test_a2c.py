from dataclasses import dataclass
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import chula_rl as rl
from chula_rl.env.cartpolecont import ContinuousCartPoleEnv
from chula_rl.policy.pg.a2c import A2CConf, A2CPolicy


def make_env():
    env = ContinuousCartPoleEnv()
    env = rl.env.wrapper.EpisodeSummary(env)
    return env


@dataclass
class Conf(A2CConf):
    n_max_interaciton: int
    n_env: int
    n_step: int
    n_hid: int


conf = Conf(
    n_max_interaciton=100_000,
    n_env=16,  # 16
    n_step=5,  # 5, 7, 10
    n_hid=10,  # 100, 30, 10
    c_v=1.0,
    c_ent=0.01,  # 0.01, 0.1
    discount_factor=0.99,
    lr=0.03,  # sgd0.03
    clip_grad=0.5,
)

env = rl.env.DummyVecEnv([make_env] * conf.n_env)
exp = rl.explorer.VecManyStepExplorer(conf.n_step,
                                      conf.n_max_interaciton,
                                      env,
                                      n_return_avg=100)

policy = A2CPolicy(4, 1, n_hid=conf.n_hid, device='cuda', conf=conf)


class PostAction(rl.policy.BasePolicyWrapper):
    def step(self, state):
        """a in (-1, 1)"""
        a = super().step(state)
        return np.clip(a, -1, 1)


def now():
    return datetime.today().strftime('%Y-%m-%d-%H:%M:%S')


policy = PostAction(policy)

writer = tf.summary.create_file_writer(f'runs/a2c/{now()}')
with writer.as_default():
    with tqdm(total=exp.n_max_interaction) as progress:
        while True:
            data = exp.step(policy)
            policy.optimize_step(data)
            stats = exp.get_stats()
            stats.update(policy.get_stats())
            if progress.n % 1000 == 0:
                print(stats)
            for k, v in stats.items():
                tf.summary.scalar(k, v, step=exp.n_interaction)
            progress.update(exp.n_interaction - progress.n)
