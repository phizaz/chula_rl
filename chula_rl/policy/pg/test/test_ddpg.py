from dataclasses import dataclass
from datetime import datetime

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import chula_rl as rl
from chula_rl.env.cartpolecont import ContinuousCartPoleEnv
from chula_rl.policy.pg.ddpg import DDPGConf, DDPGPolicy


def make_env():
    env = ContinuousCartPoleEnv()
    env = rl.env.wrapper.EpisodeSummary(env)
    return env


@dataclass
class Conf(DDPGConf):
    n_max_interaciton: int
    n_max_replay_size: int
    n_env: int
    n_hid: int
    n_sample: int


conf = Conf(
    n_max_interaciton=40_000,
    n_max_replay_size=100_000,
    n_env=1,  # 1, 4
    n_hid=30,  # 100, 30, 10
    n_sample=64,  # 64
    discount_factor=0.99,
    lr=0.03,  # 0.03
    explore_std=0.1,  # 1.0
    clip_grad=0.5,
    c_pi=0.03,  # 0.03
    tau=1 / 100,
)

env = rl.env.DummyVecEnv([make_env] * conf.n_env)
exp = rl.explorer.VecOneStepExplorer(conf.n_max_interaciton,
                                     env,
                                     use_final_a=False)
exp = rl.explorer.VecOneStepUniformReplay(exp,
                                          n_sample=conf.n_sample,
                                          n_max_size=conf.n_max_replay_size,
                                          n_env=env.n_env,
                                          obs_space=env.observation_space,
                                          act_space=env.action_space)

policy = DDPGPolicy(4, 1, n_hid=conf.n_hid, device='cuda', conf=conf)


class PostAction(rl.policy.BasePolicyWrapper):
    def step(self, state):
        """a in (-1, 1)"""
        a = super().step(state)
        return np.clip(a, -1, 1)


policy = PostAction(policy)


def now():
    return datetime.today().strftime('%Y-%m-%d-%H:%M:%S')


writer = tf.summary.create_file_writer(f'runs/a2c/{now()}')
with writer.as_default():
    with tqdm(total=exp.n_max_interaction) as progress:
        while True:
            data = exp.step(policy)
            policy.optimize_step(data)
            stats = exp.get_stats()
            stats.update(policy.get_stats())
            if progress.n % 100 == 0:
                print(stats)
            for k, v in stats.items():
                tf.summary.scalar(k, v, step=exp.n_interaction)
            progress.update(exp.n_interaction - progress.n)
