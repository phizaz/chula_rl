from collections import defaultdict

import gym
import numpy as np
import pandas as pd
from tqdm import tqdm

from .exception import *
from .explorer import BaseExplorer
from .explorer.episode_explorer import EpisodeExplorer
from .policy import BasePolicy
from .util import set_seed, tup

from .callback import *


def call_cb(name, kwargs, cbs):
    for cb in cbs:
        f = getattr(cb, name)
        if callable(f):
            f(**kwargs)


def get_stats(cbs):
    stats = {}
    for cb in cbs:
        if isinstance(cb, StatsCallback):
            stats.update(cb.stats)
    return stats


def append_dict(dict, stats):
    for k, v in stats.items():
        dict[k].append(v)


def train(explorer: BaseExplorer,
          policy: BasePolicy,
          make_env: callable,
          callbacks=[]):
    def kwargs(additional={}):
        # usual args for callbacks
        return {
            'explorer': explorer,
            'policy': policy,
            'make_env': make_env,
            **additional,
        }

    # run the training loop
    with tqdm(total=explorer.n_max_interaction) as progress:
        call_cb('before_start', kwargs(), callbacks)
        while True:
            try:
                # explore
                call_cb('before_step', kwargs(), callbacks)
                data = explorer.step(policy)
                call_cb('after_step', kwargs({'data': data}), callbacks)
                # train
                call_cb('before_optimize', kwargs({'data': data}), callbacks)
                if data is not None:
                    # no data, skip optimization
                    policy.optimize_step(data)
                call_cb('after_optimize', kwargs({'data': data}), callbacks)
                progress.update(explorer.n_interaction - progress.n)
            except InteractionExceeded:
                break
        call_cb('after_end', kwargs(), callbacks)

    # join stats from callbacks
    df = None
    for cb in callbacks:
        if isinstance(cb, StatsCallback):
            if df is None: df = cb.df
            else: df = pd.merge(df, cb.df, how='outer')

    return df


class ExplorerStatsCb(StatsCallback):
    """log explorer's rewards"""
    def __init__(self, n_log_cycle):
        super().__init__(n_log_cycle)

    def after_step(self, explorer, policy, make_env, **kwargs):
        if self.should_log(explorer):
            self.hist['n_interaction'].append(explorer.n_interaction)
            append_dict(self.hist, explorer.get_stats())


class EvalCb(StatsCallback):
    """evaluation callbacks, this will run an agent inside a new environment for a fixed number of steps, then average the results"""
    def __init__(self, n_eval_cycle, n_eval_interaction):
        super().__init__(n_eval_cycle)
        self.n_eval_interaction = n_eval_interaction

    def after_step(self, explorer, policy, make_env, **kwargs):
        if self.should_log(explorer):
            eval_stats = evaluate(policy,
                                  make_env(),
                                  n_eval_interaction=self.n_eval_interaction)
            self.hist['n_interaction'].append(explorer.n_interaction)
            append_dict(self.hist, eval_stats)


def calculate_return(r, gamma):
    g = 0
    for i in range(len(r)):
        g += r[i] * gamma**i
    return g


def evaluate(policy: BasePolicy, env: gym.Env, n_eval_interaction: int):
    """runs a policy in an env to get an estimate of its performance, used for inspection"""
    explorer = EpisodeExplorer(n_max_interaction=n_eval_interaction, env=env)
    returns = []
    return_preds = []
    while True:
        try:
            data = explorer.step(policy)
            sa = tup(data['s'][0]) + tup(data['a'][0])
            ret = calculate_return(data['r'], policy.discount_factor)

            returns.append(ret)
            return_preds.append(policy.q[sa])
        except InteractionExceeded:
            break
    # average the returns
    stats = explorer.get_hist()
    rewards = np.array(stats['reward'])
    return {
        # total reward without discount
        'eval_reward': rewards.mean(),
        'eval_std': rewards.std(),
        'eval_q1': np.quantile(rewards, q=0.1),
        'eval_q9': np.quantile(rewards, q=0.9),
        # number of episode averaged
        'eval_cnt': len(rewards),
        # reward with discount
        'eval_return': np.array(returns).mean(),
        # predicted reward from the policy
        'eval_return_pred': np.array(return_preds).mean(),
    }
