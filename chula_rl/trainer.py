from collections import defaultdict

import gym
import numpy as np
import pandas as pd
from tqdm import tqdm

from .exception import *
from .explorer import BaseExplorer
from .explorer.episode_explorer import EpisodeExplorer
from .policy import BasePolicy
from .util import set_seed


def train(explorer: BaseExplorer,
          policy: BasePolicy,
          make_env: callable,
          n_eval_cycle: int = 100,
          use_separate_eval: bool = False,
          n_eval_interaction: int = 1_000):
    n_next_eval = n_eval_cycle
    stats = defaultdict(list)
    with tqdm(total=explorer.n_max_interaction) as progress:
        while True:
            try:
                # explore
                data = explorer.step(policy)
                # train
                policy.optimize_step(data)
                progress.update(explorer.n_interaction - progress.n)

                # evaluate
                if explorer.n_interaction >= n_next_eval:
                    n_next_eval += n_eval_cycle
                    if use_separate_eval:
                        # real evaluate
                        eval_stats = evaluate(
                            policy,
                            make_env(),
                            n_eval_interaction=n_eval_interaction)
                        # keep stats
                        for k, v in eval_stats.items():
                            stats[k].append(v)
                    else:
                        # use running statistics (not very reliable)
                        stats['reward'].append(explorer.get_stats()['reward'])
                    # add column "n_interaction"
                    stats['n_interaction'].append(explorer.n_interaction)
            except InteractionExceeded:
                break

    return pd.DataFrame(stats)


def evaluate(policy: BasePolicy, env: gym.Env, n_eval_interaction: int):
    explorer = EpisodeExplorer(n_max_interaction=n_eval_interaction, env=env)
    while True:
        try:
            explorer.step(policy)
        except InteractionExceeded:
            break
    # average the returns
    stats = explorer.get_stats()['history']
    returns = np.array(stats['reward'])
    return {
        'reward': returns.mean(),
        'std': returns.std(),
        'q1': np.quantile(returns, q=0.1),
        'q9': np.quantile(returns, q=0.9),
        'cnt': len(returns),
    }
