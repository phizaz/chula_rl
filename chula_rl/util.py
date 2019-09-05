import random
from functools import partial

import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing as mp
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def tup(x):
    if isinstance(x, (tuple, list, np.ndarray)):
        return tuple(x)
    else:
        return (x, )


def tup_sa(s, a):
    return tup(s) + tup(a)


def reindex(src, tgt):
    """duplicate rows to make sure that n_interaction is the same as tgt dataframe.
    this is useful for monte carlo methods where the n_interaction column comes in irregular intervals.
    """
    out = pd.DataFrame(columns=src.columns)
    out = out.astype(src.dtypes)

    j = 0
    for index, row in tgt.iterrows():
        last_row = None
        while j < len(
                src) and row['n_interaction'] >= src['n_interaction'].loc[j]:
            last_row = src.loc[j]  # this selects dataframe with one row
            j += 1

        if last_row is not None:
            last_row.at['n_interaction'] = row['n_interaction']
            out.loc[len(out)] = last_row
            j -= 1  # allowing the last row to be reused
    return out


def plot_std(ax, group_df, y):
    """plot results with mean and std"""
    mean = group_df.agg({y: 'mean'})
    std = group_df.agg({y: 'std'})
    ax.plot(mean)
    ax.fill_between(mean.index, mean[y] - std[y], mean[y] + std[y], alpha=0.07)


def t_div(data, confidence=0.95, ddof=None):
    """calculate deviation from the mean for t-statistic given confidence"""
    n = len(data)
    se = data.std(ddof=1) / np.sqrt(n)
    t_score = stats.t.ppf(0.5 + confidence / 2,
                          df=n - 1 if ddof is None else ddof)
    return t_score * se


def plot_conf(ax, group_df, y, confidence: float = 0.95):
    """plot results with mean and confidence interval"""
    mean = group_df.agg({y: 'mean'})
    div = group_df.agg({y: partial(t_div, confidence=confidence)})
    ax.plot(mean)
    ax.fill_between(mean.index, mean[y] - div[y], mean[y] + div[y], alpha=0.07)


def parallel_map(fn, list_of_args):
    """run fn parallelly"""
    with mp.Pool() as pool:
        out = []
        with tqdm(total=len(list_of_args)) as progress:
            for res in pool.imap(fn, list_of_args):
                out.append(res)
                progress.update()
        return out