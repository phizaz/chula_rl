from collections import defaultdict

import pandas as pd


class BaseCallback:
    """callback is a way to inject a piece of code into the training loop at each stage"""
    def before_start(self, **kwargs):
        pass

    def before_step(self, **kwargs):
        pass

    def after_step(self, **kwargs):
        pass

    def before_optimize(self, **kwargs):
        pass

    def after_optimize(self, **kwargs):
        pass

    def after_end(self, **kwargs):
        pass


class StatsCallback(BaseCallback):
    """a callback with stats"""
    def __init__(self, n_log_cycle):
        self.n_log_cycle = n_log_cycle
        self.n_next_log = n_log_cycle
        self.hist = defaultdict(list)

    @property
    def df(self):
        return pd.DataFrame(self.hist)

    def should_log(self, explorer):
        if explorer.n_interaction >= self.n_next_log:
            self.n_next_log += self.n_log_cycle
            return True
        return False
