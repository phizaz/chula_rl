from .base_explorer import BaseExplorer


class BaseReplay:
    def __init__(self, explorer):
        self.explorer = explorer

    def get_stats(self):
        return self.explorer.get_stats()

    def step(self, policy):
        raise NotImplementedError()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        return getattr(self.explorer, name)