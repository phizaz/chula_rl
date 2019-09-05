from ..base_policy import BasePolicyWrapper


class ForEachData(BasePolicyWrapper):
    """apply a data list onto a policy which accepts an atomic data"""
    def optimize_step(self, data):
        keys = data.keys()
        for vals in zip(*[data[k] for k in keys]):
            d = {k: v for k, v in zip(keys, vals)}
            super().optimize_step(d)