class BasePolicy:
    def step(self, state):
        raise NotImplementedError()

    def optimize_step(self, data):
        raise NotImplementedError()


class BasePolicyWrapper:
    def __init__(self, policy: BasePolicy):
        self.policy = policy

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        return getattr(self.policy, name)

    def step(self, state):
        return self.policy.step(state)

    def optimize_step(self, data):
        self.policy.optimize_step(data)