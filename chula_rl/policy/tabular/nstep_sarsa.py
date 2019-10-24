import numpy as np
from ..base_policy import BasePolicy


class NstepExpectedSarsaPolicy(BasePolicy):
    """aggregation of one-step to n-step expected sarsa."""
    def __init__(self, lr, discount_factor, observation_space, n_action,
                 only_first: bool):
        self.lr = lr
        self.discount_factor = discount_factor
        self.observation_space = observation_space
        self.n_action = n_action
        self.only_first = only_first

        self.q = np.zeros(list(self.observation_space.high) +
                          [n_action])  # (s0, s1, a)

    def step(self, state):
        return np.argmax(self.q[tuple(state)])

    def optimize_step(self, data):
        # evaluation (prediction)
        s = np.array(data['s'])
        a = np.array(data['a'])
        r = np.array(data['r'])
        done = np.array(data['done'])
        final_s = data['final_s']

        # calculate the n-step to one-step returns
        n_step = len(r)
        g = np.zeros(r.shape)
        final_sa = tuple(final_s) + (self.step(final_s), )
        final_v = self.q[final_sa]
        current = final_v
        for i in reversed(range(0, n_step)):
            current = r[i] + (1 - done[i]) * self.discount_factor * current
            g[i] = current

        # update
        if self.only_first:  # using only g0, strictly n-step return
            first_sa = tuple(s[0]) + (a[0], )
            td_error = g[0] - self.q[first_sa]
            self.q[first_sa] += self.lr * td_error
        else:  # use all g's this would then be one-step to n-step returns
            # join index
            a = np.expand_dims(a, 1)
            sa = np.hstack((s, a))
            # dedup idx to get the first visit (might not be needed)
            _, i = np.unique(sa, return_index=True, axis=0)
            idx = tuple(sa[i].T)  # ((dim0 idxs), (dim1 idxs), (dim2 idxs))

            td_error = g[i] - self.q[idx]
            self.q[idx] += self.lr * td_error
