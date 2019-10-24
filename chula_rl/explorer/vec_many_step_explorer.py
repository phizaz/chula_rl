from collections import defaultdict

import numpy as np

from chula_rl.exception import InteractionExceeded
from chula_rl.policy.base_policy import BasePolicy

from ..env import BaseVecEnv
from .base_explorer import BaseExplorer


class VecManyStepExplorer(BaseExplorer):
    def __init__(self,
                 n_step: int,
                 n_max_interaction: int,
                 env: BaseVecEnv,
                 use_final_a: bool = False,
                 **kwargs):
        assert isinstance(env, BaseVecEnv), "needs parellel env (VecEnv)"
        super().__init__(env, **kwargs)
        self.n_step = n_step
        self.n_max_interaction = n_max_interaction
        self.use_final_a = use_final_a

        self.last_s = self.env.reset()

        self.n_interaction = 0
        self.n_ep = [0] * env.n_env

    def step(self, policy: BasePolicy):
        data = defaultdict(list)
        for i_step in range(self.n_step):
            if self.n_interaction > self.n_max_interaction:
                raise InteractionExceeded()

            # explore
            a = policy.step(self.last_s)
            s, r, done, info = self.env.step(a)

            # collect data
            data['s'].append(self.last_s)
            data['a'].append(a)
            data['r'].append(r)
            data['done'].append(done)

            # last step
            if i_step == self.n_step - 1:
                # for SARSA
                if self.use_final_a:
                    data['final_a'] = policy.step(s)
                data['final_s'] = s

            self.last_s = s

            # statistics
            for i_env, (d, i) in enumerate(zip(done, info)):
                self.n_interaction += 1
                if d:
                    self.n_ep[i_env] += 1
                    self._update_stats(self.n_interaction,
                                       i['episode']['reward'])

        # convert to numpy
        output = {}
        for k, v in data.items():
            output[k] = np.stack(v)

        return output
