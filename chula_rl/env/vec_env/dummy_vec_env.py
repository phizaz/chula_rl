"""
taken from https://github.com/MillionIntegrals/vel
"""
import numpy as np
from gym import spaces
from .base_vec_env import BaseVecEnv
from .util import copy_obs_dict, dict_to_obs, obs_space_info


class DummyVecEnv(BaseVecEnv):
    """
    VecEnv that does runs multiple environments sequentially, that is,
    the step and reset commands are send to one environment at a time.
    Useful when debugging and when num_env == 1 (in the latter case,
    avoids communication overhead)
    """
    def __init__(self, env_fns):
        """
        Arguments:
        env_fns: iterable of callables      functions that build environments
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        BaseVecEnv.__init__(self, len(env_fns), env.observation_space,
                            env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_s = {
            k: np.zeros((self.n_env, ) + tuple(shapes[k]), dtype=dtypes[k])
            for k in self.keys
        }
        self.buf_done = np.zeros((self.n_env, ), dtype=np.bool)
        self.buf_r = np.zeros((self.n_env, ), dtype=np.float32)
        self.buf_info = [{} for _ in range(self.n_env)]
        self.actions = None
        self.specs = [e.spec for e in self.envs]

    def step_async(self, actions):
        listify = True
        try:
            if len(actions) == self.n_env:
                listify = False
        except TypeError:
            pass

        if not listify:
            self.actions = actions
        else:
            assert self.n_env == 1, "actions {} is either not a list or has a wrong size - cannot match to {} environments".format(
                actions, self.n_env)
            self.actions = [actions]

    def step_wait(self):
        for e in range(self.n_env):
            action = self.actions[e]
            if isinstance(self.envs[e].action_space, spaces.Discrete):
                action = int(action)

            obs, self.buf_r[e], self.buf_done[e], self.buf_info[e] = self.envs[
                e].step(action)
            if self.buf_done[e]:
                obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return (self._obs_from_buf(), np.copy(self.buf_r),
                np.copy(self.buf_done), self.buf_info.copy())

    def reset(self):
        for e in range(self.n_env):
            obs = self.envs[e].reset()
            self._save_obs(e, obs)
        return self._obs_from_buf()

    def _save_obs(self, e, obs):
        for k in self.keys:
            if k is None:
                self.buf_s[k][e] = obs
            else:
                self.buf_s[k][e] = obs[k]

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_s))

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, mode='human'):
        if self.n_env == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def close_extras(self):
        """close all envs"""
        for env in self.envs:
            env.close()
