import gym


class ClipEpisodeLength(gym.Wrapper):
    """ Env wrapper that clips number of frames an episode can last """
    def __init__(self, env, n_max_length):
        super().__init__(env)

        self.n_max_length = n_max_length
        self.i_step = 0

    def reset(self, **kwargs):
        self.i_step = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.i_step += 1
        s, r, done, info = self.env.step(action)

        if self.i_step >= self.n_max_length:
            done = True
            info['clipped_length'] = True

        return s, r, done, info
