import requests
import numpy as np


class OnlineEnv:
    """client side
    Args:
        host: in the form 'http://ip:port/' 
        room_id: int
        superpower: only the room owner could determine this
    """
    def __init__(self, host, room_id, superpower: bool = False):
        super().__init__()
        self.host = host
        self.room_id = room_id
        self.superpower = superpower
        self._legal_moves = None
        self._token = None

    def legal_moves(self):
        assert self._legal_moves is not None
        return self._legal_moves

    def reset(self):
        print('waiting for your opponent ...')
        res = requests.post(
            f'{self.host}/room/{self.room_id}/reset{"?superpower" if self.superpower else ""}'
        )
        err_handling(res)
        obj = res.json()
        self._legal_moves = obj['legal_moves']
        self._token = obj['token']
        s = obj['return']
        s = np.array(s)
        return s

    def step(self, a):
        assert self._token is not None
        print('waiting for your opponent ...')
        res = requests.post(
            f'{self.host}/room/{self.room_id}/step/{a}?token={self._token}')
        err_handling(res)
        obj = res.json()
        self._legal_moves = obj['legal_moves']
        self._token = obj['token']
        s, r, done, info = obj['return']
        if done:
            if r == 1.:
                print('you win!')
            elif r == -1:
                print('you lose!')
            else:
                raise NotImplementedError(f'unexpected reward {r}')
        s = np.array(s)
        return s, r, done, info


def err_handling(res):
    if res.status_code != 200:
        err = res.json()
        raise Exception(err['msg'])