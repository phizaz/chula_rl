import requests
import numpy as np


class OnlineEnv:
    """client side
    Args:
        host: in the form 'http://ip:port/' 
        room_id: int
    """
    def __init__(self, host, room_id):
        super().__init__()
        self.host = host
        self.room_id = room_id
        self._legal_moves = None
        self._token = None

    def legal_moves(self):
        assert self._legal_moves is not None
        return self._legal_moves

    def reset(self):
        res = requests.post(f'{self.host}/room/{self.room_id}/reset')
        err_handling(res)
        obj = res.json()
        self._legal_moves = obj['legal_moves']
        self._token = obj['token']
        s = obj['return']
        s = np.array(s)
        return s

    def step(self, a):
        assert self._token is not None
        res = requests.post(
            f'{self.host}/room/{self.room_id}/step/{a}?token={self._token}')
        err_handling(res)
        obj = res.json()
        self._legal_moves = obj['legal_moves']
        self._token = obj['token']
        s, r, done, info = obj['return']
        s = np.array(s)
        return s, r, done, info


def err_handling(res):
    if res.status_code != 200:
        err = res.json()
        raise Exception(err['msg'])