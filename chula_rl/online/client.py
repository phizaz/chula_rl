import requests
import numpy as np


class OnlineEnv:
    """client side
    Args:
        host: in the form 'http://ip:port/' 
        room_id: int
        play_match: whether to play a series of games, or a single game; if play_match = True, ignore the superpower flag
        superpower: only the room owner could determine this
    """
    def __init__(self,
                 host,
                 room_id,
                 play_match: bool = True,
                 superpower: bool = False):
        super().__init__()
        self.host = host
        self.room_id = room_id
        self.play_match = play_match
        # if this is the second player, superpower is not given, it is depending on the room
        # or if play_match, superpower is ignored
        self.superpower = superpower
        self._stats = {
            # normal
            False: {
                'win': 0,
                'lose': 0
            },
            # superpower
            True: {
                'win': 0,
                'lose': 0
            },
        }
        self._legal_moves = None
        self._token = None
        self._next_room = None

    def legal_moves(self):
        assert self._legal_moves is not None
        return self._legal_moves

    def reset(self):
        # when there is no next room, the room_id will become None
        # you can no longer reset the environment
        assert self.room_id is not None, "the match is concluded"

        print('waiting for your opponent ...')
        params = {}
        if self.play_match: params['match'] = True
        if self.superpower: params['superpower'] = True
        if self._token: params['token'] = self._token
        res = requests.post(f'{self.host}/room/{self.room_id}/reset',
                            params=params)
        err_handling(res)
        obj = res.json()
        self._legal_moves = obj['legal_moves']
        self._token = obj['token']
        self._next_room = obj['next_room']
        # update superpower from the room
        self.superpower = obj['superpower']
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
        self._next_room = obj['next_room']
        self.superpower = obj['superpower']
        s, r, done, info = obj['return']

        # the game is end
        if done:
            # set the current room id to be the next room id
            # for the next reset
            self.room_id = self._next_room

            txt = f'a '
            if self.superpower:
                txt += f'superpower'
            else:
                txt += f'normal'
            txt += ' game results: '
            if r == 1.:
                txt += 'you win!'
                self._stats[self.superpower]['win'] += 1
            elif r == -1:
                txt += 'you lose!'
                self._stats[self.superpower]['lose'] += 1
            else:
                raise NotImplementedError(f'unexpected reward {r}')
            print('#########')
            print(txt)
            print('#########')
            print('current stats:')
            print('normal games:', self._stats[False])
            print('superpower games:', self._stats[True])

        s = np.array(s)
        return s, r, done, info


def err_handling(res):
    if res.status_code != 200:
        err = res.json()
        raise Exception(err['msg'])