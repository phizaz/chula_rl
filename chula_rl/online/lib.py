import random
import secrets
from collections import defaultdict
from dataclasses import dataclass
from queue import Queue
from threading import Event, Lock, Thread
from typing import Dict

import numpy as np

from chula_rl.alphazero.santorini.santorinigo.fast import Santorini


class SimpleEnv:
    def __init__(self):
        pass

    def reset(self):
        return {'state': np.random.randn(4).tolist()}

    def step(self, a):
        return {
            'state': np.random.randn(4).tolist(),
            'reward': random.randint(0, 10),
            'done': random.randint(0, 1),
            'info': {},
        }


def generate_token():
    return secrets.token_hex(16)


class Return:
    def __init__(self):
        self.available = Event()
        self.value = None

    def set(self, value):
        self.value = value
        self.available.set()

    def wait(self):
        self.available.wait()
        return self.value


class Env:
    def __init__(self, superpower, first_player=1):
        self.first_player = first_player
        self.env = Santorini(auto_invert=True, superpower=superpower)
        self.q1 = Queue()
        self.q2 = Queue()
        self.worker = Thread(target=self._worker)
        self.worker.daemon = True
        self.worker.start()
        self.done = False

    def _exception(self, e):
        self.done = True
        # clear the queue
        while not self.q1.empty():
            op, a, ret = self.q1.get()
            ret.set(e)
        while not self.q2.empty():
            op, a, ret = self.q2.get()
            ret.set(e)

    def _worker(self):
        """
        rearrange the command sequence
        """
        player = self.first_player  # first player
        q = [None, self.q1, self.q2]
        states = [None, 'reset', 'reset']
        not_ret = None
        while True:
            op, a, ret = q[player].get()
            # print(f'worker: player: {player} op: {op}')
            try:
                assert op == states[
                    player], f'state {op} expects {states[player]}'

                if op == 'reset':
                    if player == self.first_player:
                        # first time case
                        s = self.env.reset()
                        legal_moves = self.env.legal_moves()
                        ret.set((s, None, None, None, legal_moves))
                        states[player] = 'step'
                    else:
                        states[player] = 'step'

                    # switch player
                    if player == 1:
                        player = 2
                    else:
                        player = 1
                    not_ret = ret
                elif op == 'step':
                    # handle error
                    s, r, done, info = self.env.step(a)

                    if done:
                        ret.set((s, r, done, info, []))
                        not_ret.set((s, -r, done, info, []))
                        # end
                        self.done = True
                        break
                    else:
                        legal_moves = self.env.legal_moves()
                        not_ret.set((s, -r, done, info, legal_moves))

                    # switch player
                    if player == 1:
                        player = 2
                    else:
                        player = 1
                    not_ret = ret
                else:
                    raise NotImplementedError()
            except Exception as e:
                # exception
                # return as exception
                ret.set(e)
                if not_ret is not None:
                    not_ret.set(
                        Exception('your opponent did something unexpected'))
                self._exception(
                    Exception('something else terminated the environment'))
                break

    def reset(self, player):
        assert not self.done, 'the env is finished'
        ret = Return()
        cmd = ('reset', None, ret)
        if player == 1:
            # print('put to player 1 queue')
            self.q1.put(cmd)
        elif player == 2:
            # print('put to player 2 queue')
            self.q2.put(cmd)
        else:
            raise NotImplementedError()
        return ret

    def step(self, a, player):
        assert not self.done, 'the env is finished'
        ret = Return()
        cmd = ('step', a, ret)
        if player == 1:
            self.q1.put(cmd)
        elif player == 2:
            self.q2.put(cmd)
        else:
            raise NotImplementedError()
        return ret


def serializable(v):
    s, r, done, info, legal_moves = v
    s = s.tolist()
    return s, r, done, info, legal_moves


class Room:
    """
    Args:
        next_room: the next room id; used by others to link the rooms together
        first_player: who will be the first player (default 1)
    """
    def __init__(self,
                 superpower,
                 tokens=None,
                 next_room: int = None,
                 first_player: int = 1):
        self.superpower = superpower
        self.next_room = next_room
        self.first_player = first_player
        self.env = Env(superpower=superpower, first_player=first_player)
        if tokens is None:
            self.tokens = [None, generate_token(), generate_token()]
        else:
            # token is given with the booked rooms
            # so that we can control for the access
            self.tokens = tokens
        # if token is given, needs token to reset (to start)
        self._need_authen = tokens is not None
        self._reset_cnt = 0

    def reset(self, token=None):
        # print('room reset')
        if self._need_authen:
            # room requires authen
            assert token is not None, 'this room requires authentication'
            player = self.tokens.index(token)
        else:
            # room is open, first to come is the first player
            if self._reset_cnt == 0:
                # print('reset player 1')
                player = 1
            else:
                # print('reset player 2')
                player = 2
        self._reset_cnt += 1
        if self._reset_cnt > 2:
            raise Exception('the room has already started')

        value = self.env.reset(player).wait()
        if isinstance(value, Exception):
            raise value
        value = serializable(value)
        return {
            'return': value[0],
            'legal_moves': value[4],
            'player': player,
            'superpower': self.superpower,
            'token': self.tokens[player],
            'next_room': self.next_room,
        }

    def step(self, a, token):
        player = self.tokens.index(token)
        assert player != -1, 'invalid token'
        value = self.env.step(a, player).wait()
        if isinstance(value, Exception):
            raise value
        value = serializable(value)
        return {
            'return': value[:4],
            'legal_moves': value[4],
            'player': player,
            'superpower': self.superpower,
            'token': self.tokens[player],
            'next_room': self.next_room,
        }


class Store:
    def __init__(self):
        self.rooms = dict()


def generate_id(rooms: dict, max_id=10_000_000):
    assert len(rooms) < max_id, "max_id is too low"
    while True:
        id = random.randint(0, max_id)
        if id not in rooms:
            return id


def create_match(normal_games: int, superpower_games: int, first_room: int,
                 rooms: dict):
    # a match is a series of rooms with alteranting starters with and without
    n = normal_games + superpower_games
    assert n > 0

    first = True

    ids = []
    room_seq = []

    tokens = None

    first_player = 1

    # create normal rooms
    for i in range(normal_games):
        if first:
            first = False
            room_id = first_room
        else:
            # generate an id
            room_id = generate_id(rooms)

        rooms[room_id] = room = Room(superpower=False,
                                     tokens=tokens,
                                     first_player=first_player)
        if first_player == 1:
            first_player = 2
        else:
            first_player = 1

        if tokens is None:
            tokens = room.tokens

        ids.append(room_id)
        room_seq.append(room)

    # create superpower rooms
    for i in range(superpower_games):
        if first:
            first = False
            room_id = first_room
        else:
            # generate an id
            room_id = generate_id(rooms)

        rooms[room_id] = room = Room(superpower=True,
                                     tokens=tokens,
                                     first_player=first_player)
        if first_player == 1:
            first_player = 2
        else:
            first_player = 1

        if tokens is None:
            tokens = room.tokens

        ids.append(room_id)
        room_seq.append(room)

    # link them
    for i in range(len(room_seq) - 1):
        room_seq[i].next_room = ids[i + 1]

    # return the first room
    return room_seq[0]
