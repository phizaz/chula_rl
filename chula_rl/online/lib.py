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
    def __init__(self, superpower):
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
        player = 1
        q = [None, self.q1, self.q2]
        states = [None, 'reset', 'reset']
        not_ret = None
        while True:
            op, a, ret = q[player].get()
            try:
                assert op == states[
                    player], f'state {op} expects {states[player]}'

                if op == 'reset':
                    if player == 1:
                        # first time case
                        s = self.env.reset()
                        legal_moves = self.env.legal_moves()
                        ret.set((s, None, None, None, legal_moves))
                        states[player] = 'step'
                        player = 2
                    else:
                        states[player] = 'step'
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
            self.q1.put(cmd)
        elif player == 2:
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
    def __init__(self):
        self.env = Env(superpower=False)
        self.tokens = [None, generate_token(), generate_token()]
        self.first = True

    def reset(self):
        if self.first:
            player = 1
        else:
            player = 2
        self.first = False
        value = self.env.reset(player).wait()
        if isinstance(value, Exception):
            raise value
        value = serializable(value)
        return {
            'return': value[0],
            'legal_moves': value[4],
            'player': player,
            'token': self.tokens[player],
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
            'token': self.tokens[player],
        }


class Store:
    def __init__(self):
        self.rooms = defaultdict(Room)
