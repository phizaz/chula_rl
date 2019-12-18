import matplotlib.pyplot as plt
import numpy as np
import requests

from .client import err_handling

from chula_rl.alphazero.santorini.santorinigo.fast import Santorini


class Replay:
    def __init__(self, host, room_id):
        self.host = host
        self.room_id = room_id

        res = requests.get(f'{self.host}/room/{self.room_id}')
        err_handling(res)

        obj = res.json()
        self.next_room = obj['next_room']
        self.env = Santorini(**obj['trace']['args'])
        self.steps = obj['trace']['steps']
        self.i = -1

    def step(self):
        print(f'step: {self.i+1}/{len(self.steps)}')

        if self.i + 1 == len(self.steps):
            print('Finished!')
            print('next_room:', self.next_room)

        if self.i == -1:
            s = self.env.reset()
            show_board(s)
            self.i += 1
        else:
            s, r, done, info = self.env.step(self.steps[self.i])
            if self.i % 2 == 0:
                s[1] *= -1
            self.i += 1
            show_board(s)


def show_board(board):
    print("Parts:", *np.diag(board[2]))
    img = np.ones(board.shape[1:] + (3, ))
    for i in range(3):
        img[:, :, i] = 1.0
    current = board[1] == -1
    img[:, :, 0] -= 0.6 * current
    img[:, :, 1] -= 0.6 * current
    current = board[1] == -2
    img[:, :, 0] -= 0.3 * current
    img[:, :, 1] -= 0.3 * current
    opponent = board[1] == 1
    img[:, :, 1] -= 0.6 * opponent
    img[:, :, 2] -= 0.6 * opponent
    opponent = board[1] == 2
    img[:, :, 1] -= 0.3 * opponent
    img[:, :, 2] -= 0.3 * opponent

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    r, c = board[0].shape
    for i in range(r):
        for j in range(c):
            ax.text(j,
                    i,
                    board[0, i, j],
                    fontsize='40',
                    ha='center',
                    va='center')

    plt.show()