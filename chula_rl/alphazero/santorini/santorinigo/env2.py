import numpy as np


class Santorini:
    def __init__(self,
                 board_dim=(5, 5),
                 starting_parts=np.array([0, 22, 18, 14, 18]),
                 winning_floor=3):
        self.board_dim = board_dim
        self.starting_parts = starting_parts
        self.winning_floor = winning_floor

        self.moves = ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c']
        self.builds = ['q', 'w', 'e', 'a', 'd', 'z', 'x', 'c']

        # key to coordinates
        self.ktoc = {
            'q': np.array((-1, -1)),
            'w': np.array((-1, 0)),
            'e': np.array((-1, 1)),
            'a': np.array((0, -1)),
            'd': np.array((0, 1)),
            'z': np.array((1, -1)),
            'x': np.array((1, 0)),
            'c': np.array((1, 1))
        }
        # possbile moves, we base them on the first player
        self.itoa = [(w, m, b) for w in [-1, -2] for m in self.moves
                     for b in self.builds]
        # action to index
        self.atoi = {action: index for index, action in enumerate(self.itoa)}
        self.wtoi = {-1: 0, -2: 1, 1: 2, 2: 3}

        self.reset()

    def reset(self):
        self.current_player = -1
        self._parts = self.starting_parts.copy()
        self._workers = np.array([
            (0, 2),
            (4, 2),
            (2, 0),
            (2, 4),
        ])
        self._board = np.zeros(self.board_dim, dtype=np.int64)
        self._done = False
        return self._state

    @property
    def _state(self):
        return {
            'board': self._board,
            'workers': {k: v
                        for k, v in zip(self.wtoi.keys(), self._workers)},
            'parts': self._parts,
        }

    def _walkable(self, wid: int, dir: np.ndarray):
        others, pos = None, None
        # check boundary
        src = self._workers[wid]
        new = src + dir
        if not (0 <= new[0] < self.board_dim[0]): return False, others, pos
        if not (0 <= new[1] < self.board_dim[1]): return False, others, pos

        # not a dome
        tgt = self._board[new[0], new[1]]
        if tgt == 4: return False, others, pos

        # not too high
        cur = self._board[src[0], src[1]]
        if tgt > cur + 1: return False, others, pos

        # no other worker
        others = [v for k, v in enumerate(self._workers) if k != wid]
        for pos in others:
            if np.array_equal(pos, new): return False, others, pos

        # return
        pos = new
        return True, others, pos

    def _buildable(self, src, dir, workers: list):
        part = None
        pos = None

        # check boundary
        new = src + dir
        if not (0 <= new[0] < self.board_dim[0]): return False, part, pos
        if not (0 <= new[1] < self.board_dim[1]): return False, part, pos

        # not a dome
        tgt = self._board[new[0], new[1]]
        if tgt == 4: return False, part, pos

        # no other worker
        for pos in workers:
            if np.array_equal(pos, new): return False, part, pos

        # check parts
        part = tgt + 1
        if self._parts[part] == 0: part = -1  # move without building

        # return
        pos = new
        return True, part, pos

    def legal_moves(self):
        # all possbile moves
        out = []
        for i, (worker, mdir, bdir) in enumerate(self.itoa):
            # invert worker's number if needed
            worker = self.current_player * abs(worker)
            wid = self.wtoi[worker]

            mdir = self.ktoc[mdir]
            bdir = self.ktoc[bdir]
            walkable, others, moved = self._walkable(wid, mdir)
            if walkable:
                buildable, part, built = self._buildable(moved, bdir, others)
                if buildable:
                    out.append(i)
        return out

    def step(self, action):
        assert not self._done, "must reset"
        worker, mdir, bdir = self.itoa[action]

        # invert worker's number if needed
        worker = self.current_player * abs(worker)
        wid = self.wtoi[worker]

        mdir = self.ktoc[mdir]
        bdir = self.ktoc[bdir]
        walkable, others, moved = self._walkable(wid, mdir)
        if walkable:
            buildable, part, built = self._buildable(moved, bdir, others)
            if buildable:
                # move
                self._workers[wid] = moved
                # build
                if part != -1:
                    self._board[built[0], built[1]] = part
                    self._parts[part] -= 1
                # if win (standing on the third floor)
                if self._board[moved[0], moved[1]] == self.winning_floor:
                    reward = 1.
                    done = True
                else:
                    reward = 0.
                    done = False
            else:
                raise ValueError('illegal move')
        else:
            raise ValueError('illegal move')

        self._done = done

        # switch the player
        self.current_player *= -1

        return self._state, reward, done, {}


def state_array(state):
    board = state['board']
    workers = state['workers']
    parts = state['parts']

    w_board = np.zeros_like(board)
    for k, v in workers.items():
        w_board[tuple(v)] = k

    p_board = np.zeros_like(board)
    for i, p in enumerate(parts):
        p_board[i, i] = p

    return np.stack([board, w_board, p_board])