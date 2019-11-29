import numpy as np
from numba import njit


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

    def legal_moves(self):
        # all possbile moves
        out = []
        for i, (worker, mdir, bdir) in enumerate(self.itoa):
            # invert worker's number if needed
            worker = self.current_player * abs(worker)
            wid = self.wtoi[worker]

            mdir = self.ktoc[mdir]
            bdir = self.ktoc[bdir]
            correct, moved, built, part = _check(wid, mdir, bdir,
                                                 self._workers, self._board,
                                                 self._parts)
            if correct:
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
        correct, moved, built, part = _check(wid, mdir, bdir, self._workers,
                                             self._board, self._parts)
        if correct:
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


@njit
def _walkable(
        wid: int,
        dir: np.ndarray,
        workers: np.ndarray,
        board: np.ndarray,
):
    pos = None
    # check boundary
    src = workers[wid]
    new = src + dir
    board_dim = board.shape
    if not (0 <= new[0] < board_dim[0]): return False, pos
    if not (0 <= new[1] < board_dim[1]): return False, pos

    # not a dome
    tgt = board[new[0], new[1]]
    if tgt == 4: return False, pos

    # not too high
    cur = board[src[0], src[1]]
    if tgt > cur + 1: return False, pos

    # no other worker
    for i in range(len(workers)):
        if i != wid:
            oth = workers[i]
            if oth[0] == new[0] and oth[1] == new[1]:
                return False, pos

    # return
    pos = new
    return True, pos


@njit
def _buildable(
        src: np.ndarray,
        dir: np.ndarray,
        wid: int,
        workers: np.ndarray,
        board: np.ndarray,
        parts: np.ndarray,
):
    part = None
    pos = None

    # check boundary
    new = src + dir
    board_dim = board.shape
    if not (0 <= new[0] < board_dim[0]): return False, part, pos
    if not (0 <= new[1] < board_dim[1]): return False, part, pos

    # not a dome
    tgt = board[new[0], new[1]]
    if tgt == 4: return False, part, pos

    # no other worker
    for i in range(len(workers)):
        if i != wid:
            oth = workers[i]
            if oth[0] == new[0] and oth[1] == new[1]:
                return False, part, pos

    # check parts
    part = tgt + 1
    if parts[tgt + 1] == 0: part = -1  # move without building

    # return
    pos = new
    return True, part, pos


@njit
def _check(
        wid: int,
        mdir: np.ndarray,
        bdir: np.ndarray,
        workers: np.ndarray,
        board: np.ndarray,
        parts: np.ndarray,
):
    walkable, moved = _walkable(wid, mdir, workers=workers, board=board)
    if walkable:
        buildable, part, built = _buildable(moved,
                                            bdir,
                                            wid=wid,
                                            workers=workers,
                                            board=board,
                                            parts=parts)
        if buildable:
            return True, moved, built, part
    return False, None, None, None
