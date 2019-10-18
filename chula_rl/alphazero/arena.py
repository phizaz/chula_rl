"""
based on: https://github.com/suragnair/alpha-zero-general
"""
import time

import numpy as np

from .util.avg import AverageMeter
from .util.bar import Bar


class Arena():
    """An Arena class where any 2 agents can be pit against each other."""
    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while True:
            r = self.game.getGameEnded(board, curPlayer)

            # termination criterion
            # r = 0 means it haven't yet to reach the end
            if r != 0:
                if curPlayer == 1:
                    return r
                elif curPlayer == -1:
                    # if the second player wins, r = 1
                    # but we need to return -1 (to denote the second player)
                    return -r
                else:
                    raise NotImplementedError()

            it += 1
            if verbose:
                assert (self.display)
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.display(board)

            # canonical board = a state looked from a given player's perspective
            # for example: if we have "white" and "black" pawns, but we have only one policy which only works with "white" only
            # we could "invert" the board so that all blacks become white and vice versa
            # that is we could use the policy playing "white" but in fact it is playing black (inverted)
            # we always use "canonicalboard" for an input to a neural network
            action = players[curPlayer + 1](self.game.getCanonicalForm(
                board, curPlayer))

            # we query for valid moves wrt. the canonical board which is always subjective to the view of player 1
            valids = self.game.getValidMoves(
                self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                print(action)
                assert valids[action] > 0

            # take the action and alteranate the player
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        if verbose:
            assert (self.display)
            print("Game over: Turn ", str(it), "Result ",
                  str(self.game.getGameEnded(board, 1)))
            self.display(board)

        # this seems incorrect, because Santorini could lose because of not able to take actions
        # return self.game.getGameEnded(board, 1)

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        eps_time = AverageMeter()
        bar = Bar('Arena.playGames', max=num)
        end = time.time()
        eps = 0
        maxeps = int(num)

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0

        # player 1 starts first for n/2 games
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:} | one/two: {oneWon}/{twoWon}'.format(
                eps=eps,
                maxeps=maxeps,
                et=eps_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                oneWon=oneWon,
                twoWon=twoWon)
            bar.next()

        # player 2 starts first for n/2 games
        self.player1, self.player2 = self.player2, self.player1
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1
            # bookkeeping + plot progress
            eps += 1
            eps_time.update(time.time() - end)
            end = time.time()
            bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:} | one/two: {oneWon}/{twoWon}'.format(
                eps=eps,
                maxeps=maxeps,
                et=eps_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                oneWon=oneWon,
                twoWon=twoWon)
            bar.next()

        bar.finish()

        return oneWon, twoWon, draws
