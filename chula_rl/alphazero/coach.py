import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import attr
import numpy as np
import tensorflow as tf

from .arena import Arena
from .game import Game
from .mcts import MCTS
from .net import Net
from .util.avg import AverageMeter
from .util.bar import Bar


@attr.s(auto_attribs=True)
class Args:
    """configuration of the coach"""
    n_itr: int = 1000
    n_ep: int = 100
    n_itr_drop_temperature: int = 15
    update_thresh: float = 0.6
    n_max_queue: int = 200_000
    n_sims: int = 25
    n_compete: int = 40
    c_puct: float = 1
    checkpoint: str = './checkpoint/'
    load_model: bool = False
    load_folder_file: tuple = ('./checkpoint/', 'best.pth.tar')
    n_train_history_itr: int = 20


class Coach:
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """
    def __init__(self, game: Game, make_net, args: Args):
        self.game = game
        self.nnet = make_net()
        self.pnet = make_net()
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, args.n_sims, args.c_puct)
        # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.trainExamplesHistory = []
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard,pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        # we assume to start with player 1
        self.curPlayer = 1
        episodeStep = 0

        # run for an episode (until terminate)
        while True:
            episodeStep += 1

            # canonical board = a state looked from a given player's perspective
            # for example: if we have "white" and "black" pawns, but we have only one policy which only works with "white" only
            # we could "invert" the board so that all blacks become white and vice versa
            # that is we could use the policy playing "white" but in fact it is playing black (inverted)
            # we always use "canonicalboard" for an input to a neural network
            # because we only have one neural network which must work for both sides
            # the neural network should always think that it is either "white" or "black"
            # what we do is we switch the meaning of being white and black instead
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)

            temp = int(episodeStep < self.args.n_itr_drop_temperature)

            # pi = improved policy after the MCTS has done its job
            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            # sym = symmetries of the current state (and what is the policy on that state)
            # this allows for more generalization across states
            # this is like a kind of augmentation
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append((b, self.curPlayer, p, None))

            # take an action from the improved policy
            action = np.random.choice(len(pi), p=pi)
            # the player will alternate from 1 to -1 to 1 ...
            board, self.curPlayer = self.game.getNextState(
                board, self.curPlayer, action)

            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                # at this time, the episode ends, we get the true reward (either win or lose)
                # assigning target value to all states in sequence (that led to the end of episode)
                # (board, player, prob, None) => (board, prob, v)
                # v alternates between first and second players
                # v = reward (1, -1) * (alternate)
                return [(x[0], x[2], r * ((-1)**(x[1] != self.curPlayer)))
                        for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximium length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        n_improve = 0

        # we run the training loop for n_itr iterations
        # that means we will improve our policy for n_itr times
        for i in range(1, self.args.n_itr + 1):
            print('------ITER ' + str(i) + '------')

            # collect data from MCTS-improved policy
            # that is we use MCTS to improve each of our decision (based on the network)
            # the collected data manifest a "better" policy which could be thought of as a policy improvement
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([],
                                               maxlen=self.args.n_max_queue)

                eps_time = AverageMeter()
                bar = Bar('Self Play', max=self.args.n_ep)
                end = time.time()

                # we collect data for n_ep
                # each new ep we clear the tree, start a new blank tree
                for eps in range(self.args.n_ep):
                    self.mcts = MCTS(self.game, self.nnet, self.args.n_sims,
                                     self.args.c_puct)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                    # bookkeeping + plot progress
                    eps_time.update(time.time() - end)
                    end = time.time()
                    bar.suffix = '({eps}/{maxeps}) Eps Time: {et:.3f}s | Total: {total:} | ETA: {eta:}'.format(
                        eps=eps + 1,
                        maxeps=self.args.n_ep,
                        et=eps_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td)
                    bar.next()
                bar.finish()

                # keep all experience into a database called history which will comprise of many old experience
                self.trainExamplesHistory.append(iterationTrainExamples)

            # if the experience collected so far is deemed too much, we drop some
            if len(self.trainExamplesHistory) > self.args.n_train_history_itr:
                print("len(trainExamplesHistory) =",
                      len(self.trainExamplesHistory),
                      " => remove the oldest trainExamples")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            # make a copy of the network before training
            # the old network => pnet
            # the new network => nnet
            self.nnet.save_checkpoint(folder=self.args.checkpoint,
                                      filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint,
                                      filename='temp.pth.tar')

            # train the network using the data collected from the MCTS-improved policy
            # we just try to move our current network to the MCTS improved policy (and values)
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)
            self.nnet.train(trainExamples)

            # competing the new policy with the old one using Arena class
            # it will compete for n_compete episodes to get a good estimate of its strength
            print('PITTING AGAINST PREVIOUS VERSION')
            pmcts = MCTS(self.game, self.pnet, self.args.n_sims,
                         self.args.c_puct)  # reset search tree
            nmcts = MCTS(self.game, self.nnet, self.args.n_sims,
                         self.args.c_puct)  # reset search tree
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                          self.game)
            pwins, nwins, draws = arena.playGames(self.args.n_compete)
            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' %
                  (nwins, pwins, draws))
            tf.summary.scalar('coach/win_ratio',
                              nwins / (nwins + pwins),
                              step=i)

            # comparing the new policy with the old one
            # the new policy must win more than 60% of the time to be considered as an improvement
            # otherwise it is discarded
            if pwins + nwins == 0 or float(nwins) / (
                    pwins + nwins) < self.args.update_thresh:
                print('REJECTING NEW MODEL')
                # we load the latest saved policy
                self.nnet.load_checkpoint(folder=self.args.checkpoint,
                                          filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                # this number will tell roughly how good our policy has become (how many updates)
                n_improve += 1
                self.nnet.save_checkpoint(folder=self.args.checkpoint,
                                          filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint,
                                          filename='best.pth.tar')

            tf.summary.scalar('coach/n_improve', n_improve, step=i)

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(
            folder,
            self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0],
                                 self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            print(examplesFile)
            r = input("File with trainExamples not found. Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            print("File with trainExamples found. Read it.")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            f.closed
            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
