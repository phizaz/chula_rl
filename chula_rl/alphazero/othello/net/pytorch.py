import os
import time

import attr
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from chula_rl.alphazero.net import Net
from chula_rl.alphazero.util.avg import AverageMeter
from chula_rl.alphazero.util.bar import Bar


@attr.s(auto_attribs=True)
class NetArgs:
    lr: float = 0.001
    dropout: float = 0.3
    n_ep: int = 10
    n_bs: int = 64
    cuda: str = torch.cuda.is_available()
    n_ch: int = 256


class OthelloNet(Net):
    def __init__(self, game, args: NetArgs):
        self.args = args
        self.nnet = Backbone(game, args)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()

        if self.args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(self.args.n_ep):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()
            data_time = AverageMeter()
            batch_time = AverageMeter()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            end = time.time()

            bar = Bar('Training Net', max=int(len(examples) / self.args.n_bs))
            batch_idx = 0

            while batch_idx < int(len(examples) / self.args.n_bs):
                sample_ids = np.random.randint(len(examples),
                                               size=self.args.n_bs)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if self.args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(
                    ), target_pis.contiguous().cuda(), target_vs.contiguous(
                    ).cuda()

                # measure data loading time
                data_time.update(time.time() - end)

                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                batch_idx += 1

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss_pi: {lpi:.4f} | Loss_v: {lv:.3f}'.format(
                    batch=batch_idx,
                    size=int(len(examples) / self.args.n_bs),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    lpi=pi_losses.avg,
                    lv=v_losses.avg,
                )
                bar.next()
            bar.finish()

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.from_numpy(board.astype(np.float32))
        if self.args.cuda: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1))**2) / targets.size()[0]

    def save_checkpoint(self,
                        folder='checkpoint',
                        filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".
                  format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self,
                        folder='checkpoint',
                        filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])


class Backbone(nn.Module):
    def __init__(self, game, args):
        super().__init__()
        # game params
        self.args = args
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        self.conv1 = nn.Conv2d(1, self.args.n_ch, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.args.n_ch,
                               self.args.n_ch,
                               3,
                               stride=1,
                               padding=1)
        self.conv3 = nn.Conv2d(self.args.n_ch, self.args.n_ch, 3, stride=1)
        self.conv4 = nn.Conv2d(self.args.n_ch, self.args.n_ch, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(self.args.n_ch)
        self.bn2 = nn.BatchNorm2d(self.args.n_ch)
        self.bn3 = nn.BatchNorm2d(self.args.n_ch)
        self.bn4 = nn.BatchNorm2d(self.args.n_ch)

        self.fc1 = nn.Linear(
            self.args.n_ch * (self.board_x - 4) * (self.board_y - 4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: n_bs x board_x x board_y
        s = s.view(-1, 1, self.board_x,
                   self.board_y)  # n_bs x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))  # n_bs x n_ch x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))  # n_bs x n_ch x board_x x board_y
        s = F.relu(self.bn3(
            self.conv3(s)))  # n_bs x n_ch x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(
            self.conv4(s)))  # n_bs x n_ch x (board_x-4) x (board_y-4)
        s = s.view(-1,
                   self.args.n_ch * (self.board_x - 4) * (self.board_y - 4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))),
                      p=self.args.dropout,
                      training=self.training)  # n_bs x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))),
                      p=self.args.dropout,
                      training=self.training)  # n_bs x 512

        pi = self.fc3(s)  # n_bs x action_size
        v = self.fc4(s)  # n_bs x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
