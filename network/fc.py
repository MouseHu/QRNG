import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np


# from dataset import RNGDataset


class ResBlock(nn.Module):
    def __init__(self, block, activation=nn.ReLU(), shape=256):
        super(ResBlock, self).__init__()
        self.block = block
        self.activation = activation
        self.bn = nn.BatchNorm1d(num_features=shape)

    def forward(self, x):
        y = self.block(x)
        y = self.bn(y)
        y = self.activation(y)
        return x + y


class ResFC(nn.Module):

    def __init__(self, num_classes=256, input_bits=8, seqlen=100):
        super(ResFC, self).__init__()
        MIDDLE_SHAPE = 4096
        self.input_bits = input_bits
        self.seqlen = seqlen
        self.input_fc_1 = nn.Linear(seqlen * (2 ** input_bits), 1024)
        self.input_fc_2 = nn.Linear(1024, MIDDLE_SHAPE)
        self.residual = self.make_residual(MIDDLE_SHAPE)
        self.output_fc_1 = nn.Linear(MIDDLE_SHAPE, num_classes)
        # self.embed = nn.Embedding(256, 256)

    def make_residual(self, shape=256, times=5):
        layers = []
        for i in range(times):
            # layers.append(ResBlock(nn.Linear(shape, shape)))
            layers.append(nn.Linear(shape, shape))
            layers.append(nn.BatchNorm1d(num_features=shape))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = F.one_hot(x.long(), num_classes=2 ** self.input_bits).float()
        x = x.reshape(-1, self.seqlen * (2 ** self.input_bits))
        x = nn.ReLU()(self.input_fc_1(x))
        x = nn.ReLU()(self.input_fc_2(x))
        x = self.residual(x)

        x = self.output_fc_1(x)
        # x = nn.Softmax(dim=1)(x)
        # print(x.shape,y.shape)
        loss = nn.CrossEntropyLoss()(x, y)
        _, predict = torch.max(x, dim=1)
        correct = (predict == y).sum()

        return correct, loss, None


class BellResFC(nn.Module):

    def __init__(self, num_classes=16, xy_bits=3, ab_bits=1, seqlen=100):
        super(BellResFC, self).__init__()
        MIDDLE_SHAPE = 4096
        self.input_length = (2 ** xy_bits) + (2 ** ab_bits)
        self.seqlen = seqlen
        self.input_fc_1 = nn.Linear(seqlen * self.input_length, 1024)
        self.input_fc_2 = nn.Linear(1024, MIDDLE_SHAPE)
        self.residual = self.make_residual(MIDDLE_SHAPE)
        self.output_fc_1 = nn.Linear(MIDDLE_SHAPE, num_classes)
        # self.embed = nn.Embedding(256, 256)
        self.xy_bits = xy_bits
        self.ab_bits = ab_bits

    def make_residual(self, shape=256, times=5):
        layers = []
        for i in range(times):
            # layers.append(ResBlock(nn.Linear(shape, shape)))
            layers.append(nn.Linear(shape, shape))
            layers.append(nn.BatchNorm1d(num_features=shape))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, xy_seq, ab_seq, xy, ab):
        xy_seq_onehot = F.one_hot(xy_seq, num_classes=2 ** self.xy_bits).float()
        ab_seq_onehot = F.one_hot(ab_seq, num_classes=2 ** self.ab_bits).float()
        xy_seq_onehot = xy_seq_onehot.reshape(-1, self.seqlen, 2 ** self.xy_bits)
        ab_seq_onehot = ab_seq_onehot.reshape(-1, self.seqlen, 2 ** self.ab_bits)
        x = torch.cat([xy_seq_onehot, ab_seq_onehot], dim=-1)
        x = x.reshape(-1, self.input_length * self.seqlen)
        x = nn.ReLU()(self.input_fc_1(x))
        x = nn.ReLU()(self.input_fc_2(x))
        x = self.residual(x)

        x = self.output_fc_1(x)
        x = x.reshape(-1, 2 ** self.xy_bits, 2 ** self.ab_bits)
        # x = nn.Softmax(dim=2)(x)
        # print(x.shape,y.shape)
        xy = xy.unsqueeze(-1).unsqueeze(-1)
        xy2 = xy.expand((-1, -1, 2 ** self.ab_bits))
        x = torch.gather(x, dim=1, index=xy2).squeeze()

        loss = nn.CrossEntropyLoss()(x, ab)
        _, predict = torch.max(x, dim=1)
        sum_correct = (predict == ab).sum()
        xy = xy.detach().cpu().numpy().squeeze()
        correct = (predict == ab).detach().cpu().numpy()
        distribution = []
        for i in range(2 ** self.xy_bits):
            # print(correct,xy)
            distribution.append((i, np.sum(correct[xy == i]), np.sum(xy == i)))
        # print(np.sum(correct))
        return sum_correct, loss, distribution
