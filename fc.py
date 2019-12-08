import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from dataset import RNGDataset


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

    def __init__(self, num_classes=256, input_bits=8):
        super(ResFC, self).__init__()
        MIDDLE_SHAPE = 2048
        self.input_bits = input_bits
        # self.num_class = num_classes
        self.input_fc_1 = nn.Linear(20 * (2 ** input_bits), 1024)
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
        x = x.reshape(-1, 20 * (2 ** self.input_bits))
        x = nn.ReLU()(self.input_fc_1(x))
        x = nn.ReLU()(self.input_fc_2(x))
        x = self.residual(x)

        x = self.output_fc_1(x)
        x = nn.Softmax(dim=1)(x)
        loss = nn.CrossEntropyLoss()(x, y)
        _, predict = torch.max(x, dim=1)
        correct = (predict == y).sum()

        return correct, loss
