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

    def __init__(self, num_classes=256):
        super(ResFC, self).__init__()
        MIDDLE_SHAPE = 2048
        self.input_fc_1 = nn.Linear(100 * 256, 1024)
        self.input_fc_2 = nn.Linear(1024, MIDDLE_SHAPE)
        self.residual = self.make_residual(MIDDLE_SHAPE)
        self.output_fc_1 = nn.Linear(MIDDLE_SHAPE, num_classes)
        # self.embed = nn.Embedding(256, 32)

    def make_residual(self, shape=256, times=5):
        layers = []
        for i in range(times):
            # layers.append(ResBlock(nn.Linear(shape, shape)))
            layers.append(nn.Linear(shape, shape))
            # layers.append(nn.Dropout())
            layers.append(nn.BatchNorm1d(num_features=shape))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = F.one_hot(x.long(), num_classes=256).float()
        # x = self.embed(x.long())
        # print(x.shape,y.shape)
        x = x.reshape(-1, 100 * 256)
        x = nn.ReLU()(self.input_fc_1(x))
        # x = nn.Dropout()(x)
        x = nn.ReLU()(self.input_fc_2(x))
        # x = nn.Dropout()(x)
        x = self.residual(x)

        x = self.output_fc_1(x)
        x = nn.Softmax(dim=1)(x)
        # print(x.shape)
        loss = nn.CrossEntropyLoss()(x, y)
        _, predict = torch.max(x, dim=1)
        correct = (predict == y).sum()

        return correct, loss
