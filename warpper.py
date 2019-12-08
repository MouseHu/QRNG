import numpy as np
import torch
from torch import nn


class Warpper(nn.Module):
    def __init__(self, inner_module):
        super(Warpper,self).__init__()
        self.inner_module = inner_module

    def forward(self,x,y):
        logits=self.inner_module(x.long())
        # print(logits.shape)
        loss = nn.CrossEntropyLoss()(logits,y)
        _, predict = torch.max(logits, dim=1)
        correct = (predict == y).sum()

        return correct, loss