import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from network.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class CNN(nn.Module):
    def __init__(self, num_classes=256, input_bits=8, embed_dim=8, seq_len=100, backbone='resnet50'):
        super(CNN, self).__init__()
        self.input_bits = input_bits
        self.seq_len = seq_len
        # self.embed = nn.Embedding(256, 256)
        self.embed = nn.Embedding(2 ** input_bits, embed_dim)
        self.backbone = {
            'resnet18': resnet18(n_classes=num_classes, input_channels=embed_dim),
            'resnet34': resnet34(n_classes=num_classes, input_channels=embed_dim),
            'resnet50': resnet50(n_classes=num_classes, input_channels=embed_dim),
            'resnet101': resnet101(n_classes=num_classes, input_channels=embed_dim),
            'resnet152': resnet152(n_classes=num_classes, input_channels=embed_dim),
        }.get(backbone, 'resnet50')

    def forward(self, x, y):
        embedding = self.embed(x.long())
        input_text = embedding.permute(0, 2, 1)
        resnet_out = self.backbone(input_text)

        loss = nn.CrossEntropyLoss()(resnet_out, y)
        _, predict = torch.max(resnet_out, dim=1)
        correct = (predict == y).sum()

        return correct, loss, None


class BellCNN(nn.Module):
    def __init__(self, num_classes=256, xy_bits=3, ab_bits=1, seq_len=100, backbone='resnet50'):
        super(BellCNN, self).__init__()
        self.xy_bits = xy_bits
        self.ab_bits = ab_bits
        self.seq_len = seq_len
        embed_dim_xy, embed_dim_ab = xy_bits, ab_bits
        # self.embed = nn.Embedding(256, 256)
        self.embed_xy = nn.Embedding(2 ** xy_bits, embed_dim_xy)
        self.embed_ab = nn.Embedding(2 ** ab_bits, embed_dim_ab)
        embed_dim = embed_dim_xy + embed_dim_ab
        backbone_func = {
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152,
        }.get(backbone, 'resnet50')
        self.backbone = backbone_func(n_classes=num_classes, input_channels=embed_dim)

    def forward(self, xy_seq, ab_seq, xy, ab):
        embedding_xy = self.embed_xy(xy_seq)
        embedding_ab = self.embed_ab(ab_seq)
        embedding = torch.cat([embedding_xy, embedding_ab], dim=-1)
        input_text = embedding.permute(0, 2, 1)
        resnet_out = self.backbone(input_text)

        out = resnet_out.reshape(-1, 2 ** self.xy_bits, 2 ** self.ab_bits)
        # x = nn.Softmax(dim=2)(x)
        # print(x.shape,y.shape)
        xy = xy.unsqueeze(-1).unsqueeze(-1)
        xy2 = xy.expand((-1, -1, 2 ** self.ab_bits))
        out = torch.gather(out, dim=1, index=xy2).squeeze()

        loss = nn.CrossEntropyLoss()(out, ab)
        maxp, predict = torch.max(out, dim=1)
        # maxp = maxp.exp() /torch.sum(out.exp(), dim=1)
        sum_correct = (predict == ab).sum()
        xy = xy.detach().cpu().numpy().squeeze()
        correct = (predict == ab).detach().cpu().numpy()
        distribution = []
        # maxp, predict = maxp.detach().cpu().numpy(), predict.detach().cpu().numpy()
        for i in range(2 ** self.xy_bits):
            # print(correct,xy)
            distribution.append((i, np.sum(correct[xy == i]), np.sum(xy == i)))
        # print(np.sum(correct))
        info = {
            "distribution": distribution,
            # "max_prop": maxp,
            # "prediction": predict,
            # "correct": ab.detach().cpu().numpy().squeeze(),
            # "xybit": xy,

        }

        return sum_correct, loss, info
