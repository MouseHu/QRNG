import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from dataset import QRNGDataset
from dataset import BinaryQRNGDataset
from fc import ResFC
from lstm_attn import AttentionModel
from self_attn import SelfAttention
from rcnn import RCNN
from warpper import Warpper
import argparse

parser = argparse.ArgumentParser(description='QRNG Argparser')
parser.add_argument("--predict", type=int, default=0,dest="predict")
parser.add_argument("--batch_size", type=int, default=8192,dest="batch_size")
parser.add_argument("--cuda", type=bool, default=True,dest="cuda")
parser.add_argument("--epochs", type=int, default=40,dest="epochs")
parser.add_argument("--seqlen", type=int, default=100,dest="seqlen")
args = parser.parse_args()

device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

maxlen = 30
step = 1
log_interval = 20
a, b = 'raw_150', 'raw_150'

train_loader = torch.utils.data.DataLoader(
#    BinaryQRNGDataset('./data2/vacuum/**/150m*/**/raw*.dat', split=[0, 0.7], num_class=256, nbits=12, predict_bit=args.predict),
    BinaryQRNGDataset('./data/qrng/Vacuum_Fluctuation/rawdata-5-16-combine1G_150m.dat', split=[0, 0.7], num_class=4096, nbits=12, predict_bit=args.predict,maxlen=args.seqlen),
    batch_size=args.batch_size,
    shuffle=True,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    #BinaryQRNGDataset('./data2/vacuum/**/150m*/**/raw*.dat', split=[0.7, 1], num_class=256, nbits=12, predict_bit=args.predict),
    BinaryQRNGDataset('./data/qrng/Vacuum_Fluctuation/rawdata-5-16-combine1G_150m.dat', split=[0.7, 1],num_class=4096,nbits=12,maxlen=args.seqlen, predict_bit=args.predict),
    batch_size=args.batch_size,
    shuffle=True, **kwargs)

# model = Warpper(RCNN(batch_size=args.batch_size)).to(device)
# model = Warpper(SelfAttention(batch_size=args.batch_size)).to(device)
# model = Warpper(AttentionModel(batch_size=args.batch_size)).to(device)

model = ResFC(num_classes=2, input_bits=12,seqlen=args.seqlen)
if torch.cuda.device_count() > 1:
    print("Let's use {} GPUs".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

for epoch in range(1, args.epochs + 1):
    model.train()
    train_loss = 0
    total_correct = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        correct, loss = model(x, y)
        loss.sum().backward()
        total_correct += correct.sum().item()
        train_loss += loss.sum().item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},Acc:{:.6f} [{}]'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.sum().item() / len(x), total_correct / (batch_idx + 1) / args.batch_size, b, end='\r'))

    if epoch % 1 == 0:
        model.eval()
        test_loss = 0
        total_correct = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                correct, loss = model(x, y)
                # loss.backward()
                total_correct += correct.sum().item()
                train_loss += loss.sum().item()
            test_loss /= len(test_loader.dataset)
            print('===== Test set loss: {:.4f} acc {:.6f} ====='.format(test_loss,
                                                                        total_correct / len(test_loader.dataset)))
            torch.save(model.state_dict(), "./model/FC_{}_epoch{}_predict{}.model".format(b, epoch, args.predict))
