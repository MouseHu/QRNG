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
cuda = True
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

maxlen = 10
step = 5
epochs = 100
batch_size = 16384
log_interval = 2
a, b = 'raw','raw'
train_loader = torch.utils.data.DataLoader(
    BinaryQRNGDataset('./data/qrng/Vacuum_Fluctuation/rawdata-5-16-combine1G_150m.dat', split=[0, 0.7],num_class=2048),
    batch_size=batch_size,
    shuffle=True,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    BinaryQRNGDataset('./data/qrng/Vacuum_Fluctuation/rawdata-5-16-combine1G_150m.dat', split=[0.7, 1],num_class=2048),
    batch_size=batch_size,
    shuffle=True, **kwargs)

#model = Warpper(RCNN(batch_size=batch_size)).to(device)
# model = Warpper(SelfAttention(batch_size=batch_size)).to(device)
#model = Warpper(AttentionModel(batch_size=batch_size)).to(device)
model = ResFC(num_classes =2,input_bits=12)
if torch.cuda.device_count()>1:
    print("Let's use {} GPUs".format(torch.cuda.device_count()))
    model = nn.DataParallel(model)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    total_correct = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # print(data.shape)
        optimizer.zero_grad()
        correct, loss = model(x, y)
        loss.sum().backward()
        total_correct += correct.sum().item()
        train_loss += loss.sum().item()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     for p in list(filter(lambda p: p.grad is not None, model.parameters())):
        #         print(p.grad.data.norm(2).item())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},Acc:{:.6f} [{}]'.format(
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.sum().item() / len(x), total_correct / (batch_idx + 1) / batch_size, b, end='\r'))

    if epoch % 3 == 0:
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
            print('===== Test set loss: {:.4f} acc {:.6f} ====='.format(test_loss, total_correct / len(test_loader.dataset)))
            torch.save(model.state_dict(),"./model/FC_{}_{}.model".format(b,epoch))
