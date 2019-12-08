import torch
import numpy as np
import torch.nn.functional as F


class RNGDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir, maxlen=100, step=3, num_class=256):
        super(RNGDataset, self).__init__()
        print("Loading RNG data...")
        self.file_dir = file_dir
        self.num_class = num_class
        self.data = np.fromfile(file_dir, dtype=np.uint8)

        #sentences = [data[i: (i + maxlen)] for i in range(0, len(data) - maxlen, step)]
        #next_char = [data[i + maxlen] for i in range(0, len(data) - maxlen, step)]
        print("converting to tensor..")
        #self.X = torch.tensor(np.array(sentences))
        #self.y = torch.tensor(np.array(next_char)).long()
        self.data = torch.tensor(self.data)
        self.size = (len(self.data)-maxlen)//step
        self.step = step
        self.maxlen = maxlen
        print("size:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert 0 <= item < self.size
        # print(self.num_class)
        # print(self.X[item])
        start = self.step*item
        return self.data[start:start+self.maxlen], self.data[start+self.maxlen].long()


class BinaryRNGDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir, maxlen=100, step=3, num_class=256):
        super(BinaryRNGDataset, self).__init__()
        print("Loading RNG data...")
        self.file_dir = file_dir
        self.num_class = num_class
        data = np.fromfile(file_dir, dtype=np.uint8)

        sentences = [data[i: (i + maxlen)] for i in range(0, len(data) - maxlen, step)]
        next_char = [data[i + maxlen] > 127 for i in range(0, len(data) - maxlen, step)]
        print("converting to tensor..")
        self.X = torch.tensor(sentences)
        self.y = torch.tensor(np.array(next_char)).long()
        self.size = len(sentences)
        print("size:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert 0 <= item < self.size
        # print(self.num_class)
        # print(self.X[item])
        return self.X[item], self.y[item]


class QRNGDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir, maxlen=100, step=3, num_class=256, split=[0, 1]):
        super(QRNGDataset, self).__init__()
        print("Loading RNG data...")
        self.file_dir = file_dir
        self.num_class = num_class
        self.data = np.fromfile(file_dir, dtype=np.uint8)
        self.size = ((len(self.data) - maxlen) // step)
        self.data = torch.tensor(self.data)
        self.step = step
        self.maxlen = maxlen
        # print(self.size)
        self.split = [int(x * self.size) for x in split]
        # print(self.split)
        self.size = self.split[1] - self.split[0]
        # print("size:", self.size)
        print("size:", (self.split[1] - self.split[0]))

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert 0 <= item < self.size
        # print(self.num_class)
        # print(self.X[item])
        start = self.step * item + self.split[0]
        return self.data[start:start + self.maxlen], self.data[start + self.maxlen].long()
