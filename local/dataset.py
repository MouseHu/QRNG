import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os


class RNGDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir, maxlen=100, step=3, num_class=256):
        super(RNGDataset, self).__init__()
        print("Loading RNG data...")
        self.file_dir = file_dir
        self.num_class = num_class
        self.data = np.fromfile(file_dir, dtype=np.uint8)
        self.size = (len(self.data) - maxlen) // step
        self.data = torch.tensor(self.data)
        self.step = step
        self.maxlen = maxlen
        # print("size:", self.size)
        print("size:", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert 0 <= item < self.size
        # print(self.num_class)
        # print(self.X[item])
        start = self.step * item
        return self.data[start:start + self.maxlen], self.data[start + self.maxlen].long()


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
    def __init__(self, file_dir, maxlen=100, step=3, num_class=256, split=[0, 1], twelve=True):
        super(QRNGDataset, self).__init__()
        print("Loading RNG data...")
        self.file_dir = file_dir
        self.num_class = num_class
        self.twelve = twelve
        self.data = self.load_data()
        self.size = ((len(self.data) - maxlen) // step)

        self.step = step
        self.maxlen = maxlen
        # print(self.size)
        self.split = [int(x * self.size) for x in split]
        # print(self.split)
        self.size = self.split[1] - self.split[0]
        self.data = self.data[self.split[0] * step:self.split[1] * step + maxlen]
        self.data = torch.tensor(self.data)
        # print("size:", self.size)
        print("size:", (self.split[1] - self.split[0]))

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert 0 <= item < self.size
        # print(self.num_class)
        # print(self.X[item])
        start = self.step * item
        return self.data[start:start + self.maxlen], self.data[start + self.maxlen].long()

    def load_data(self):
        if os.path.isdir(self.file_dir):
            files = os.listdir(self.file_dir)
            files = [f for f in files if '.dat' in f]
            data = [self.read_data(os.path.join(self.file_dir, f), twelve=self.twelve) for f in files]
            return np.concatenate(data)
        else:
            return self.read_data(self.file_dir, twelve=self.twelve)

    def read_data(self, data_chunk, twelve=True):
        if not twelve:
            return np.fromfile(data_chunk, dtype=np.uint8)
        data = np.fromfile(data_chunk, dtype=np.uint8)
        fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
        fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
        snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
        return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])


class BinaryQRNGDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir, maxlen=20, step=3, num_class=256, split=[0, 1], twelve=True):
        super(BinaryQRNGDataset, self).__init__()
        print("Loading RNG data...")
        self.file_dir = file_dir
        self.num_class = num_class
        self.twelve = twelve
        self.data = self.load_data()
        self.size = ((len(self.data) - maxlen) // step)
        self.data = torch.from_numpy(self.data.astype(np.int32))
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
        return self.data[start:start + self.maxlen], (self.data[start + self.maxlen] > 2047).long()

    def load_data(self):
        if os.path.isdir(self.file_dir):
            files = os.listdir(self.file_dir)
            files = [f for f in files if '.dat' in f]
            data = [self.read_data(os.path.join(self.file_dir, f), twelve=self.twelve) for f in files]
            return np.concatenate(data)
        else:
            return self.read_data(self.file_dir, twelve=self.twelve)

    def read_data(self, data_chunk, twelve=True):
        if not twelve:
            return np.fromfile(data_chunk, dtype=np.uint8)
        data = np.fromfile(data_chunk, dtype=np.uint8)
        fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
        fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
        snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
        return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])
