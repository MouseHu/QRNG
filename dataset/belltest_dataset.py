import torch
import torch.utils.data
import numpy as np
from dataset.dataset import RNGDataset


class BellTestDataset(RNGDataset):
    """
    Dataset for Bell Test data.
    X   -> Alice    ->  a
    Y   -> Bob      ->  b
    """

    def __init__(self, file_dir, seqlen=100, split=(0, 0.7), nbits=8):
        super(BellTestDataset, self).__init__(nbits)
        self.file_dir = file_dir
        self.seqlen = seqlen
        self.split = split
        self.xy, self.ab = self.process_data(file_dir, split)
        self.size = len(self.ab) - seqlen
        self.split = []

    def __len__(self):
        return self.size

    def process_data(self, file_dir, split):
        raw_data = self.load_data(file_dir)
        print(len(raw_data))
        raw_data = raw_data[int(split[0] * len(raw_data)):int(split[1] * len(raw_data))]
        print(len(raw_data))
        # xy = [x >> 5 for x in raw_data] + [(x >> 1) % 8 for x in raw_data]
        xy = np.concatenate([(raw_data >> 5).reshape(-1, 1), ((raw_data >> 1) % 8).reshape(-1, 1)], axis=1)
        # ab = [(x >> 4) % 2 for x in raw_data] + [x % 2 for x in raw_data]
        ab = np.concatenate([((raw_data >> 4) % 2).reshape(-1, 1), (raw_data % 2).reshape(-1, 1)], axis=1)
        xy, ab = xy.reshape(-1), ab.reshape(-1)
        xy, ab = torch.from_numpy(xy), torch.from_numpy(ab)
        return xy, ab

    def __getitem__(self, item):
        assert 0 <= item < self.size
        if item % 2 == 0:
            start = item
        else:
            start = item - 1
        return self.xy[start:start + self.seqlen].long(), self.ab[start:start + self.seqlen].long(), \
               self.xy[item + self.seqlen].long(), self.ab[item + self.seqlen].long()
