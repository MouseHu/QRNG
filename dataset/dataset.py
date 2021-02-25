import torch
import torch.utils.data
import numpy as np
import torch.nn.functional as F
import os
import glob


class QRNGDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, file_dir, seqlen=20, step=3, num_class=256, split=(0, 1), nbits=12):
        super(QRNGDataset, self).__init__()
        print("Loading RNG data...")
        self.file_dir = file_dir
        self.num_class = num_class
        self.nbits = nbits
        self.data = self.load_data()
        self.size = ((len(self.data) - seqlen) // step)

        self.step = step
        self.seqlen = seqlen
        self.split = [int(x * self.size) for x in split]
        self.size = self.split[1] - self.split[0]
        self.data = self.data[self.split[0] * step:self.split[1] * step + seqlen]
        self.data = torch.from_numpy(self.data.astype(np.int32))
        # print("size:", self.size)
        print("size:", (self.split[1] - self.split[0]))

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        assert 0 <= item < self.size
        start = self.step * item
        return self.data[start:start + self.seqlen], self.data[start + self.seqlen].long()

    def load_data(self):
        if os.path.isdir(self.file_dir):
            files = os.listdir(self.file_dir)
            files = [f for f in files if '.dat' in f]
            data = [self.read_data(os.path.join(self.file_dir, f), nbits=self.nbits) for f in files]
            return np.concatenate(data)
        else:
            return self.read_data(self.file_dir, nbits=self.nbits)

    def read_data(self, data_chunk, nbits=12):
        if nbits != 12:
            return np.fromfile(data_chunk, dtype=np.uint8)
        data = np.fromfile(data_chunk, dtype=np.uint8)
        fst_uint8, mid_uint8, lst_uint8 = np.reshape(data, (data.shape[0] // 3, 3)).astype(np.uint16).T
        fst_uint12 = (fst_uint8 << 4) + (mid_uint8 >> 4)
        snd_uint12 = ((mid_uint8 % 16) << 8) + lst_uint8
        return np.reshape(np.concatenate((fst_uint12[:, None], snd_uint12[:, None]), axis=1), 2 * fst_uint12.shape[0])


class BinaryQRNGDataset(QRNGDataset):
    def __init__(self, file_dir, seqlen=20, step=3, num_class=256, split=(0, 1), nbits=12, predict_bit=0):
        super(BinaryQRNGDataset, self).__init__(file_dir, seqlen, step, num_class, split, nbits)
        self.predict_bit = predict_bit
        print("loaded binary data, using the {}-th bit for prediction".format(predict_bit))

    def __getitem__(self, item):
        assert 0 <= item < self.size
        start = self.step * item
        return self.data[start:start + self.seqlen], (
                self.data[start + self.seqlen] % (2 ** (self.predict_bit + 1)) >= 2 ** (self.predict_bit)).long()


class QRNGDatasetMiddle(QRNGDataset):
    def __init__(self, file_dir, seqlen=20, step=3, num_class=256, split=(0, 1), nbits=12):
        self.half_len = seqlen
        seqlen = seqlen * 2 + 1  # make sure seqlen is odd
        super(QRNGDatasetMiddle, self).__init__(file_dir, seqlen, step, num_class, split, nbits)

    def __getitem__(self, item):
        assert 0 <= item < self.size
        start = self.step * item
        data_pre = self.data[start:start + self.half_len]
        data_post = self.data[start + self.half_len + 1:start + 2 * self.half_len + 1]
        data = torch.cat([data_pre, data_post], dim=0)
        return data, self.data[start + self.half_len + 1]
