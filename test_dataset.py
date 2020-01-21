import os
import numpy as np
from dataset import BinaryQRNGDataset
import matplotlib.pyplot as plt


def load_data(file_dir):
    if os.path.isdir(file_dir):
        files = os.listdir(file_dir)
        files = [f for f in files if '.dat' in f]
        data = [np.fromfile(f, dtype=np.uint8) for f in files]
        return np.concatenate(data)
    else:
        return np.fromfile(file_dir, dtype=np.uint8)


dataset = BinaryQRNGDataset("./data2/vacuum/", split=[0,0.7],
                            num_class=256,
                            nbits=12)
# dataset = load_data("./data/qrng/IDQ/1G/")
data = dataset.data.numpy()
y = [data[dataset.step * item + dataset.split[0] + dataset.maxlen]%(2**0)==0 for item in range(len(dataset))]

y = np.array(y)

y = np.bincount(y)
x = np.arange(len(y))

print(np.max(y) / np.sum(y))
plt.plot(x, y)
plt.savefig("bincount.png")
