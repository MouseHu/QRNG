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


dataset = BinaryQRNGDataset("./data/qrng/Vacuum_Fluctuation/Random-key-rawdata-5-16-combine1G_150m.dat", split=[0.7, 1],
                            num_class=256,
                            twelve=False)
# dataset = load_data("./data/qrng/IDQ/1G/")
data = dataset.data.numpy()
y = [data[dataset.step * item + dataset.split[0] + dataset.maxlen] > 127 for item in range(len(dataset))]

y = np.array(y)

y = np.bincount(y)
x = np.arange(len(y))

print(np.max(y) / np.sum(y))
plt.plot(x, y)
plt.show()
