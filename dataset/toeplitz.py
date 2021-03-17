import os

import numpy as np
from scipy import linalg

ratio = 0.75
n = 2 ** 10
k = np.floor(n * ratio)
L = n + k - 1

raw_data_dir = "/data1/hh/qrng/"
final_data_dir = "/data1/hh/qrng/"
seed_dir = "/data1/hh/qrng/"

# load data
if os.path.isdir(raw_data_dir):
    files = os.listdir(raw_data_dir)
    files = [f for f in files if '.dat' in f]
    data = [np.fromfile(os.path.join(raw_data_dir, f), dtype=np.uint8) for f in files]
    raw_data = np.concatenate(data)
else:
    raw_data = np.fromfile(raw_data_dir, dtype=np.uint8)

# load seed
seed = np.fromfile(seed_dir, dtype=np.uint8)
seed = np.unpackbits(seed[::-1], bigorder='little').reshape(-1)
seed = seed[:L]
toeplitz = linalg.toeplitz(seed, seed)

data_size = len(raw_data)
output_datas = []
for i in range(data_size * 8 / n):
    input_data = raw_data[i * n // 8:(i + 1) * n // 8]
    input_data = np.unpackbits(input_data).reshape(-1)
    output_data = toeplitz * input_data
    output_data = (output_data % 2).reshape(-1, 8)
    output_data = np.packbits(output_data)
    output_datas.append(output_data)

output_datas = np.array(output_datas).reshape(-1)
output_datas.tofile(final_data_dir)
