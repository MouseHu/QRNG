import os
import numpy as np
from scipy import linalg
import cupy as cp

ratio = 0.75
n = 2 ** 10
k = int(np.floor(n * ratio))
L = n + k - 1

raw_data_dir = "../data/crng_nist"
final_data_dir = "../data/crng_nist_m32_final.dat"
seed_dir = "../data/key29"

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
seed = np.unpackbits(seed[::-1], bitorder='little').reshape(-1)
seed = seed[:L]
toeplitz = linalg.toeplitz(seed, seed)[:k, :n]
toeplitz = cp.array(toeplitz)

data_size = len(raw_data)
batch_size = 1024 * 1024
chunk_size = batch_size * (n // 8)
output_datas = []

for i in range(data_size // chunk_size):
    print(i)
    input_data = raw_data[i * chunk_size:(i + 1) * chunk_size]
    input_data = np.unpackbits(input_data).reshape(batch_size, n).T
    input_data = cp.array(input_data)
    output_data = cp.matmul(toeplitz, input_data).T
    output_data = (output_data % 2).reshape(batch_size, k // 8, 8)
    output_data = cp.packbits(output_data)
    output_data = cp.asnumpy(output_data)

    output_datas.append(output_data.reshape(-1))

output_datas = np.array(output_datas).reshape(-1)
output_datas.tofile(final_data_dir)
