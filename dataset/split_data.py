import os
import numpy as np

raw_data_dir = "../data/CRNG_M32"
final_data_dir = "../data/CRNG_M32_nist/"

# raw_data_dir = "../data/linux-256-192-linux-final.dat"
# final_data_dir = "../data/linux-256-192-linux-final_nist/"

if os.path.isdir(raw_data_dir):
    files = os.listdir(raw_data_dir)
    files = [f for f in files if '.dat' in f]
    data = [np.fromfile(os.path.join(raw_data_dir, f), dtype=np.uint8) for f in files]
    raw_data = np.concatenate(data)
else:
    raw_data = np.fromfile(raw_data_dir, dtype=np.uint8)

num_split = 1000
raw_data = raw_data[:(len(raw_data) // num_split) * num_split]
final_size = len(raw_data) // num_split

if not os.path.exists(final_data_dir):
    os.makedirs(final_data_dir, exist_ok=True)
for i in range(num_split):
    print(i)
    part_data = raw_data[i * final_size:(i + 1) * final_size]
    part_data.tofile(os.path.join(final_data_dir, "{}.dat".format(i)))
