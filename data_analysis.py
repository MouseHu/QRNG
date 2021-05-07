import numpy as np
import matplotlib.pyplot as plt

data_dir = "/data1/bell_test_9994.dat"
# data_dir = "/data1/bell_test.dat"

raw_data = np.fromfile(data_dir, dtype=np.uint8)

# xy = np.concatenate([(raw_data >> 5).reshape(-1, 1), ((raw_data >> 1) % 8).reshape(-1, 1)], axis=1)
# ab = np.concatenate([((raw_data >> 4) % 2).reshape(-1, 1), (raw_data % 2).reshape(-1, 1)], axis=1)
# xy, ab = xy.reshape(-1), ab.reshape(-1)
data = np.concatenate([(raw_data >> 4) % 2, raw_data % 2], axis=0)
xya = np.concatenate([raw_data >> 5, (raw_data >> 1) % (2 ** 3)], axis=0)

# print(np.bincount(xya))

print(np.bincount(data))
max_data,total_data = 0,0
for i in range(8):
    sub_data = data[xya == i]
    distribution = np.bincount(sub_data)
    print(distribution)
    print(np.max(distribution) / np.sum(distribution))
    max_data += np.max(distribution)
    total_data+=np.sum(distribution)

print(max_data,total_data,max_data/total_data)
