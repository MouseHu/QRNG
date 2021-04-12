import os
import pickle
import numpy as np

file_dir = "/data1/Bell_test_9994-28089"
save_dir = "/data1/bell_test_9994.dat"


def result2int(result):
    i, duel = result
    i = (i % 4) * 2
    a, b = duel
    a, b = (int(a) + 1) // 2, (int(b) + 1) // 2
    return ((i*2 + a) << 4) + ((i + 1)*2 + b)


def read_file(filedir):
    print(filedir)
    with open(filedir, "r") as f:
        data = f.readlines()
        data = [(i, d.strip("\n").split("\t")) for i, d in enumerate(data)]
        data = [result2int(d) for d in data if 'NaN' not in d[1]]
        data = np.array(data).astype(np.uint8)
    return data.reshape(-1)
    # print(data.shape)


def read_data(filedir):
    if os.path.isdir(filedir):
        files = os.listdir(filedir)
        files = [f for f in files if '.dat' in f]
        data = [read_file(os.path.join(filedir, f)) for f in files]
        return np.concatenate(data)
    else:
        return read_file(filedir)


data = read_data(file_dir)
# data = read_file(file_dir + "/2783_RRT_210314_2.4343.dat")

print(data.shape)

with open(save_dir, "wb") as f:
    data.tofile(f)
