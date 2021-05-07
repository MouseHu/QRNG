import os
import pickle
import numpy as np
# from dataset.util import result2int, result2int_with_nan, compute_s_value
def str2int(x):
    if 'NaN' in x:
        return 1
    else:
        return int(x) + 1


def result2int(result):
    i, duel = result
    i = (i % 4) * 2
    a, b = duel
    a, b = (int(a) + 1) // 2, (int(b) + 1) // 2
    return ((i * 2 + a) << 4) + ((i + 1) * 2 + b)


def result2int_with_nan(result):
    i, duel = result
    i = (i % 4) * 2
    a, b = duel
    a, b = str2int(a), str2int(b)
    return [i * 4 + a, (i + 1) * 4 + b]


def compute_s_value(data):
    data = np.array(data)
    data = data.reshape(-1, 4, 2) % 4
    valid = (data[:, :, 0] != 1).reshape(-1)
    invalid = np.any(data[:, :, 0] == 1, axis=1)
    ab = (data[:, :, 0] - 1) * (data[:, :, 1] - 1)
    s = ab[:, 0] + ab[:, 1] + ab[:, 2] - ab[:, 3]
    s[invalid] = -5
    # print(s)
    s = np.repeat(s, 4)
    s = s[valid]
    return s

file_dir = "/data1/Bell_test_9994-28089"
s_value_save_dir = "/data1/bell_test_9994_s_value.dat"
save_dir = "/data1/bell_test_9994_sorted.dat"
total_data = 0


def read_file(filedir):
    global total_data
    print(filedir)
    with open(filedir, "r") as f:
        data = f.readlines()
        total_data += len(data)
        data = [(i, d.strip("\n").split("\t")) for i, d in enumerate(data)]
        assert len(data) % 4 == 0
        s_value = compute_s_value([result2int_with_nan(d) for d in data])
        data = [result2int(d) for d in data if 'NaN' not in d[1]]

        data = np.array(data).astype(np.uint8).reshape(-1)
        s_value = np.array(s_value + 5).astype(np.uint8).reshape(-1)  # s \in [1,9]
        assert len(data), len(s_value)
    # return np.arange(1)
    return data, s_value
    # print(data.shape)


def read_data(filedir):
    if os.path.isdir(filedir):
        files = os.listdir(filedir)
        files = [f for f in files if '.dat' in f]
        files.sort(key=lambda name: int(name.split("_")[0]))
        result = [read_file(os.path.join(filedir, f)) for f in files]
        data = [r[0] for r in result]
        s_value = [r[1] for r in result]
        return np.concatenate(data), np.concatenate(s_value)
    else:
        return read_file(filedir)


data,s_value = read_data(file_dir)
# data = read_file(file_dir + "/2783_RRT_210314_2.4343.dat")

print(data.shape)
print(s_value.shape)
print(total_data)
with open(save_dir, "wb") as f:
    data.tofile(f)

with open(s_value_save_dir, "wb") as f:
    s_value.tofile(f)
