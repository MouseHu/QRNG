import numpy as np


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
