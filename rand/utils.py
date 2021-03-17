import numpy as np
import pickle


data = np.fromfile('/home/wth/rand/x.dat', dtype=np.uint8).astype(np.float)
print(data.shape)
print(data.max())
print(data.min())

data1 = []
data2 = []

ite = 0

while True:
    if ite * 800 + 800 <= 134217696:
        if ite % 100 >= 93:
            data2.append(data[ite * 800: ite * 800 + 800])
        else:
            data1.append(data[ite * 800: ite * 800 + 800])
    else:
        break

    if ite % 10000 == 1:
        print('iteration:', ite, len(data1), len(data2))

    ite += 1

data1 = np.concatenate(data1)
data2 = np.concatenate(data2)

print(data1.shape)
print(data2.shape)

with open('/home/wth/rand/xTrain.pkl', 'wb') as f:
    pickle.dump(data1, f)

with open('/home/wth/rand/xTest.pkl', 'wb') as f:
    pickle.dump(data2, f)