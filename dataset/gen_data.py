import numpy as np

m_ = 32
data_size = 2 ** 20
count = 0


def rngint(nbit=8, m=2 ** m_, a=1103515245, c=12345):
    global count
    count += 1
    # if count % (data_size // 1000) == 0:
    #     print(count)
    return np.uint8(rng() * (2 ** nbit))


def rng(m=2 ** m_, a=1103515245, c=12345):
    rng.current = (a * rng.current + c) % m
    return float(rng.current) / m


# setting the seed
rng.current = 10
for i in range(2**10):
    print(i)
    data = np.array([rngint() for _ in range(data_size)], dtype=np.uint8)
    data.tofile('./data/crng_nist/CRNG_M{}_{}.dat'.format(m_,i))

# data = np.array([rngint() for _ in range(data_size * 1000000)], dtype=np.uint8)
# data.tofile('./data/CRNG_{}M_M{}_test.bin'.format(data_size, m_))
