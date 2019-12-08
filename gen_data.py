import numpy as np

m_ = 32
data_size = 500


def rngint(nbit=8, m=2 ** m_, a=1103515245, c=12345):
    return np.uint8(rng() * (2 ** nbit))


def rng(m=2 ** m_, a=1103515245, c=12345):
    rng.current = (a * rng.current + c) % m
    return float(rng.current) / m


# setting the seed
rng.current = 10

data = np.array([rngint() for i in range(data_size * 1000000)], dtype=np.uint8)
data.tofile('./data/CRNG_{}M_M{}.bin'.format(data_size, m_))

data = np.array([rngint() for i in range(data_size * 1000000)], dtype=np.uint8)
data.tofile('./data/CRNG_{}M_M{}_test.bin'.format(data_size, m_))
