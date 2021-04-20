import numpy as np
import csv
import matplotlib.pyplot as plt

data_file ="C:/Users/Mouse Hu/Desktop/logs.csv"

with open(data_file) as f:
    data = csv.reader(f)
    success_rate = [row[-2] for row in data]
    success_rate = success_rate[1:]
    success_rate = [float(s) for s in success_rate]
    success_rate = np.array(success_rate)
with open(data_file) as f:
    data = csv.reader(f)
    returns = [row[4] for row in data]
    returns = returns[1:]
    returns = [float(s) for s in returns]
    returns = np.array(returns)

x = np.arange(len(success_rate))
plt.plot(x[::100],returns[::100])
plt.tight_layout()
# plt.plot(x[::10],success_rate[::10])
plt.show()



