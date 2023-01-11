import pandas as pd
import numpy as np

# Import data
data = pd.read_excel('')

# Data forward standardization processing
label_need = data.keys()[1:10]
data1 = data[label_need].values
data2 = data1.copy()
[m, n] = data2.shape
index_all = np.arange(n)

# Forward indicator position
for j in index_all:
    d_max = max(data1[:, j])
    d_min = min(data1[:, j])
    data2[:, j] = (data1[:, j] - d_min) / (d_max - d_min)

# comparability
the = np.std(data2, axis=0)
data3 = data2.copy()
# contradiction
data3 = list(map(list, zip(*data2)))  # Matrix transpose
r = np.corrcoef(data3)  # Find Pearson correlation coefficient
f = np.sum(1 - r, axis=1)
# Information carrying capacity
c = the * f

w = c / sum(c)   # weight
s = np.dot(data2, w)
Score = 100 * s / max(s)  # score
for i in range(0, len(Score)):
    print(f"{data['sample'][i]}：{Score[i]}")
