import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''files = ['fast_brief.csv']

fnames = ['fast_brief']'''
files = ['sift_brief.csv']

fnames = ['sift_brief']

data = np.genfromtxt("fast_brief.csv", delimiter=",", names=["x", "y"])
plt.plot(data['x'], data['y'])

plt.xlabel("Execution Time")
plt.ylabel("FPS")
plt.legend()
plt.show()
