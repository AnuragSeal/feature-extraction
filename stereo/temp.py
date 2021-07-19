import numpy as np
import matplotlib.pyplot as plt

fnames = ['AGAST','AKAZE','BRIEF','BRISK','FAST','GFTT','KAZE','MSER','ORB','SIFT','SURF']

x = np.arange(11)

y1 = [42.826, 20.226, 82.108, 36.042, 47.792, 43.552, 4.928, 11.082, 55.962, 12.22, 9.44]
y2 = [72.09, 20.924, 94.768, 49.468, 81.118, 41.124, 5.060, 10.3, 63.436, 14.166, 13.596]
y3 = [66.894, 22.525, 101.33, 58.394, 76.102, 51.302, 5.446, 19.742, 71.598, 18.266, 12.166]
y4 = [30.926, 19.44, 78.498, 26.884, 33.886, 39.782, 4.774, 8.36, 48.866, 10.694, 8.186]

width = 0.2

fig = plt.figure('avg_fps')

plt.bar(x-0.3, y1, width, color='red')
plt.bar(x-0.1, y2, width, color='blue')
plt.bar(x+0.1, y3, width, color='violet')
plt.bar(x+0.3, y4, width, color='brown')

plt.xticks(x, fnames)
plt.xlabel("Feature Extractors")
plt.ylabel("fps")
plt.legend(["cloudy","foggy","rainy","sunny"])
plt.show()