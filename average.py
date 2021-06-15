import numpy as np
import numpy as np
import pandas as pd

path = 'pythonProject1'


files = ['fast_brief.csv','orbbrief.csv','sift_brief.csv','ORB_BF.csv','siftvid.csv','Fastorb.csv']

fnames = ['fast_brief','orb_brief','sift_brief','ORB','sift','fast_orb']

d = []

for i in range(0,len(files)):
    data = pd.read_csv(path+files[i])
    data = np.array(data.iloc(1)[1])
    data = sum(data)/len(data)
    d.append(data)

fig = plt.figure(figsize = (10, 5))

plt.bar(fnames, d, color ='maroon',
        width = 0.4)

plt.xlabel("Feature Trackers")
plt.ylabel("Avg FPS")
plt.show()