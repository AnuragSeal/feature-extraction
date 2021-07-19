from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

p = '/home/mrunal/Python Projects/videoconv/csv/sunny_matches_vs_time/'


files = ['agast.csv','akaze.csv','brief.csv','brisk.csv','fast.csv','gftt.csv','kaze.csv','mser.csv','orb.csv','sift.csv','surf.csv']


fnames = ['AGAST','AKAZE','BRIEF','BRISK','FAST','GFTT','KAZE','MSER','ORB','SIFT','SURF']



fig = plt.figure('sunny_matches_vs_time')

for i in range(0,len(fnames)):
    df = pd.read_csv(p+files[i])
    df['fps'] = df.iloc[:,1].rolling(20).mean()
    df['time'] = df.iloc[:,0]
    x = np.array(df.time)
    y = np.array(df.fps)
    plt.plot(x,y,label=fnames[i])

plt.xlabel("Execution Time")
#plt.ylabel("FPS")
#plt.ylabel("KEYPOINTS")
plt.ylabel("MATCHES")
plt.legend()
plt.show()