import numpy as np
import matplotlib.pyplot as plt

fnames = ['fast_brief','Fast_orb','ORB_BF','Orb_brief','sift_brief','sift_bf']

x = np.arange(6)
'''initial time
y1 = [0.0625, 0.124, 1.20, 3.391, 0.203, 0.312]'''
''''#final time
y2 = [31.90, 19.03, 15.62, 15.86, 25.43, 56.72]'''
#average training keypionts
#y3 = [1538, 1474, 500, 500, 765, 973]
#average testing keypionts
#y4 = [652, 588, 316, 316, 198, 374]
#average matches
y5 = [179, 163, 110, 316, 70, 160]
width = 0.2


plt.bar(x, y5, width, color='cyan')
#plt.bar(x, y2, width, color='orange')
#plt.bar(x, y3, width, color='green')

#plt.bar(x, y4, color='brown')

#plt.bar(x, y3, color='blue')



plt.xticks(x, fnames)
#plt.xlabel("Features")
#plt.ylabel("Values")
#plt.legend(["Tinitial"])
plt.legend(["avg matches"])
plt.show()