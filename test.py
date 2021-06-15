import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('outdoor.png', -1)
#cv2.imshow('image2', img)
plt.imshow(img)
plt.show()