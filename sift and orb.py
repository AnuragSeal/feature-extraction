import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image1 = cv2.imread('books.jpg')

# Convert the training image to RGB
training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

# Convert the training image to gray scale
training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
plt.imshow(training_image)
sift = cv2.xfeatures2d.SIFT_create()
orb = cv2.ORB_create()

train_keypoints = sift.detect(training_gray, None)

descriptors1 = orb.compute(training_gray, None)

keypoints_without_size = np.copy(training_image)

cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))

plt.imshow(keypoints_without_size)

# Print the number of keypoints detected in the training image
print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))
plt.show()