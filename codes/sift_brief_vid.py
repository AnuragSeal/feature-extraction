import cv2
import numpy as np
import time

start_time = time.time()
t = []
kp_list = []
fkp_list = []
matches_list = []
fps_list = []
prev_frame_time = 0
new_frame_time = 0

cap = cv2.VideoCapture('out_low.mp4')
#create a Brute Force matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#create an orb object
sift = cv2.SIFT_create()
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

#import all the required images
img = cv2.imread('out_low.JPG')


#convert them to grayscale
#this is required for future processes
gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


#get keypoints for each image using detect method using grayscale images
kp = sift.detect(gray,None)
kp,des = brief.compute(gray,kp)

while True:
    _,frame = cap.read()
    if frame is not None:
        frame = cv2.resize(frame,None,fx=0.4,fy=0.4,interpolation=cv2.INTER_AREA)
        fgray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fkp = sift.detect(fgray,None)
        fkp,fdes = brief.compute(fgray,fkp)
        matches = bf.match(des,fdes)
        matches = sorted(matches, key=lambda x:x.distance)
        fin = cv2.drawMatches(img, kp, frame, fkp, matches, frame, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        kp_list.append(len(kp))
        fkp_list.append(len(fkp))
        matches_list.append(len(matches))
        t.append(time.time() - start_time)
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps_list.append([time.time() - start_time,fps])
        cv2.imshow('WEBCAM', fin)
    else:
        break
    c = cv2.waitKey(1)
    if c == 27:
        break

print("Tinitial: %s"%t[0])
print("Tfinal: %s"%(time.time() - start_time))
print('Average Train Image Keypoints: %d'%int(sum(kp_list)/len(kp_list)))
print('Average Test Image Keypoints: %d'%int(sum(fkp_list)/len(fkp_list)))
print('Average Matches: %d'%int(sum(matches_list)/len(matches_list)))
np.savetxt("sb_OL.csv",
           fps_list,
           delimiter =", ",
           fmt ='% s')