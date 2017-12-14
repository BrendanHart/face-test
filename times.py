import cv2
import numpy as np
import imutils
import os
import time

scale_values = [1.01, 1.1, 1.3, 1.5]
hog_scales = [(4,4), (8,8), (16,16)]

def get_files(root_paths):
    paths = []
    for p in root_paths:
        paths = paths + [os.path.join(root, name)
                            for root, dirs, files in os.walk(p)
                                for name in files]
    return paths

faces = get_files(["faces"])

lbp_cascade = cv2.CascadeClassifier('/usr/share/opencv/lbpcascades/lbpcascade_frontalface.xml')
haar_cascade = cv2.CascadeClassifier('faces.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

if len(faces) < 5:
    sys.exit("Not enough faces!")


for scale_val in scale_values:
    h_time = 0.0
    l_time = 0.0
    for i in range(5):
        img = cv2.imread(faces[i])
        img = imutils.resize(img, width=min(400, img.shape[1]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        start = time.time()
        haar_cascade.detectMultiScale(gray, scale_val, 3)
        h_time += (time.time() - start)

        start = time.time()
        lbp_cascade.detectMultiScale(gray, scale_val, 3)
        l_time += (time.time() - start)


    print "SCALE " + str(scale_val) + " HAAR-" + str(h_time/5.0) + " LBP-" + str(l_time/5.0)

for scale in scale_values:
    g_time = 0.0
    for i in range(5):
        img = cv2.imread(faces[i])
        img = imutils.resize(img, width=min(400, img.shape[1]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        start = time.time()
        (rects, weights) = hog.detectMultiScale(gray,
                            winStride=(4,4),
                            padding=(8,8),
                            scale=scale)
        g_time += (time.time() - start)
    print "SCALE " + str(scale) + " HOG-" + str(g_time/5.0)
