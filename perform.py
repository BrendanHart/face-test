import cv2
import numpy as np
import imutils
import os

NUM_NEGATIVES = 800
NUM_POSITIVES = 800

scale_values = [1.01, 1.1, 1.3, 1.5]
strides = [(4,4), (8,8), (16,16)]
neighbour_values = [2, 3, 4, 5, 6]

def get_files(root_paths):
    paths = []
    for p in root_paths:
        paths = paths + [os.path.join(root, name)
                            for root, dirs, files in os.walk(p)
                                for name in files]
    return paths

faces = get_files(["faces"])
nofaces = get_files(["nofaces"])

if len(faces) < NUM_POSITIVES:
    sys.exit("Not enough positives!")

if len(nofaces) < NUM_NEGATIVES:
    sys.exit("Not enough negatives!")

lbp_cascade = cv2.CascadeClassifier('/usr/share/opencv/lbpcascades/lbpcascade_frontalface.xml')
haar_cascade = cv2.CascadeClassifier('faces.xml')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#for scale_val in scale_values:
#    for neighbour_val in neighbour_values:
#
#        h_true_negatives = 0
#        h_false_negatives = 0
#        h_true_positives = 0
#        h_false_positives = 0
#
#        l_true_negatives = 0
#        l_false_negatives = 0
#        l_true_positives = 0
#        l_false_positives = 0
#
#
#        positives_used = 0
#        negatives_used = 0
#
#        for path in faces:
#            
#            print "SCALE " + str(scale_val) + " NEIGHBOURS " + str(neighbour_val) + ": " + str(100*(positives_used+negatives_used)/float(NUM_POSITIVES+NUM_NEGATIVES))+"%"
#
#            if positives_used >= NUM_POSITIVES:
#                break
#
#            img = cv2.imread(path)
#            img = imutils.resize(img, width=min(400, img.shape[1]))
#            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#            if len(haar_cascade.detectMultiScale(gray, scale_val, neighbour_val)) > 0:
#                h_true_positives += 1
#            else:
#                h_false_negatives += 1
#
#            if len(lbp_cascade.detectMultiScale(gray, scale_val, neighbour_val)) > 0:
#                l_true_positives += 1
#            else:
#                l_false_negatives += 1
#
#
#            positives_used += 1
#
#        for path in nofaces:
#
#            print "SCALE " + str(scale_val) + " NEIGHBOURS " + str(neighbour_val) + ": " + str(100*(positives_used+negatives_used)/float(NUM_POSITIVES+NUM_NEGATIVES))+"%"
#
#            if negatives_used >= NUM_NEGATIVES:
#                break
#            
#            img = cv2.imread(path)
#            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#            if len(haar_cascade.detectMultiScale(gray, scale_val, neighbour_val)) > 0:
#                h_false_positives += 1
#            else:
#                h_true_negatives += 1
#
#            if len(lbp_cascade.detectMultiScale(gray, scale_val, neighbour_val)) > 0:
#                l_false_positives += 1
#            else:
#                l_true_negatives += 1
#
#            negatives_used += 1
#
#        h_precision = (float(h_true_positives)/float(h_true_positives+h_false_positives))
#        h_recall = (float(h_true_positives)/float(h_true_positives+h_false_negatives))
#        l_precision = (float(l_true_positives)/float(l_true_positives+l_false_positives))
#        l_recall = (float(l_true_positives)/float(l_true_positives+l_false_negatives))
#
#        print "SCALE: " + str(scale_val) + " NEIGHBOURS: " + str(neighbour_val)
#        print "HAAR RESULT: " + str( float(2 * h_precision * h_recall) / float(h_precision + h_recall))
#        print "LBP RESULT: " + str( float(2 * l_precision * l_recall) / float(l_precision + l_recall))

for scale in scale_values:
    for stride in strides:

        g_true_negatives = 0
        g_false_negatives = 0
        g_true_positives = 0
        g_false_positives = 0

        positives_used = 0
        negatives_used = 0

        for path in faces:
            
            print "SCALE " + str(scale) + " STRIDE " + str(stride) + " " + str(100*(positives_used+negatives_used)/float(NUM_POSITIVES+NUM_NEGATIVES))+"%"

            if positives_used >= NUM_POSITIVES:
                break

            img = cv2.imread(path)
            img = imutils.resize(img, width=min(400, img.shape[1]))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            (rects, weights) = hog.detectMultiScale(gray,
                                winStride=stride,
                                padding=(stride[0]*2, stride[1]*2),
                                scale=scale)
            if len(rects) > 0:
                g_true_positives += 1
            else:
                g_false_negatives += 1

            positives_used += 1

        for path in nofaces:

            print "SCALE " + str(scale) + " STRIDE " + str(stride) + " " + str(100*(positives_used+negatives_used)/float(NUM_POSITIVES+NUM_NEGATIVES))+"%"

            if negatives_used >= NUM_NEGATIVES:
                break
            
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            (rects, weights) = hog.detectMultiScale(gray,
                                winStride=stride,
                                padding=(stride[0]*2, stride[1]*2),
                                scale=scale)
            if len(rects) > 0:
                g_false_positives += 1
            else:
                g_true_negatives += 1

            negatives_used += 1

        g_precision = (float(g_true_positives)/float(g_true_positives+g_false_positives))
        g_recall = (float(g_true_positives)/float(g_true_positives+g_false_negatives))

        print "SCALE " + str(scale) + " STRIDE " + str(stride) + " HOG RESULT: " + str( float(2 * g_precision * g_recall) / float(g_precision + g_recall))
