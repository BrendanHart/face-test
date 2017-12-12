import cv2
import numpy as np
import imutils
import os

NUM_NEGATIVES = 800
NUM_POSITIVES = 800

scale_values = [1.01, 1.3, 1.5]
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

for scale_val in scale_values:
    for neighbour_val in neighbour_values:

        h_true_negatives = 0
        h_false_negatives = 0
        h_true_positives = 0
        h_false_positives = 0

        l_true_negatives = 0
        l_false_negatives = 0
        l_true_positives = 0
        l_false_positives = 0

        g_true_negatives = 0
        g_false_negatives = 0
        g_true_positives = 0
        g_false_positives = 0

        positives_used = 0
        negatives_used = 0

        for path in faces:
            
            print "SCALE " + str(scale_val) + " NEIGHBOURS " + str(neighbour_val) + ": " + str(100*(positives_used+negatives_used)/float(NUM_POSITIVES+NUM_NEGATIVES))+"%"

            if positives_used >= NUM_POSITIVES:
                break

            img = cv2.imread(path)
            img = imutils.resize(img, width=min(400, img.shape[1]))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if len(haar_cascade.detectMultiScale(gray, scale_val, neighbour_val)) > 0:
                h_true_positives += 1
            else:
                h_false_negatives += 1

            if len(lbp_cascade.detectMultiScale(gray, scale_val, neighbour_val)) > 0:
                l_true_positives += 1
            else:
                l_false_negatives += 1

            (rects, weights) = hog.detectMultiScale(gray,
                                winStride=(4,4),
                                padding=(8,8),
                                scale=1.05)
            if len(rects) > 0:
                g_true_positives += 1
            else:
                g_false_negatives += 1

            positives_used += 1

        for path in nofaces:

            print "SCALE " + str(scale_val) + " NEIGHBOURS " + str(neighbour_val) + ": " + str(100*(positives_used+negatives_used)/float(NUM_POSITIVES+NUM_NEGATIVES))+"%"

            if negatives_used >= NUM_NEGATIVES:
                break
            
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if len(haar_cascade.detectMultiScale(gray, scale_val, neighbour_val)) > 0:
                h_false_positives += 1
            else:
                h_true_negatives += 1

            if len(lbp_cascade.detectMultiScale(gray, scale_val, neighbour_val)) > 0:
                l_false_positives += 1
            else:
                l_true_negatives += 1

            (rects, weights) = hog.detectMultiScale(gray,
                                winStride=(4,4),
                                padding=(8,8),
                                scale=1.05)
            if len(rects) > 0:
                g_false_positives += 1
            else:
                g_true_negatives += 1

            negatives_used += 1

        print "HAAR RESULT: TP-"+str(h_true_positives)+" FP-"+str(h_false_positives)+" TN-"+str(h_true_negatives)+" FN-"+str(h_false_negatives)
        print "LBP RESULT: TP-"+str(l_true_positives)+" FP-"+str(l_false_positives)+" TN-"+str(l_true_negatives)+" FN-"+str(l_false_negatives)
        print "HOG RESULT: TP-"+str(g_true_positives)+" FP-"+str(g_false_positives)+" TN-"+str(g_true_negatives)+" FN-"+str(g_false_negatives)
