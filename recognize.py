import cv2
import numpy as np
from os import listdir, path
from sys import argv
import matplotlib.pyplot as plt

from numpy.random import random

# Constants
epsilon = 0.1
thresh = 0.8

# Read test image
testImage = cv2.imread(argv[2])

# Create mask from contour
edges = cv2.Canny(cv2.cvtColor(cv2.blur(testImage, (3, 3)), cv2.COLOR_BGR2GRAY), 100, 255)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
mask = np.zeros(testImage.shape[:2], dtype=np.uint8)
for contour in contours:
    #approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour,True),True)
    #mask = cv2.fillPoly(mask, [np.reshape(approx, (approx.shape[0], 2))], (255))
    mask = cv2.fillPoly(mask, [np.reshape(contour, (contour.shape[0], 2))], (255))

cv2.imwrite("mask.jpg", mask) # FIXME: output mask
cv2.imwrite("edges.jpg", edges) # FIXME: output edges
cv2.imwrite("contour.jpg", cv2.drawContours(np.copy(testImage), contours, -1, (0,255,0), 3)) # FIXME: output contours

# Get contour of mask
maskContours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Create average template
#templates = np.array([np.float32(cv2.imread(path.join(argv[1], i))) for i in listdir(argv[1])])
#tempImage = np.uint8(cv2.cvtColor(np.sum(templates, axis=0) / templates.shape[0], cv2.COLOR_BGR2GRAY))

#cv2.imwrite("template.jpg", tempImage) # FIXME: output template

# Get SIFT features
sift = cv2.SIFT_create()
testKp, testDes = sift.detectAndCompute(cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY), mask)

cv2.imwrite("test_sift.jpg", cv2.drawKeypoints(np.copy(testImage), testKp, testImage)) # FIXME: output SIFT feature of test image

# Create FLANN matcher
FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(
    {"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
    {"checks": 50}
)
# Train FLANN matcher
templates = []
for tempFile in listdir(argv[1]):
    tempImage = cv2.imread(path.join(argv[1], tempFile))
    tempKp, tempDes = sift.detectAndCompute(tempImage, None)
    flann.add([tempDes])
    templates.append((tempImage, tempKp))

# Get all matches
goodMatches = [m for m, n in flann.knnMatch(testDes, k=2) if m.distance < thresh * n.distance]

# Get recognition matrix
recognition = np.zeros((len(templates), len(maskContours))).astype(np.uint)
for match in goodMatches:
    for contourIdx, contour in enumerate(maskContours):
        if cv2.pointPolygonTest(contour, testKp[match.queryIdx].pt, False) != -1:
            recognition[match.imgIdx, contourIdx] += 1
# Filter out frequencies that's not the maxinum in a template
for tempIdx, tempFreq in enumerate(recognition):
    maxFreq = np.max(tempFreq)
    for contIdx, contFreq in enumerate(tempFreq):
        if contFreq != maxFreq:
            recognition[tempIdx, contIdx] = 0
# Transpose recognition
recognition = recognition.transpose()

# Output matches for each template
for tempIdx, template in enumerate(templates):
    # Draw matches
    tempImage, tempKp = template
    matches = [m for m in goodMatches if m.imgIdx == tempIdx]
    matchImage = cv2.drawMatches(
        testImage, testKp,
        tempImage, tempKp,
        matches, None,
        matchColor=(0,255,0),
        singlePointColor=(0,0,255),
        flags = cv2.DrawMatchesFlags_DEFAULT
    )
    # Draw contour if recognized
    for contIdx, contour in enumerate(maskContours):
        if tempIdx == np.argmax(recognition[contIdx]):
            cv2.drawContours(matchImage, contour, -1, (255,0, 0), 3)
    cv2.imwrite(path.join("output", "match_{}.jpg".format(tempIdx)), matchImage) # FIXME: output SIFT matches
