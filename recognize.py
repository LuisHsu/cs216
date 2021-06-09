import cv2
import numpy as np
from os import listdir, path
from sys import argv

# Constants
thresh = 0.8 # Threshold of distance for good matches
minMatches = 6 # Minimun matches required that a contour needs

# Read test image
testImage = cv2.imread(argv[2])

# Create mask from contour
edges = cv2.Canny(cv2.cvtColor(cv2.blur(testImage, (3, 3)), cv2.COLOR_BGR2GRAY), 100, 255) # Kernel size [3, 3], threshold 100-255
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
mask = np.zeros(testImage.shape[:2], dtype=np.uint8)
for contour in contours:
    mask = cv2.fillPoly(mask, [np.reshape(contour, (contour.shape[0], 2))], (255)) 

cv2.imwrite("mask.jpg", mask) # [OUTPUT] mask
cv2.imwrite("edges.jpg", edges) # [OUTPUT] edges
cv2.imwrite("contour.jpg", cv2.drawContours(np.copy(testImage), contours, -1, (0,255,0), 3)) # [OUTPUT] contours with green color

# Get contour of mask
maskContours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Get SIFT features
sift = cv2.SIFT_create()
splitedTest = testImage.transpose((2, 0, 1))
testKpB, testDesB = sift.detectAndCompute(splitedTest[0], mask)
testKpG, testDesG = sift.detectAndCompute(splitedTest[1], mask)
testKpR, testDesR = sift.detectAndCompute(splitedTest[2], mask)

cv2.imwrite("test_sift_B.jpg", cv2.drawKeypoints(np.copy(testImage), testKpB, testImage)) # [OUTPUT] SIFT feature of test image
cv2.imwrite("test_sift_G.jpg", cv2.drawKeypoints(np.copy(testImage), testKpG, testImage)) # [OUTPUT] SIFT feature of test image
cv2.imwrite("test_sift_R.jpg", cv2.drawKeypoints(np.copy(testImage), testKpR, testImage)) # [OUTPUT] SIFT feature of test image

# Create FLANN matcher
FLANN_INDEX_KDTREE = 1
flannB = cv2.FlannBasedMatcher(
    {"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
    {"checks": 50}
)
flannG = cv2.FlannBasedMatcher(
    {"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
    {"checks": 50}
)
flannR = cv2.FlannBasedMatcher(
    {"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
    {"checks": 50}
)
# Train FLANN matcher
templates = []
for tempFile in listdir(argv[1]):
    tempImage = cv2.imread(path.join(argv[1], tempFile))
    splitedTemp = tempImage.transpose((2, 0, 1))
    tempKpB, tempDesB = sift.detectAndCompute(splitedTemp[0], None)
    tempKpG, tempDesG = sift.detectAndCompute(splitedTemp[1], None)
    tempKpR, tempDesR = sift.detectAndCompute(splitedTemp[2], None)
    flannB.add([tempDesB])
    flannG.add([tempDesG])
    flannR.add([tempDesR])
    templates.append((tempImage, [tempKpB, tempKpG, tempKpR]))

# Get all matches
goodMatchesB = [m for m, n in flannB.knnMatch(testDesB, k=2) if m.distance < thresh * n.distance]
goodMatchesG = [m for m, n in flannG.knnMatch(testDesG, k=2) if m.distance < thresh * n.distance]
goodMatchesR = [m for m, n in flannR.knnMatch(testDesR, k=2) if m.distance < thresh * n.distance]

# Get recognition matrix
recognition = np.zeros((len(templates), len(maskContours))).astype(np.uint)
for matchB in goodMatchesB:
    for contourIdx, contour in enumerate(maskContours):
        if cv2.pointPolygonTest(contour, testKpB[matchB.queryIdx].pt, False) != -1:
            recognition[matchB.imgIdx, contourIdx] += 1
for matchG in goodMatchesG:
    for contourIdx, contour in enumerate(maskContours):
        if cv2.pointPolygonTest(contour, testKpG[matchG.queryIdx].pt, False) != -1:
            recognition[matchG.imgIdx, contourIdx] += 1
for matchR in goodMatchesR:
    for contourIdx, contour in enumerate(maskContours):
        if cv2.pointPolygonTest(contour, testKpR[matchR.queryIdx].pt, False) != -1:
            recognition[matchR.imgIdx, contourIdx] += 1

# Wipe out lesser frequencies
for tempIdx, tempFreq in enumerate(recognition):
    if np.sum(tempFreq) < minMatches:
        recognition[tempIdx] *= np.zeros(recognition.shape[1]).astype(recognition.dtype)
    else:
        maxFreq = np.max(tempFreq)
        for contIdx, contFreq in enumerate(tempFreq):
            if contFreq != maxFreq:
                recognition[tempIdx, contIdx] = 0

# Transpose recognition
recognition = recognition.transpose()
print(recognition)

# Output matches for each template
for tempIdx, template in enumerate(templates):
    # Get template
    tempImage, tempKp = template
    splitedTemp = tempImage.transpose((2, 0, 1))

    # Get RGB template images
    tempImageB = np.copy(splitedTemp)
    tempImageB[1] *= 0
    tempImageB[2] *= 0
    tempImageB = tempImageB.transpose((1, 2, 0))
    tempImageG = np.copy(splitedTemp)
    tempImageG[0] *= 0
    tempImageG[2] *= 0
    tempImageG = tempImageG.transpose((1, 2, 0))
    tempImageR = np.copy(splitedTemp)
    tempImageR[0] *= 0
    tempImageR[1] *= 0
    tempImageR = tempImageR.transpose((1, 2, 0))

    # Draw matches
    matchesB = [m for m in goodMatchesB if m.imgIdx == tempIdx]
    matchImage = cv2.drawMatches(
        testImage, testKpB,
        tempImageB, tempKp[0],
        matchesB, None,
        matchColor=(255, 204, 0),
        singlePointColor=(128, 102, 0),
        flags = cv2.DrawMatchesFlags_DEFAULT
    )

    matchesG = [m for m in goodMatchesG if m.imgIdx == tempIdx]
    matchImage = cv2.drawMatches(
        matchImage, testKpG,
        tempImageG, tempKp[1],
        matchesG, None,
        matchColor=(51, 255, 153),
        singlePointColor=(51, 102, 102),
        flags = cv2.DrawMatchesFlags_DEFAULT
    )

    matchesR = [m for m in goodMatchesR if m.imgIdx == tempIdx]
    matchImage = cv2.drawMatches(
        matchImage, testKpR,
        tempImageR, tempKp[2],
        matchesR, None,
        matchColor=(153, 102, 255),
        singlePointColor=(102, 51, 153),
        flags = cv2.DrawMatchesFlags_DEFAULT
    )

    # Draw contour if recognized
    for contIdx, contour in enumerate(maskContours):
        if (np.sum(recognition[contIdx]) >= minMatches) and (tempIdx == np.argmax(recognition[contIdx])):
            cv2.drawContours(matchImage, contour, -1, (0, 255, 255), 3)
    cv2.imwrite(path.join("output", "match_{}.jpg".format(tempIdx)), matchImage) # [OUTPUT] SIFT matches
