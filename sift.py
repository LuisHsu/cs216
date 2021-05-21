import numpy as np
import cv2
from sys import argv

sift = cv2.SIFT_create()

tempImage = cv2.imread(argv[1])
testImage = cv2.imread(argv[2])

tempKp, tempDes = sift.detectAndCompute(tempImage, None)
testKp, testDes = sift.detectAndCompute(testImage, None)

FLANN_INDEX_KDTREE = 1
flann = cv2.FlannBasedMatcher(
    {"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
    {"checks": 50}
)
matches = flann.match(testDes, tempDes)
dists = [x.distance for x in matches]
print("Max: {}, Min: {}, Mean: {}".format(np.max(dists), np.min(dists), np.mean(dists)))

# TODO: filter good matches
good = matches[:int(len(matches) / 2)]

# testPts = np.float32([testKp[m.queryIdx].pt for m in good])
# tempPts = np.float32([tempKp[m.trainIdx].pt for m in good])

# cv2.imwrite("temp_sift.jpg", cv2.drawKeypoints(tempImage, tempKp, tempImage))
# test_sift = np.copy(testImage)
# cv2.imwrite("test_sift.jpg", cv2.drawKeypoints(testImage, testKp, test_sift))

cv2.imwrite("match.jpg",cv2.drawMatches(
    testImage, testKp,
    tempImage, tempKp,
    good, None,
    matchColor=(0,255,0),
    singlePointColor=(255,0,0),
    flags = cv2.DrawMatchesFlags_DEFAULT
))

# M, mask = cv2.findHomography(tempPts, testPts, cv2.RANSAC)
# warped = cv2.warpPerspective(testImage, np.linalg.inv(M), (tempImage.shape[1], tempImage.shape[0]))
# cv2.imwrite("warp.jpg", warped)

# matchesMask = mask.ravel().tolist()
# h,w = tempImage.shape[:2]
# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# dst = cv2.perspectiveTransform(pts,M)

# testImage = cv2.polylines(testImage,[np.int32(dst)], True, (0, 0, 255), 2, cv2.LINE_AA)
# cv2.imwrite("output.jpg", testImage)
