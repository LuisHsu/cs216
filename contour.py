import cv2
import numpy as np
from sys import argv

image = cv2.imread(argv[1])
blurred = cv2.blur(image, (3, 3))
edges = cv2.Canny(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY), 100, 255)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Draw mask
epsilon = 0.1
mask = np.zeros(image.shape)
for contour in contours:
    approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour,True),True)
    mask = cv2.fillPoly(mask, [np.reshape(approx, (approx.shape[0], 2))], (255, 255, 255))
    

cv2.imwrite("edges.jpg", edges)
cv2.imwrite("mask.jpg", mask)
#cv2.imwrite("contour.jpg", cv2.drawContours(image, contours, -1, (0,255,0), thickness=cv2.FILLED))