import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

image = cv2.imread(argv[1])
edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 80, 255)
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

##plt.imshow(edges, cmap="gray")
for contour in contours:
    print(len(point))
#plt.show()