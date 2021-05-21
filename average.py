import numpy as np
import cv2

from os import listdir, path

images = np.array([cv2.imread(path.join("spade", i)).astype(float) for i in listdir("spade")])

average = np.sum(images, axis=0) / images.shape[0]

cv2.imwrite("average.jpg", average)