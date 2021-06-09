# Poker card recognize

## Prerequisite

* Python 3

* OpenCV-python

* Numpy

## Run

Before running, a folder named `output` should be created in the working directory

```shell
python3 recognize.py <template_folder> <test_image>
```

Arguments:

* template_folder: Folder that contains template images

* test_image: The file name of test image

## Description of code

### Constants

There are 2 constants in the implementions:

* thresh = 0.8
    Threshold of distance for good matches
* minMatches = 6
    Minimun matches required that a contour needs

### Read test image

Use `cv2.imread` and pass `<test_image>` argument from command line (argv[2]).

### Create mask from contour

#### Canny edge detection

* Blur test image with kernel size 3 x 3
* Convert to grayscale
* Call `cv2.Canny` to perform Canny detection with threshold 100 to 255

#### Find contours

Call `cv2.findContours` with `cv2.RETR_LIST` that indicate the return value as a List, and pass `cv2.CHAIN_APPROX_NONE` that make OpenCV not to simplify the contours.

#### Draw mask

* Create a black mask by all zero value with the same size of test image.
* Use `cv2.fillPoly` that draw the contours with white color.

#### Get contour of mask

Find contours with the same setting as test image, but change input image to the mask

#### Get SIFT features

* Create SIFT object
* Split RGB channel, and call `sift.detectAndCompute` with the mask to get keypoints and descriptors.

#### Create FLANN matcher

Create 3 FLANN matcher with KD Tree algrithm and 50 checks.

#### Train FLANN matcher

* Read template files in the template folder
* Split RGB channel, and call `sift.detectAndCompute` with the mask to get keypoints and descriptors.
* Add descriptors to FLANN matcher of the same channel
* Store test image and keypoints to a list

#### Get all matches

* Get matches from matcher by call `knnMatch` with k=2
* Filter good matches if test distance less then training distance times thresh constant

#### Get recognition matrix

* Create matrix of size [len(templates), len(maskContours)] with all zero value
* Iterate through all matches, use `cv2.pointPolygonTest` to test if a match keypoint resides in a contour, and increase the value by 1 with the value in the matrix

#### Wipe out lesser frequencies

* If the total frequency of a template is less than `minMatches` constant, zero out that.
* Zero out the frequency that is not the maximum value of a template

#### Transpose recognition

Transpose recognition matrix and print.

#### Output matches for each template

For each template:
* Get stored template images and keypoints, split the RGB channels of template image.
* Get RGB images by zero out other channel for each RGB template image
* Draw the matches of this template with test image, test keypoints, template image, template keypoint, and respectively colors for each RGB channels.
* If the total amount matches of a mask contour not less than `minMatches` and the index of template is the maxinum in the contour, draw the contour with yellow border (regarded as recognized).
* Save the output image of this template to a file in `output` folder
