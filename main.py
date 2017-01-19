#-*- coding: utf-8 -*-

"""
TODO
The goals / steps of this project are the following:

- Compute the camera calibration matrix and distortion coefficients given a
set of chessboard images.
- Apply the distortion correction to the raw image.
- Use color transforms, gradients, etc., to create a thresholded binary image.
- Apply a perspective transform to rectify binary image ("birds-eye view").
- Detect lane pixels and fit to find lane boundary.
- Determine curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane
curvature and vehicle position.
"""

import argparse
import cv2
import glob

from image import Image
from image_processor import ImageProcessor


def pipeline(img, opts):
    pass


def transform(img, processor):
    """given an image, return the transformed image"""
    pass


def main():
    imageProcessor = ImageProcessor.getImageProcessor()
    for fname in glob.glob('camera_cal/calibration*.jpg'):
        img = Image(fname=fname)
        imageProcessor.transform(img, debug=True)
        pass

    pass


if __name__ == "__main__":
    # args = None  # TODO:
    main()
