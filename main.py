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
import config
import time

import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip

from image import Image
from image_processor import ImageProcessor


def process_images(processor, debug=False, write_out=True):
    for fname in glob.glob('./test_images/*.jpg'):
        parts = list(filter(None, fname.split('/')))
        name = parts[-1]
        img = Image(fname=fname)
        processor.transform(img, debug)
        result = cv2.cvtColor(img.value, cv2.COLOR_BGR2RGB)
        if write_out or config.write_out:
            plt.imsave('{}/{}'.format(config.output_path, name), result)


def process_video_frame(processor):

    def _process_video_frame(image):
        # TODO: replace with something more efficient
        img = Image(img=image)
        processor.transform(img, debug=False)
        return cv2.cvtColor(img.value, cv2.COLOR_BGR2RGB)

    return _process_video_frame


def process_video(fspath, processor, write_out=True):
    parts = list(filter(None, fspath.split('/')))
    name = parts[-1]
    clip = VideoFileClip(fspath, audio=False)
    modified_clip = clip.fl_image(process_video_frame(processor))
    # if write_out or config.write_out:
    #     modified_clip.write_videofile(
    #         '{}/{}'.format(config.output_path, name),
    #         audio=False
    #     )


def main():
    processor = ImageProcessor.getImageProcessor()
    process_video('./project_video.mp4', processor)


if __name__ == "__main__":
    # args = None  # TODO:
    main()
