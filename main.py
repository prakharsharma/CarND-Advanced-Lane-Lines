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

import numpy as np
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip

from image import Image
from image_processor import ImageProcessor


def process_images(img_processor, debug=False, write_out=True):
    for fname in glob.glob('./test_images/*.jpg'):
        parts = list(filter(None, fname.split('/')))
        name = parts[-1]
        img = Image(fname=fname)
        img_processor.transform(img, debug)
        result = cv2.cvtColor(img.value, cv2.COLOR_BGR2RGB)
        if write_out or config.write_out:
            plt.imsave('{}/{}'.format(config.output_path, name), result)


def process_video_frame(vid_processor):

    def _process_video_frame(image):
        # TODO: replace with something more efficient
        img = Image(img=image)
        result = vid_processor.process(img, debug=False)
        return result

    return _process_video_frame


def test_process_image(image):
    # nonlocal frames, base_path
    frame_count = 0
    base_path = '/tmp/vid'
    def _worker(image):
        cp = np.copy(image)
        nonlocal frame_count, base_path
        frame_count += 1
        fname = '/tmp/vid_{}.jpg'.format(frame_count)
        plt.imsave(cp, image)
        return cp

    return _worker


num_frames = 0
def haha(image):
    global num_frames
    num_frames += 1
    return np.copy(image)


def process_video(fspath, processor, write_out=True):
    parts = list(filter(None, fspath.split('/')))
    name = parts[-1]
    clip = VideoFileClip(fspath, audio=False)
    modified_clip = clip.fl_image(
        # process_video_frame(processor)
        # test_process_image
        haha
    )
    global num_frames
    print("num of frames: {}".format(num_frames))
    # if write_out or config.write_out:
    #     modified_clip.write_videofile(
    #         '{}/{}'.format(config.output_path, name),
    #         audio=False
    #     )


def main():
    img_processor = ImageProcessor.getImageProcessor()
    process_video('./project_video.mp4', img_processor)


if __name__ == "__main__":
    # args = None  # TODO:
    main()
