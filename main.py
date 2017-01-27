#-*- coding: utf-8 -*-

"""
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
import utils

import numpy as np
import matplotlib.pyplot as plt

from moviepy.editor import VideoFileClip

from image import Image
from image_processor import ImageProcessor
from video_processor import VideoProcessor


def process_images(img_processor, debug=False, write_out=True):
    for fname in glob.glob('./test_images/*.jpg'):
        parts = list(filter(None, fname.split('/')))
        name = parts[-1]
        img = Image(fname=fname)
        img_processor.transform(img, debug)
        result = cv2.cvtColor(img.value, cv2.COLOR_BGR2RGB)
        if write_out or config.write_out:
            plt.imsave('{}/{}'.format(
                # config.output_path,
                './output_images2',
                name
            ), result)


def process_video_frame(vid_processor):

    def _process_video_frame(image):
        # TODO: replace with something more efficient
        frame = Image(img=image)
        result = vid_processor.process(frame, debug=False)
        return result

    return _process_video_frame


def process_video(fspath, processor, write_out=True):
    ts = int(time.time()*1000)
    # parts = list(filter(None, fspath.split('/')))
    # name = parts[-1]
    name = 'project_video-{}.mp4'.format(ts)
    out_fspath = './output_videos/{}'.format(name)
    clip = VideoFileClip(fspath, audio=False)
    out_clip = clip.fl_image(process_video_frame(processor))
    if write_out or config.write_out:
        out_clip.write_videofile(
            # '{}/{}'.format(config.output_path, name),
            out_fspath,
            audio=False
        )
    utils.dump_video_stats(processor.stats,
                           './output_videos/stats-{}.csv'.format(ts))


def main():
    img_processor = ImageProcessor.getImageProcessor()
    process_images(img_processor)
    vid_processor = VideoProcessor.get_video_processor(img_processor)
    process_video('./project_video.mp4', vid_processor)


if __name__ == "__main__":
    # TODO: add command line options for handling image and video mode
    main()
