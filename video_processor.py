"""
Utility class to process video
"""

import config
import cv2

import numpy as np

from image_processor import ImageProcessor


# Define a class to receive the characteristics of each line detection
class Line(object):

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None


class VideoProcessor(object):

    def __init__(self, img_processor, config):
        self.cfg = config
        self.img_processor = img_processor
        self.last_proc_img = None
        self.left_lane = None
        self.right_lane = None

    def process(self, img, debug=None):
        if debug is not None:  # override debug
            self.cfg.debug = bool(debug)

        if not self.last_proc_img:
            self.img_processor.transform(img, self.cfg.debug)
            result = img.value
        else:
            # TODO: do the real thing
            result = img.value

        self.last_proc_img = img

        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    @classmethod
    def get_video_processor(cls):
        img_proc = ImageProcessor.getImageProcessor()
        proc = cls(img_proc, config)
        return proc
