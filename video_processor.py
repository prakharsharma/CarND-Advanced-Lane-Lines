"""
Utility class to process video
"""

import cv2
import numpy as np

import config
import utils

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

    cfg = config
    img_processor = None
    last_frame = None
    curr_frame = None
    past_frames = []

    def __init__(self, img_processor, config):
        self.cfg = config
        self.img_processor = img_processor

    def process(self, frame, debug=None):

        if debug is not None:  # override debug
            self.cfg.debug = bool(debug)

        self.curr_frame = frame

        if not self.last_frame:
            self.process_fresh_frame()
        else:
            self.process_based_on_past_frames()

        self.last_frame = self.curr_frame

        return cv2.cvtColor(self.curr_frame.value, cv2.COLOR_BGR2RGB)

    def process_fresh_frame(self):
        self.img_processor.transform(self.curr_frame, self.cfg.debug)
        # print(
        #     "turn_dir: {}, left_lane_curverad: {}, right_lane_curverad: {}, "
        #     "curverad: {}, lane_width_mean: {}, lane_width_stddev: {}, "
        #     "pos_off_center: {}".format(
        #         self.curr_frame.turn_dir, self.curr_frame.left_lane.curverad,
        #         self.curr_frame.right_lane.curverad,
        #         self.curr_frame.curverad, self.curr_frame.lane_width_mean,
        #         self.curr_frame.lane_width_stddev,
        #         self.curr_frame.pos_off_center
        #     )
        # )

    def process_based_on_past_frames(self):

        # do the necessary transforms
        self.img_processor.undistort(self.curr_frame)
        self.img_processor.binary_threshold(self.curr_frame)
        self.img_processor.perspective_transform(self.curr_frame)

        # get all possible x, y for lane lines in curr image using image from last frame
        left_yvals = self.last_frame.left_lane.yvals
        left_fit = self.last_frame.left_lane.fit
        leftx = left_fit[0] * left_yvals ** 2 + \
                left_fit[1] * left_yvals + left_fit[2]

        right_yvals = self.last_frame.right_lane.yvals
        right_fit = self.last_frame.right_lane.fit
        rightx = right_fit[0] * right_yvals ** 2 + \
                 right_fit[1] * right_yvals + right_fit[2]

        # TODO: for debug, plot initial points

        # print("left_yvals: {}, leftx: {}, right_yvals: {}, rightx: {}".format(
        #     len(left_yvals), len(leftx),
        #     len(right_yvals), len(rightx))
        # )

        leftx, left_yvals, rightx, right_yvals = \
            utils.better_lane_points(
                self.curr_frame.value,
                leftx, left_yvals,
                rightx, right_yvals,
                config.window_size
            )

        # TODO: for debug, plot new points

        # fit second order polynomial based on computed x and y vals
        left_fit = np.polyfit(left_yvals, leftx, 2)
        right_fit = np.polyfit(right_yvals, rightx, 2)

        # TODO: for debug, plot lane lines using fitted polynomial

        self.curr_frame.left_lane.yvals = left_yvals
        self.curr_frame.left_lane.fit = left_fit
        self.curr_frame.left_lane.x = leftx

        self.curr_frame.right_lane.yvals = right_yvals
        self.curr_frame.right_lane.fit = right_fit
        self.curr_frame.right_lane.x = rightx

        self.curr_frame.detected = True

        self.curr_frame.lane_curvature()
        self.curr_frame.vehicle_pos_wrt_lane_center()
        self.curr_frame.dist_bw_lanes()

        curverad_change = utils.percent_change(self.curr_frame.curverad,
                                               self.last_frame.curverad)
        width_change = utils.percent_change(self.curr_frame.lane_width_mean,
                                            self.last_frame.lane_width_mean)

        print(
            "turn_dir: {}, left_lane_curverad: {}, right_lane_curverad: {}, "
            "curverad: {}, lane_width_mean: {}, lane_width_stddev: {}, "
            "pos_off_center: {}".format(
                self.curr_frame.turn_dir, self.curr_frame.left_lane.curverad,
                self.curr_frame.right_lane.curverad,
                self.curr_frame.curverad, self.curr_frame.lane_width_mean,
                self.curr_frame.lane_width_stddev,
                self.curr_frame.pos_off_center
            )
        )

        print("width_change: {}, curverad_change: {}".format(
            width_change, curverad_change
        ))

        # TODO: fork flow based on detection confidence

        # warp back
        self.img_processor.warp_back(self.curr_frame)

        # TODO: smooth x and y vals over the past few iterations

        return self.curr_frame

    @classmethod
    def get_video_processor(cls):
        img_proc = ImageProcessor.getImageProcessor()
        proc = cls(img_proc, config)
        return proc
