#-*- coding: utf-8 -*-

"""
abstraction of an image
"""

import cv2
import numpy as np

import config

from lane_line import LaneLine


class ImageBadInputError(Exception):
    pass


class ImageError(Exception):
    pass


class Image(object):

    img = None

    left_lane = LaneLine()
    right_lane = LaneLine()

    lane_width_mean = None
    lane_width_stddev = None

    curverad = None

    turn_dir = None

    pos_off_center = None

    detected = False
    detection_confidence = None

    def __init__(self, img=None, fname=None):
        if img is None and fname is None:
            raise ImageBadInputError
        if img is None and fname:
            img = cv2.imread(fname)
        self._stages = ['original']
        self._stage_map = {
            'original': img
        }
        self.name = 'original'
        self.value = img
        self.perspective_transform_mat = None
        self.inv_perspective_transform_mat = None
        # self.lane = {}

    def add_stage(self, name, value, isNewCurr=True):
        self._stages.append(name)
        self._stage_map[name] = value
        if isNewCurr:
            self.name = name
            self.value = value

    def image_for_stage(self, name):
        return self._stage_map[name]

    def dist_bw_lanes(self, step_size=16):
        """computes mean and stddev of lane width across the image"""

        if not self.detected:
            raise ImageError("lane detection prerequisite for computing lane "
                             "width")

        left_yvals = self.left_lane.yvals
        left_fit = self.left_lane.fit

        right_yvals = self.right_lane.yvals
        right_fit = self.right_lane.fit

        h, w = self.image_for_stage('original').shape[:2]
        m = step_size
        i = m

        yvals = []
        dist = []
        while i >= 0:
            y = int(i * h / m)
            if y < left_yvals[-1]:
                # print("left lane ended")
                break
            if y < right_yvals[-1]:
                # print("right lane ended")
                break
            x_left = left_fit[0] * y ** 2 + left_fit[1] * y + left_fit[2]
            x_right = right_fit[0] * y ** 2 + right_fit[1] * y + right_fit[2]
            dist_pix = x_right - x_left
            dist_m = dist_pix * config.xm_per_pix
            i -= 1
            dist.append(dist_m)
            yvals.append(y)

        self.lane_width_mean = np.mean(dist)
        self.lane_width_stddev = np.std(dist)

        # return yvals, dist

    def lane_curvature(self):

        if not self.detected:
            raise ImageError("lane detection prerequisite for determining "
                             "radius of curvature of lane lines")

        leftx = self.left_lane.x
        left_yvals = self.left_lane.yvals

        rightx = self.right_lane.x
        right_yvals = self.right_lane.yvals

        # find curvature
        left_y_eval = np.max(left_yvals)
        right_y_eval = np.max(right_yvals)

        left_yvals2 = left_yvals * config.ym_per_pix
        right_yvals2 = right_yvals * config.ym_per_pix

        left_fit_cr = np.polyfit(left_yvals2, leftx * config.xm_per_pix, 2)
        right_fit_cr = np.polyfit(right_yvals2, rightx * config.xm_per_pix, 2)

        left_curverad = ((1 + (
        2 * left_fit_cr[0] * left_y_eval + left_fit_cr[1]) ** 2) ** 1.5) \
                        / np.absolute(2 * left_fit_cr[0])

        right_curverad = ((1 + (
        2 * right_fit_cr[0] * right_y_eval + right_fit_cr[1]) ** 2) ** 1.5) \
                         / np.absolute(2 * right_fit_cr[0])

        self.turn_dir = 'left' if left_curverad < right_curverad else 'right'
        self.curverad = min(left_curverad, right_curverad)

        self.left_lane.curverad = left_curverad
        self.right_lane.curverad = right_curverad

    def vehicle_pos_wrt_lane_center(self):
        """
        1. Find starting (closest to the vehicle) points of the lane (x_left, h) and (x_right, h)
            1. Intersection of the left lane line with bottom of the image gives us starting point of left lane, i.e. (x_left, h)
            2. Intersection of the right lane line with bottom of the image gives us starting point of left lane, i.e. (x_right, h)
        2. Find center of the vehicle, assuming vehicle to be located at the center of the image.
        3. Find position of vehicle wrt lane center
        """

        if not self.detected:
            raise ImageError("lane detection prerequisite for computing "
                             "vehicle pos wrt lane center")

        try:
            warped = self.image_for_stage('perspectiveTransform')
        except KeyError:
            raise ImageError("warped image required for computing vehicle pos "
                             "wrt lane center")

        h, w = warped.shape

        left_fit = self.left_lane.fit
        right_fit = self.right_lane.fit

        x_left = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
        x_right = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]

        lane_width = x_right - x_left
        lane_center = x_left + lane_width / 2.0

        x_vehicle = w / 2.0

        x_vehicle_off_center = x_vehicle - lane_center
        x_vehicle_off_center_m = x_vehicle_off_center * config.xm_per_pix

        if config.debug:
            print("starting pos of lane, left: {}, right: {}".format(
                x_left,x_right))
            print("lane center: {}, vehicle pos: {}".format(
                lane_center, x_vehicle))
            print("vehicle is {:.4f}m {} of center".format(
                abs(x_vehicle_off_center_m),
                "left" if x_vehicle_off_center < 0 else "right"))

        self.pos_off_center = x_vehicle_off_center_m

    def determine_detection_confidence(self):
        # TODO
        pass


if __name__ == "__main__":
    pass
