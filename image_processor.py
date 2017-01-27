#-*- coding: utf-8 -*-

"""
abstraction of image processor

hyper parameters
- Sobel kernel size
- thresholds

Threshold pointers
- magnitude of gradient: sobel_kernel=9, mag_thresh=(30, 100).
- direction of gradient: sobel_kernel=15, thresh=(0.7, 1.3).
- Gray: (180, 255)
- RGB: (180, 255)
- R channel: (200, 255)
- S channel: (90, 255)
- H channel: (15, 100)

Ideas

- inverse of H channel && S channel
- combine R channel and S channel
    - R_binary || S_binary ?
- combine S channel and X gradient
    - s_thresh=(170, 255), sx_thresh=(20, 100)
"""

import cv2
import numpy as np
import os
import os.path

import matplotlib.pyplot as plt

from camera_calibration import CameraCalibrator


def xyvals_from_list(xyval_list):
    yvals = []
    x = []
    for p in xyval_list:
        x.append(p[0])
        yvals.append(p[1])
    yvals = np.array(yvals, dtype=np.uint32)
    x = np.array(x, dtype=np.uint32)
    return x, yvals


class ImageProcessorError(Exception):
    pass


class ImageProcessor(object):

    def __init__(self, cameraCalibrator, config):
        self.cameraCalibrator = cameraCalibrator
        self.sobelKsize = 5
        self.cfg = config
        if self.cfg.debug and not os.path.exists(self.cfg.debugPrefix):
            os.makedirs(self.cfg.debugPrefix)

    def undistort(self, img):
        dst = cv2.undistort(
            img.value,
            self.cameraCalibrator.cameraMatrix,
            self.cameraCalibrator.distortionCoefficient,
            None,
            self.cameraCalibrator.cameraMatrix
        )
        img.add_stage('undistorted', dst)
        if self.cfg.debug:
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

    def gradient_threshold(self, img, orient='x'):
        gray = cv2.cvtColor(img.image_for_stage('undistorted'),
                            cv2.COLOR_BGR2GRAY)
        sobel = cv2.Sobel(
            gray,
            cv2.CV_64F,
            int(orient == 'x'),
            int(orient == 'y'),
            ksize=self.cfg.sobelKsize
        )
        # Absolute x derivative to accentuate lines away from horizontal
        absSobel = np.absolute(sobel)
        scaledSobel = np.uint8(255 * absSobel / np.max(absSobel))
        # Threshold x gradient
        sxBinary = np.zeros_like(scaledSobel)
        sxBinary[(scaledSobel >= self.cfg.sobelThresh[orient][0]) &
                 (scaledSobel <= self.cfg.sobelThresh[orient][1])] = 1
        img.add_stage('sobel{}Binary'.format(orient), sxBinary)
        if self.cfg.debug:
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

    def sChannel_threshold(self, img):
        hls = cv2.cvtColor(img.image_for_stage('undistorted'),
                           cv2.COLOR_BGR2HLS)
        sChannel = hls[:, :, 2]
        # Threshold color channel
        sBinary = np.zeros_like(sChannel)
        sBinary[(sChannel >= self.cfg.sThresh[0]) &
                (sChannel <= self.cfg.sThresh[1])] = 1
        img.add_stage('sChannelBinary', sBinary)
        if self.cfg.debug:
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

    def binary_threshold(self, img):
        self.gradient_threshold(img, 'x')
        self.sChannel_threshold(img)

        sxBinary = img.image_for_stage('sobelxBinary')
        sBinary = img.image_for_stage('sChannelBinary')

        # Combine the two binary thresholds
        combinedBinary = np.zeros_like(sxBinary)
        combinedBinary[(sBinary == 1) | (sxBinary == 1)] = 1
        img.add_stage('combinedBinary', combinedBinary)
        if self.cfg.debug:
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

    def perspective_transform(self, img):
        line_len = lambda p1, p2: np.sqrt(
            (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

        src = np.float32([
            # top left
            [586., 458.],

            # top right
            [697., 458.],

            # bottom right
            [1089., 718.],

            # bottom left
            [207., 718.]
        ])

        # l1 = line_len(src[0], src[3])
        l2 = line_len(src[1], src[2])

        rect_width = src[2][0] - src[3][0]
        top_right = [src[2][0], src[2][1] - l2]

        # print("l1: {}, l2: {}, rect_width: {}".format(l1, l2, rect_width))

        dst = np.float32([
            # top left
            [top_right[0] - rect_width, top_right[1]],

            # top right
            top_right,

            # bottom right
            src[2],

            # bottom left
            src[3]
        ])
        img.perspective_transform_mat = cv2.getPerspectiveTransform(src, dst)
        img.inv_perspective_transform_mat = cv2.getPerspectiveTransform(dst,
                                                                        src)
        h, w = img.image_for_stage('original').shape[:2]
        warped = cv2.warpPerspective(
            img.value,
            img.perspective_transform_mat,
            (w, h),
            flags=cv2.INTER_LINEAR
        )
        img.add_stage('perspectiveTransform', warped)
        if self.cfg.debug:
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

    def detect_lane_pixels(self, img):
        # detect lane pixels
        warped = img.image_for_stage('perspectiveTransform')
        h, w = warped.shape

        # detect starting x for left and right lane
        histogram = np.sum(warped[h/2:, :], axis=0)
        p1 = np.argmax(histogram[:w/2])
        p2 = np.argmax(histogram[w/2:])
        if self.cfg.debug:
            print("left peak: x: {}, y: {}".format(p1, histogram[p1]))
            print("right peak: x: {}, y: {}".format(640 + p2,
                                                    histogram[640 + p2]))

        left_lane = []
        right_lane = []
        window_size = 140
        step_size = 10
        too_far = 20

        left_lane.append([p1, h])
        right_lane.append([w / 2 + p2, h])

        left_lane_center_x = p1
        right_lane_center_x = w / 2 + p2
        curr_top = h

        # sliding window algorithm to detect lane line pixels
        while curr_top >= step_size:
            # define window for left and right lane
            y_range = (curr_top - step_size, curr_top)
            left_x_range = (left_lane_center_x - window_size / 2,
                            left_lane_center_x + window_size / 2)
            right_x_range = (right_lane_center_x - window_size / 2,
                             right_lane_center_x + window_size / 2)

            # update center of left lane
            if left_x_range[0] >= 0:
                left_histogram = np.sum(warped[y_range[0]:y_range[1],
                                        left_x_range[0]:left_x_range[1]],
                                        axis=0)
                left_peak = np.argmax(left_histogram)
                if left_peak:
                    last_y = left_lane[-1][1]
                    if last_y - y_range[0] <= too_far * step_size:
                        left_lane.append([left_x_range[0] + left_peak,
                                          y_range[0] + step_size / 2])
                        left_lane_center_x = left_x_range[0] + left_peak
                    else:
                        # print("too far from the last left peak, ignore")
                        pass

            # update center of right lane
            if right_x_range[0] >= 0:
                right_histogram = np.sum(warped[y_range[0]:y_range[1],
                                         right_x_range[0]:right_x_range[1]],
                                         axis=0)
                right_peak = np.argmax(right_histogram)
                if right_peak:
                    # check if current peak isn't too far from the last peak,
                    # to keep stray pixels out
                    last_y = right_lane[-1][1]
                    if last_y - y_range[0] <= too_far * step_size:
                        right_lane.append([right_x_range[0] + right_peak,
                                           y_range[0] + step_size / 2])
                        right_lane_center_x = right_x_range[0] + right_peak
                    else:
                        # print("too far from the last right peak, ignore")
                        pass

            # print("y_range: {}, left_x_range: {}, right_x_range: {},"
            #       "left_peak: ({}, {}), right_peak: ({}, {})".format(
            #         y_range, left_x_range, right_x_range,
            #         left_peak, left_x_range[0] + left_peak,
            #         right_peak, right_x_range[0] + right_peak
            # ))

            # slide the window up for next iteration
            curr_top -= step_size

        # DIAGNOSTIC_TODO: write an image with detected lane line pixels
        # plotted over warped binary image

        return left_lane, right_lane

    def detect_lane_lines(self, img):
        # detect lane pixels

        left_lane, right_lane = self.detect_lane_pixels(img)

        # fit polynomial around lane pixels to get approximation for lane lines
        leftx, left_yvals = xyvals_from_list(left_lane)
        rightx, right_yvals = xyvals_from_list(right_lane)

        # Fit a second order polynomial
        left_fit = np.polyfit(left_yvals, leftx, 2)

        right_fit = np.polyfit(right_yvals, rightx, 2)

        img.left_lane.yvals = left_yvals
        img.left_lane.fit = left_fit
        img.left_lane.x = leftx

        img.right_lane.yvals = right_yvals
        img.right_lane.fit = right_fit
        img.right_lane.x = rightx

        img.detected = True
        # DIAGNOSTIC_TODO: write debug step to write an image with drawn
        # fitted lane lines

    def curvature_and_vehicle_pos(self, img):
        """
        determines radius of curvature of left and right lanes and position
        of the vehicle wrt lane center
        """

        img.lane_curvature()
        img.vehicle_pos_wrt_lane_center()
        img.dist_bw_lanes()

    def warp_back(self, img, left_fit=None, left_yvals=None, right_fit=None,
                  right_yvals=None, curverad=None, pos_off_center=None):
        """
        draws detected lane lines on undistorted image and does a reverse
        perspective transformation
        """

        warped = img.image_for_stage('perspectiveTransform')
        h, w = warped.shape

        if left_fit is None:
            left_fit = img.left_lane.fit
        if left_yvals is None:
            left_yvals = img.left_lane.yvals
        left_fitx = left_fit[0] * left_yvals ** 2 + \
                    left_fit[1] * left_yvals + left_fit[2]

        if right_fit is None:
            right_fit = img.right_lane.fit
        if right_yvals is None:
            right_yvals = img.right_lane.yvals
        right_fitx = right_fit[0] * right_yvals ** 2 + \
                     right_fit[1] * right_yvals + right_fit[2]

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, left_yvals]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, right_yvals])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse
        # perspective matrix (Minv)
        newwarp = cv2.warpPerspective(
            color_warp,
            img.inv_perspective_transform_mat,
            (w, h)
        )
        # Combine the result with the original image
        undist = img.image_for_stage('undistorted')
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        if curverad is None:
            curverad = img.curverad
        if pos_off_center is None:
            pos_off_center = img.pos_off_center

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            result,
            'Radius of curvature = {:.2f}m'.format(curverad),
            (50, 50),
            font,
            1.2,
            (255, 255, 255),
            2
        )
        cv2.putText(
            result,
            'Vehicle is {:.2f}m {} of center'.format(
                abs(pos_off_center),
                "left" if pos_off_center < 0 else "right"
            ),
            (50, 100),
            font,
            1.2,
            (255, 255, 255),
            2
        )

        img.add_stage('warpBack', result)

        if self.cfg.debug:
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value)

    def transform(self, img, debug=None):
        if debug is not None:  # override debug
            self.cfg.debug = bool(debug)

        if self.cfg.debug:
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

        # correct for distortion
        self.undistort(img)

        # create a thresholded binary image.
        self.binary_threshold(img)

        # Apply a perspective transform to rectify binary image.
        self.perspective_transform(img)

        # Detect lane pixels and fit to find lane boundary.
        self.detect_lane_lines(img)

        # Determine curvature of the lane and vehicle position with respect to
        # center.
        self.curvature_and_vehicle_pos(img)

        # Warp the detected lane boundaries back onto the original image.
        self.warp_back(img)

        # print(
        #     "turn_dir: {}, left_lane_curverad: {}, right_lane_curverad: {}, "
        #     "curverad: {}, lane_width_mean: {}, lane_width_stddev: {}, "
        #     "pos_off_center: {}".format(
        #         img.turn_dir, img.left_lane.curverad, img.right_lane.curverad,
        #         img.curverad, img.lane_width_mean, img.lane_width_stddev,
        #         img.pos_off_center
        #     )
        # )

        return img

    @classmethod
    def getImageProcessor(cls):
        import config
        calibrator = CameraCalibrator.getCameraCalibrator()
        proc = cls(calibrator, config)
        return proc


if __name__ == "__main__":
    pass
