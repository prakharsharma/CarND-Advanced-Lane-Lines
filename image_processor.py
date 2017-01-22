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
        img.addStage('undistorted', dst)
        if self.cfg.debug:
            # cv2.imwrite("{}/{}.jpg".format(prefix, img.name), img.value)
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

    def gradientThreshold(self, img, orient='x'):
        gray = cv2.cvtColor(img.imageForStage('undistorted'),
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
        img.addStage('sobel{}Binary'.format(orient), sxBinary)
        if self.cfg.debug:
            # cv2.imwrite("{}/{}.jpg".format(prefix, img.name), img.value)
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

    def sChannelThreshold(self, img):
        hls = cv2.cvtColor(img.imageForStage('undistorted'), cv2.COLOR_BGR2HLS)
        sChannel = hls[:, :, 2]
        # Threshold color channel
        sBinary = np.zeros_like(sChannel)
        sBinary[(sChannel >= self.cfg.sThresh[0]) &
                (sChannel <= self.cfg.sThresh[1])] = 1
        img.addStage('sChannelBinary', sBinary)
        if self.cfg.debug:
            # cv2.imwrite("{}/{}.jpg".format(prefix, img.name), img.value)
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

    def binaryThreshold(self, img):
        self.gradientThreshold(img, 'x')
        self.sChannelThreshold(img)

        sxBinary = img.imageForStage('sobelxBinary')
        sBinary = img.imageForStage('sChannelBinary')

        # Combine the two binary thresholds
        combinedBinary = np.zeros_like(sxBinary)
        combinedBinary[(sBinary == 1) | (sxBinary == 1)] = 1
        img.addStage('combinedBinary', combinedBinary)
        if self.cfg.debug:
            # cv2.imwrite("{}/{}.jpg".format(prefix, img.name), img.value)
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

    def perspectiveTransform(self, img):
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
        img.perspectiveTransformMat = cv2.getPerspectiveTransform(src, dst)
        img.invPerspectiveTransformMat = cv2.getPerspectiveTransform(dst, src)
        h, w = img.imageForStage('original').value.shape[:2]
        warped = cv2.warpPerspective(
            img.value,
            img.perspectiveTransformMat,
            (w, h),
            flags=cv2.INTER_LINEAR
        )
        img.addStage('perspectiveTransform', warped)
        if self.cfg.debug:
            # cv2.imwrite("{}/{}.jpg".format(prefix, img.name), img.value)
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

    def detectLaneLines(self, img):
        # detect lane pixels

        # fit a polynomial around lane pixels to get an approximation for lane
        # lines
        pass

    def transform(self, img):
        if self.cfg.debug:
            # cv2.imwrite("{}/{}.jpg".format(prefix, img.name), img.value)
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

        # correct for distortion
        self.undistort(img)

        # create a thresholded binary image.
        self.binaryThreshold(img)

        # Apply a perspective transform to rectify binary image.
        self.perspectiveTransform(img)

        # Detect lane pixels and fit to find lane boundary.
        self.detectLaneLines(img)

        # Determine curvature of the lane and vehicle position with respect to
        # center.

        # Warp the detected lane boundaries back onto the original image.

        # Output visual display of the lane boundaries and numerical estimation
        # of lane curvature and vehicle position.

    def regionMaskForLaneLines(self, img):
        pass

    @classmethod
    def getImageProcessor(cls):
        import config
        calibrator = CameraCalibrator.getCameraCalibrator()
        proc = cls(calibrator, config)
        return proc


if __name__ == "__main__":
    pass
