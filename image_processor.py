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

import matplotlib.pyplot as plt

from camera_calibration import CameraCalibrator


class ImageProcessorError(Exception):
    pass


class ImageProcessor(object):

    def __init__(self, cameraCalibrator, config):
        self.cameraCalibrator = cameraCalibrator
        self.sobelKsize = 5
        self.cfg = config
        if self.cfg.debug:
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

    def transform(self, img):
        if self.cfg.debug:
            # cv2.imwrite("{}/{}.jpg".format(prefix, img.name), img.value)
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

        # correct for distortion
        self.undistort(img)

        # create a thresholded binary image.
        gray = cv2.cvtColor(img.value, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.cfg.sobelKsize)
        # Absolute x derivative to accentuate lines away from horizontal
        absSobelx = np.absolute(sobelx)
        scaledSobel = np.uint8(255 * absSobelx / np.max(absSobelx))
        # Threshold x gradient
        sxBinary = np.zeros_like(scaledSobel)
        sxBinary[(scaledSobel >= self.cfg.sxThresh[0]) &
                 (scaledSobel <= self.cfg.sxThresh[1])] = 1
        img.addStage('sobelxBinary', sxBinary)
        if self.cfg.debug:
            # cv2.imwrite("{}/{}.jpg".format(prefix, img.name), img.value)
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

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

        # Combine the two binary thresholds
        combinedBinary = np.zeros_like(sxBinary)
        combinedBinary[(sBinary == 1) | (sxBinary == 1)] = 1
        img.addStage('combinedBinary', combinedBinary)
        if self.cfg.debug:
            # cv2.imwrite("{}/{}.jpg".format(prefix, img.name), img.value)
            plt.imsave("{}/{}.jpg".format(self.cfg.debugPrefix, img.name),
                       img.value, cmap='gray')

        # Apply a perspective transform to rectify binary image.
        self.perspectiveTransform(img)

    def perspectiveTransform(self, img):
        # TODO: find src and dst points
        src = None
        dst = None
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

    @classmethod
    def getImageProcessor(cls):
        import config
        calibrator = CameraCalibrator.getCameraCalibrator()
        proc = cls(calibrator, config)
        return proc


if __name__ == "__main__":
    pass
