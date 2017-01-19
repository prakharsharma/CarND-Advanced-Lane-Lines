#-*- coding: utf-8 -*-

"""
"""

import glob
import numpy as np
import cv2
import pickle


class CameraCalibrationValueError(Exception):
    pass


class CameraCalibrationError(Exception):
    pass


class CameraCalibrator(object):

    cameraMatrix = None
    distortionCoefficient = None

    def __init__(self):
        pass

    def load(self, fspath='camera_calibration.p'):
        d = pickle.load(open(fspath, 'rb'))

        cameraMatrix = d.get('cameraMatrix', None)
        if cameraMatrix is None:
            raise CameraCalibrationValueError
        self.cameraMatrix = cameraMatrix

        distortionCoefficient = d.get('distortionCoefficient', None)
        if distortionCoefficient is None:
            raise CameraCalibrationValueError

        self.distortionCoefficient = distortionCoefficient

    def calibrate(self, images, nx, ny, store=True,
                  fspath='camera_calibration.p'):

        objpoints, imgpoints = [], []
        objp = np.zeros((ny*nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        img_shape = None

        for _, fname in enumerate(images):
            img = cv2.imread(fname)
            if not img_shape:
                img_shape = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
                # print('found {} corners for file: {}'.format(len(corners),
                #                                              fname))
            else:
                # print('corners not found for file: {}'.format(fname))
                pass

        # print('#objpoints: {}, #imgpoints: {}'.format(len(objpoints),
        #                                               len(imgpoints)))
        if objpoints and imgpoints and len(objpoints) == len(imgpoints):
            img_h, img_w, _ = img_shape
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, (img_w, img_h), None, None)
            self.cameraMatrix = mtx
            self.distortionCoefficient = dist

            if store:
                pickle.dump({
                    'cameraMatrix': self.cameraMatrix,
                    'distortionCoefficient': self.distortionCoefficient
                }, open(fspath, 'wb'))

            return self.cameraMatrix, self.distortionCoefficient
        else:
            raise CameraCalibrationError

    @classmethod
    def getCameraCalibrator(cls):
        images = glob.glob('camera_cal/calibration*.jpg')
        calibrator = cls()
        calibrator.calibrate(images, nx=9, ny=6)
        return calibrator


if __name__ == "__main__":
    images = glob.glob('camera_cal/calibration*.jpg')
    calibrator = CameraCalibrator()
    mtx, dst = calibrator.calibrate(images, nx=9, ny=6)
    print(mtx)
    print(dst)
