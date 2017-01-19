#-*- coding: utf-8 -*-

"""
abstraction of an image
"""

import cv2


class ImageBadInputError(Exception):
    pass


class Image(object):

    def __init__(self, img=None, fname=None):
        if img is None and fname is None:
            raise ImageBadInputError
        if img is None and fname:
            img = cv2.imread(fname)
        self._stages = ['original']
        self._stageMap = {
            'original': img
        }
        self.name = 'original'
        self.value = img
        self.perspectiveTransformMat = None
        self.invPerspectiveTransformMat = None

    def addStage(self, name, value, isNewCurr=True):
        self._stages.append(name)
        self._stageMap[name] = value
        if isNewCurr:
            self.name = name
            self.value = value

    def imageForStage(self, name):
        return self._stageMap[name]


if __name__ == "__main__":
    pass
