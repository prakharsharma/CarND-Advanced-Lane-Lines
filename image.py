#-*- coding: utf-8 -*-

"""
abstraction of an image
"""

import cv2

from lane_line import LaneLine


class ImageBadInputError(Exception):
    pass


class Image(object):

    img = None
    left_lane = LaneLine()
    right_lane = LaneLine()
    # lane_width = None
    curvature = None
    turn_dir = None
    pos_off_center = None
    detected = False

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


if __name__ == "__main__":
    pass
