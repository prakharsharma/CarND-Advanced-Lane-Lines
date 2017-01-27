"""
abstraction of a lane line
"""


class LaneLine(object):

    def __init__(self):
        self.fit = None
        self.x = None
        self.yvals = None
        self.curverad = None

    def reset(self):
        # if self.x is not None and len(self.x) > 0:
        #     del self.x[:]
        # if self.yvals is not None and len(self.yvals) > 0:
        #     del self.yvals[:]
        self.x = []
        self.yvals = []
        self.fit = None
        self.curverad = None
