"""
Utility class to process video
"""

import cv2
import numpy as np

import config
import utils

from image_processor import ImageProcessor


class NoLanePointsFoundError(Exception):
    pass


class InconsistentLaneLinesError(Exception):
    pass


class VerySmallLaneError(Exception):
    pass


class VideoProcessor(object):

    def __init__(self, img_processor, config):
        self.cfg = config
        self.img_processor = img_processor
        self.curr_frame = None
        self.past_frames = []
        self.bad_frame_streak = 0
        self.frame_count = 0
        self.stats = []
        self.curr_frame_is_bad = False

    def process(self, frame, debug=None):
        if debug is not None:  # override debug
            self.cfg.debug = bool(debug)

        self.frame_count += 1

        self.curr_frame = frame
        self.curr_frame_is_bad = False

        if len(self.past_frames) < self.cfg.min_past_frames:
            self.process_fresh_frame()
        else:
            try:
                self.process_based_on_past_frames()
            except NoLanePointsFoundError:
                self.curr_frame_is_bad = True
                print("Frame#{} no lane points found using past image".format(
                    self.frame_count
                ))
            except InconsistentLaneLinesError:
                self.curr_frame_is_bad = True
            except VerySmallLaneError:
                self.curr_frame_is_bad = True

        if self.curr_frame_is_bad:
            self.handle_bad_frame()
        else:
            self.bad_frame_streak = 0

        self.collect_stats()

        return cv2.cvtColor(self.curr_frame.value, cv2.COLOR_BGR2RGB)

    def process_fresh_frame(self, recovery=False):

        if not recovery:
            self.img_processor.transform(self.curr_frame, self.cfg.debug)
        else:
            self.img_processor.detect_lane_lines(self.curr_frame)
            self.img_processor.curvature_and_vehicle_pos(self.curr_frame)

        self.past_frames.append(self.curr_frame)
        self.finalize_lane_lines()

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

        last_frame = self.past_frames[-1]

        # get all possible x, y for lane lines in curr image using image from last frame
        left_yvals = last_frame.left_lane.yvals
        left_fit = last_frame.left_lane.fit
        leftx = left_fit[0] * left_yvals ** 2 + \
                left_fit[1] * left_yvals + left_fit[2]

        right_yvals = last_frame.right_lane.yvals
        right_fit = last_frame.right_lane.fit
        rightx = right_fit[0] * right_yvals ** 2 + \
                 right_fit[1] * right_yvals + right_fit[2]

        # DIAGNOSTIC_TODO: plot initial points

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

        if len(left_yvals) < self.cfg.min_lane_len or \
                len(right_yvals) < self.cfg.min_lane_len:
            print("Frame#{} very small lane".format(self.frame_count))
            raise VerySmallLaneError

        per_chng_x = int(round(
            100.0 * min(len(leftx), len(rightx))/max(len(leftx), len(rightx))
        ))
        per_chng_y = int(round(
            100.0 * min(len(left_yvals), len(right_yvals)) /
            max(len(left_yvals), len(right_yvals))
        ))
        if per_chng_x < self.cfg.tolerable_change_in_lanes:
            print("Frame#{} big difference in X vals of left and right "
                  "lane".format(self.frame_count))
            raise InconsistentLaneLinesError
        if per_chng_y < self.cfg.tolerable_change_in_lanes:
            print("Frame#{} big difference in Y vals of left and right "
                  "lane".format(self.frame_count))
            raise InconsistentLaneLinesError

        # DIAGNOSTIC_TODO: plot new points

        # fit second order polynomial based on computed x and y vals
        try:
            left_fit = np.polyfit(left_yvals, leftx, 2)
            right_fit = np.polyfit(right_yvals, rightx, 2)
        except:
            raise NoLanePointsFoundError

        # DIAGNOSTIC_TODO: plot lane lines using fitted polynomial

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

        ## IDEAS: can use percent change in radius of curvature and lane width
        ## to determine detection confidence
        # curverad_change = utils.percent_change(self.curr_frame.curverad,
        #                                        last_frame.curverad)
        # width_change = utils.percent_change(self.curr_frame.lane_width_mean,
        #                                     last_frame.lane_width_mean)

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

        self.past_frames.append(self.curr_frame)
        self.finalize_lane_lines()

        return self.curr_frame

    def finalize_lane_lines(self):
        # keep at the most {max_past_frames} most recent frames
        if len(self.past_frames) >= self.cfg.max_past_frames:
            self.past_frames.pop(0)
        self.smoothen_fit()

    def smoothen_fit(self):
        """smooth x and y vals over the past few iterations"""

        # build list of x and y vals collected from past frames
        leftx_list = []
        left_yvals_list = []

        rightx_list = []
        right_yvals_list = []

        for f in self.past_frames:
            leftx_list.append(f.left_lane.x)
            left_yvals_list.append(f.left_lane.yvals)

            rightx_list.append(f.right_lane.x)
            right_yvals_list.append(f.right_lane.yvals)

        # smooth by taking average
        leftx = utils.mean(leftx_list)
        left_yvals = utils.mean(left_yvals_list)

        rightx = utils.mean(rightx_list)
        right_yvals = utils.mean(right_yvals_list)

        # fit a polyline
        left_fit = np.polyfit(left_yvals, leftx, 2)
        right_fit = np.polyfit(right_yvals, rightx, 2)

        # warp back
        if self.curr_frame_is_bad:
            curverad = self.past_frames[-1].curverad
            pos_off_center = self.past_frames[-1].pos_off_center
        else:
            curverad = self.curr_frame.curverad
            pos_off_center = self.curr_frame.pos_off_center

        self.img_processor.warp_back(self.curr_frame, left_fit, left_yvals,
                                     right_fit, right_yvals, curverad,
                                     pos_off_center)

    def handle_bad_frame(self):
        """handles bad frame"""
        self.bad_frame_streak += 1
        self.curr_frame.reset_detection()
        if self.bad_frame_streak > self.cfg.longest_bad_streak:
            # do recovery
            print("frame#{} bad frame streak reached".format(self.frame_count))
            self.bad_frame_streak = 0
            self.curr_frame_is_bad = False
            self.process_fresh_frame(True)
        else:
            self.finalize_lane_lines()

    def collect_stats(self):
        """collect stats that are helpful for debugging"""
        frame = self.curr_frame
        if self.curr_frame_is_bad:
            frame = self.past_frames[-1]
        self.stats.append([
            '{}'.format(self.frame_count),
            '{}'.format(len(frame.left_lane.x)),
            '{}'.format(len(frame.left_lane.yvals)),
            '{}'.format(len(frame.right_lane.x)),
            '{}'.format(len(frame.right_lane.yvals)),
            '{:.2f}'.format(frame.left_lane.curverad),
            '{:.2f}'.format(frame.right_lane.curverad),
            '{:.2f}'.format(frame.lane_width_mean),
            '{}'.format(frame.turn_dir)
        ])

    @classmethod
    def get_video_processor(cls, image_processor=None):
        if image_processor:
            img_proc = image_processor
        else:
            img_proc = ImageProcessor.getImageProcessor()
        proc = cls(img_proc, config)
        return proc
