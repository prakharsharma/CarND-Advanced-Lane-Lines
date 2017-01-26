"""
utility functions
"""

import numpy as np


def percent_change(new_val, old_val):
    return np.round(
        100.0 * abs(new_val - old_val)/float(old_val),
        2
    )


def vehicle_pos_wrt_lane_center(image, left_fit, right_fit, xm_per_pix):
    h, w = image.shape[:2]

    x_left = left_fit[0] * h ** 2 + left_fit[1] * h + left_fit[2]
    x_right = right_fit[0] * h ** 2 + right_fit[1] * h + right_fit[2]

    lane_width = x_right - x_left
    lane_center = x_left + lane_width / 2.0

    x_vehicle = w / 2.0

    x_vehicle_off_center = x_vehicle - lane_center
    x_vehicle_off_center_m = x_vehicle_off_center * xm_per_pix

    # print("starting pos of lane, left: {}, right: {}".format(x_left, x_right))
    # print("lane center: {}, vehicle pos: {}".format(lane_center, x_vehicle))
    # print("vehicle is {:.4f}m {} of center".format(
    #     abs(x_vehicle_off_center_m),
    #     "left" if x_vehicle_off_center < 0 else "right")
    # )
    return x_vehicle_off_center_m


def find_curvature(leftx, left_yvals, rightx, right_yvals, xm_per_pix,
                   ym_per_pix):
    left_y_eval = np.max(left_yvals)
    right_y_eval = np.max(right_yvals)

    left_yvals2 = left_yvals * ym_per_pix
    right_yvals2 = right_yvals * ym_per_pix

    left_fit_cr = np.polyfit(left_yvals2, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_yvals2, rightx * xm_per_pix, 2)

    left_curverad = ((1 + (2*left_fit_cr[0]*left_y_eval + left_fit_cr[1])**2)**1.5) \
                                 /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*right_y_eval + right_fit_cr[1])**2)**1.5) \
                                    /np.absolute(2*right_fit_cr[0])
    return left_curverad, right_curverad


def find_peak(warped, center, window_size):
    """find peak in the warped image inside a bounding box around center"""

    x_range = (center[0] - window_size / 2, center[0] + window_size / 2)
    y_range = (center[1] - window_size / 2, center[1] + window_size / 2)

    histogram = np.sum(
        warped[y_range[0]:y_range[1], x_range[0]:x_range[1]],
        axis=0
    )
    peak = np.argmax(histogram)
    if peak:
        return x_range[0] + peak, center[1]


def better_lane_points(warped, left_init_x, left_init_yvals, right_init_x,
                       right_init_yvals, window_size=100):
    """look for lane pixels in a small window around the initial points"""

    left_y, left_x = [], []

    right_y, right_x = [], []

    i = 0
    while i < min(len(left_init_yvals), len(right_init_yvals)):

        # left lane
        peak = find_peak(warped, (left_init_x[i], left_init_yvals[i]),
                         window_size)
        if peak:
            left_x.append(peak[0])
            left_y.append(peak[1])

        # right lane
        peak = find_peak(warped, (right_init_x[i], right_init_yvals[i]),
                         window_size)
        if peak:
            right_x.append(peak[0])
            right_y.append(peak[1])

        i += 1

    if i == len(left_init_yvals):
        # print("left lane ran out")
        init_y = right_init_yvals
        init_x = right_init_x
        y = right_y
        x = right_x
    else:
        # print("right lane ran out")
        init_y = left_init_yvals
        init_x = left_init_x
        y = left_y
        x = left_x

    while i < len(init_y):

        peak = find_peak(warped, (init_x[i], init_y[i]), window_size)
        if peak:
            x.append(peak[0])
            y.append(peak[1])

        i += 1

    left_x = np.array(left_x, dtype=np.uint32)
    left_y = np.array(left_y, dtype=np.uint32)

    right_x = np.array(right_x, dtype=np.uint32)
    right_y = np.array(right_y, dtype=np.uint32)
    # print("left_y: {}, left_x: {}, right_y: {}, right_x: {}".format(
    #     len(left_y), len(left_x), len(right_y), len(right_x))
    # )

    return left_x, left_y, right_x, right_y
