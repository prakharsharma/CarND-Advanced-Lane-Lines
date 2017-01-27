"""
contains all the config params
"""

# debug flag
debug = True

# directory for debug data
debugPrefix = './debug'

# Sobel kernel size
sobelKsize = 3

# threshold for sobel X
# sxThresh = (20, 100)
sobelThresh = {
    'x': (20, 100)
}

# threshold for S channel
sThresh = (170, 255)

# meters per pixel in y dimension
ym_per_pix = 30/720.

# meters per pixel in x dimension
xm_per_pix = 3.7/700.

# to load an existing output of camera calibration
load_calibration = True

# fspath to look for output of camera calibration
camera_calibration_out = './camera_calibration.p'

# write output to fs
write_out = True

# output path
output_path = './output_images'

# window used to do sliding window
window_size = 100

# max number of past frames that we keep track of
max_past_frames = 10

# num of frames after we reset
reset_after_frames = 100

# percent diff between left and right lanes
tolerable_change_in_lanes = 85

# minimum number of past frames required for incremental processing
min_past_frames = 3

# longest streak of bad frames allowed
longest_bad_streak = 3

# min len
min_lane_len = 30
