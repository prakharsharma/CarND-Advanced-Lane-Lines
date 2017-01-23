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
