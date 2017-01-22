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