import numpy as np


def get_subwindow_no_window(img, pos, sz):
    # square sub-window:
    if np.isscalar(sz):
        sz = np.array((sz, sz))

    # make sure the size is not to small
    if sz[0] < 1:
        sz[0] = 2

    if sz[1] < 1:
        sz[1] = 2

    xs = np.floor(pos[1]) + np.arange(sz[1]) - np.floor(sz[1] / 2.)
    ys = np.floor(pos[0]) + np.arange(sz[0]) - np.floor(sz[0] / 2.)

    # check for out-of-bounds coordinates, and set them to the values at
    # the borders
    xs[xs < 0] = 0
    ys[ys < 0] = 0
    xs[img.shape[1]-1 < xs] = img.shape[1] - 1
    ys[img.shape[0]-1 < ys] = img.shape[0] - 1

    padded_image = img[np.int16(ys), :, :]
    padded_image = padded_image[:, np.int16(xs), :]

    xs, ys = np.meshgrid(xs, ys)
    return ys, xs, padded_image