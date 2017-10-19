from image_process import features_pedro_py as pyhog
import numpy as np
from PIL import Image


def get_pixels( img, pos, sz, resize_target):
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

    # extract imgage
    img_patch = img[ys.astype(int), :, :]
    img_patch = img_patch[:, xs.astype(int), :]

    if resize_target.size is 0:
        resized_patch = img_patch
    else:
        img_pil = Image.fromarray(np.uint8(img_patch))
        resized_patch = np.asarray(img_pil.resize(resize_target.astype(int)))
    return resized_patch


def get_pyhog(image, cell_size):
    """
    This function is borrowed from dimatura's implementation of HoG
    reference : https://github.com/dimatura/pyhog
    """
    image = image.astype(np.float64)/255.0
    image_c = image.copy("F")
    hog = pyhog.process(image_c, cell_size)
    return hog