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


def get_features(image, features, cell_size, fg_size=None):
    if image.ndim == 4:
        im_height, im_width, n_img_channel, n_image = image.shape
    else:
        im_height, im_width, n_img_channel = image.shape
        n_image = 1

    colorImage = n_img_channel == 3

    # compute total dimension of all features
    total_feature_dim = 0
    for n in range(len(features)):

        if 'use_for_color' in features[n]['fparams']:
            features[n]['fparams']['useForColor'] = True

        if 'use_for_gray' in features[n]['fparams']:
            features[n]['fparams']['use_for_gray'] = True

        if (features[n]['fparams']['use_for_color'] and colorImage) or\
            (features[n]['fparams']['use_for_gray'] and not colorImage):
            total_feature_dim = total_feature_dim + features[n]['fparams']['n_dim']

    if fg_size is None:
        if image.ndim == 4:
            img = image[:,:,:,0]
        else:
            img = image
        fg_size = features[0]['get_feature'](img, features[0]['fparams'], cell_size).shape
        # if cell_size == -1:
        #
        # else:
        #     fg_size = [np.floor(im_height / cell_size), np.floor(im_width / cell_size)]

    # temporary hack for fixing deep features
    if cell_size == -1:
        cf = features[0]
        if (cf['fparams']['useForColor'] and colorImage)\
                or (cf['fparams']['useForGray'] and not colorImage):
            feature_pixels, support_sz = cf.getFeature(image, cf.fparams, cell_size)
    else:
        # compute the feature set
        if image.ndim == 4:
            feature_pixels = np.zeros((int(fg_size[0]), int(fg_size[1]), total_feature_dim, n_image))
            dim_current = 0
            for n in range(len(features)):
                cf = features[n]
                for l_image in range(n_image):
                    if (cf['fparams']['useForColor'] and colorImage) \
                            or (cf['fparams']['useForGray'] and not colorImage):
                        feature_pixels[:, :, dim_current:dim_current + cf['fparams']['n_dim'], l_image] = \
                            cf['get_feature'](np.squeeze(image[:, :, :, l_image]), cf['fparams'], cell_size)

                dim_current = dim_current + cf['fparams']['n_dim']
        elif image.ndim == 3:
            feature_pixels = np.zeros((int(fg_size[0]), int(fg_size[1]), total_feature_dim))
            dim_current = 0
            for n in range(len(features)):
                cf = features[n]
                if (cf['fparams']['useForColor'] and colorImage)\
                        or (cf['fparams']['useForGray'] and not colorImage):
                    feature_pixels[:, :, dim_current:dim_current+cf['fparams']['n_dim']] =\
                        cf['get_feature'](image, cf['fparams'], cell_size)

                dim_current = dim_current + cf['fparams']['n_dim']

    support_sz = np.array((im_height, im_width))
    return feature_pixels, support_sz


def get_pyhog(img, params, cell_size):
    """
    This function is borrowed from dimatura's implementation of HoG
    reference : https://github.com/dimatura/pyhog
    """
    img = img.astype(np.float64)/255.0
    img_c = img.copy("F")
    hog = pyhog.process(img_c, cell_size)
    return hog