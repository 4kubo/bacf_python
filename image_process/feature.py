import cv2
from image_process import features_pedro_py as pyhog
import numpy as np
from PIL import Image
from numba import jit
from scipy import ndimage


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


def get_random_feature(image, param, cell_size):
    return np.cos(np.random.rand(50, 50, 31))

def get_not_fhog(img, param, cell_size):
    n_orients = 9
    block_size = 1
    n_cell_y = img.shape[0] / cell_size
    n_cell_x = img.shape[1] / cell_size

    dxy = np.array([-1,0,1])
    filteredImageX = ndimage.convolve1d( img, dxy, axis=0, mode='constant' )
    filteredImageY = ndimage.convolve1d( img, dxy, axis=1, mode='constant' )

    gradientMagnitude = np.sqrt(filteredImageX*filteredImageX + filteredImageY*filteredImageY)
    gradientOrientation = np.arctan2( filteredImageY, filteredImageX+0.0001 )
    gradientOrientation[gradientOrientation<0] += np.pi

    go1 = (gradientOrientation*180/np.pi ) / ( 180/n_orients )
    go2 = (gradientOrientation*180/np.pi + (180/n_orients)) / ( 180/n_orients )

    go1 = go1.astype(int)
    go2 = go2.astype(int)

    linInt = ((gradientOrientation*180/np.pi)-np.array(gradientOrientation*180/np.pi,dtype="int"))/(180.0/n_orients)
    go2[linInt==0] = go1[linInt==0]

    go1[go1>=n_orients] = 0
    go2[go2>=n_orients] = 0

    histogram = np.zeros((n_cell_y, n_cell_x, n_orients))
    for i in range(n_cell_y):
        for j in range(n_cell_x):
            go1ij = go1[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size]
            go2ij = go2[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size]
            goij = gradientOrientation[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size]
            gmij = gradientMagnitude[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size]
            linIntij = linInt[i*cell_size:(i+1)*cell_size,j*cell_size:(j+1)*cell_size]
            for k in range(n_orients):
                histogram[i][j][k] += np.sum((1-linIntij[go1ij==k])*gmij[go1ij==k])
                histogram[i][j][k] += np.sum(linIntij[go2ij==k]*gmij[go2ij==k])


    cellSum = np.zeros((n_cell_y, n_cell_x))
    for i in range(n_cell_y):
        for j in range(n_cell_x):
            cellSum[i][j] = np.sum(histogram[i][j][:]*histogram[i][j][:])

    blockIdx = [[[0,0],[-1,0],[0,-1],[-1,-1]], [[0,0],[0,-1],[1,0],[1,-1]], [[0,0],[-1,0],[0,1],[-1,1]], [[0,0],[1,0],[0,1],[1,1]]];
    result = np.empty(0)
    for i in range(n_cell_y):
        for j in range(n_cell_x):
            feat = np.zeros((block_size*block_size,n_orients))
            for k in range(block_size*block_size):
                sum = 0
                for l in range(len(blockIdx[k])):
                    y = (i+blockIdx[k][l][1]+n_cell_y)%n_cell_y
                    x = (j+blockIdx[k][l][0]+n_cell_x)%n_cell_x
                    sum += cellSum[y][x];
                if sum != 0:
                    feat[k][:] = histogram[i][j][:]/sum
            result = np.r_[result,np.sum(feat,0)]
            result = np.r_[result,np.sum(feat,1)]

    sum = np.sum(np.abs(result))
    if sum != 0:
        result = np.sqrt(result/sum)
    print("result.shape", result.shape)
    return result

n_orients = 9
FLT_EPSILON = 1e-07

@jit
def func1(dx, dy, boundary_x, boundary_y, height, width, numChannels):
    r = np.zeros((height, width), np.float32)
    alfa = np.zeros((height, width, 2), np.int)

    for j in range(1, height-1):
        for i in range(1, width-1):
            c = 0
            x = dx[j, i, c]
            y = dy[j, i, c]
            r[j, i] = np.sqrt(x*x + y*y)

            for ch in range(1, numChannels):
                tx = dx[j, i, ch]
                ty = dy[j, i, ch]
                magnitude = np.sqrt(tx*tx + ty*ty)
                if(magnitude > r[j, i]):
                    r[j, i] = magnitude
                    c = ch
                    x = tx
                    y = ty

            mmax = boundary_x[0]*x + boundary_y[0]*y
            maxi = 0

            for kk in range(0, n_orients):
                dotProd = boundary_x[kk]*x + boundary_y[kk]*y
                if(dotProd > mmax):
                    mmax = dotProd
                    maxi = kk
                elif(-dotProd > mmax):
                    mmax = -dotProd
                    maxi = kk + n_orients

            alfa[j, i, 0] = maxi % n_orients
            alfa[j, i, 1] = maxi
    return r, alfa

@jit
def func2(dx, dy, boundary_x, boundary_y, r, alfa, nearest, w, k, height, width, sizeX, sizeY, p, stringSize):
    mapp = np.zeros((sizeX*sizeY*p), np.float32)
    for i in range(sizeY):
        for j in range(sizeX):
            for ii in range(k):
                for jj in range(k):
                    if((i * k + ii > 0) and (i * k + ii < height - 1) and (j * k + jj > 0) and (j * k + jj < width  - 1)):
                        mapp[i*stringSize + j*p + alfa[k*i+ii,j*k+jj,0]] +=  r[k*i+ii,j*k+jj] * w[ii,0] * w[jj,0]
                        mapp[i*stringSize + j*p + alfa[k*i+ii,j*k+jj,1] + n_orients] +=  r[k*i+ii,j*k+jj] * w[ii,0] * w[jj,0]
                        if((i + nearest[ii] >= 0) and (i + nearest[ii] <= sizeY - 1)):
                            mapp[(i+nearest[ii])*stringSize + j*p + alfa[k*i+ii,j*k+jj,0]] += r[k*i+ii,j*k+jj] * w[ii,1] * w[jj,0]
                            mapp[(i+nearest[ii])*stringSize + j*p + alfa[k*i+ii,j*k+jj,1] + n_orients] += r[k*i+ii,j*k+jj] * w[ii,1] * w[jj,0]
                        if((j + nearest[jj] >= 0) and (j + nearest[jj] <= sizeX - 1)):
                            mapp[i*stringSize + (j+nearest[jj])*p + alfa[k*i+ii,j*k+jj,0]] += r[k*i+ii,j*k+jj] * w[ii,0] * w[jj,1]
                            mapp[i*stringSize + (j+nearest[jj])*p + alfa[k*i+ii,j*k+jj,1] + n_orients] += r[k*i+ii,j*k+jj] * w[ii,0] * w[jj,1]
                        if((i + nearest[ii] >= 0) and (i + nearest[ii] <= sizeY - 1) and (j + nearest[jj] >= 0) and (j + nearest[jj] <= sizeX - 1)):
                            mapp[(i+nearest[ii])*stringSize + (j+nearest[jj])*p + alfa[k*i+ii,j*k+jj,0]] += r[k*i+ii,j*k+jj] * w[ii,1] * w[jj,1]
                            mapp[(i+nearest[ii])*stringSize + (j+nearest[jj])*p + alfa[k*i+ii,j*k+jj,1] + n_orients] += r[k*i+ii,j*k+jj] * w[ii,1] * w[jj,1]
    return mapp

def get_fhog(image, params, cell_size):
    """
    reference : https://github.com/uoip/KCFpy
    :param image: 
    :param params: 
    :param cell_size: 
    :return: 
    """
    kernel = np.array([[-1.,  0., 1.]], np.float32)

    height = image.shape[0]
    width = image.shape[1]
    assert(image.ndim==3 and image.shape[2])
    numChannels = 3 #(1 if image.ndim==2 else image.shape[2])

    sizeX = width / cell_size
    sizeY = height / cell_size
    px = 3 * n_orients
    p = px
    stringSize = sizeX * p

    mapp = {}
    mapp['sizeX'] = sizeX
    mapp['sizeY'] = sizeY
    mapp['numFeatures'] = p
    mapp['map'] = np.zeros((mapp['sizeX']*mapp['sizeY']*mapp['numFeatures']), np.float32)

    dx = cv2.filter2D(np.float32(image), -1, kernel)   # np.float32(...) is necessary
    dy = cv2.filter2D(np.float32(image), -1, kernel.T)

    arg_vector = np.arange(n_orients+1).astype(np.float32) * np.pi / n_orients
    boundary_x = np.cos(arg_vector)
    boundary_y = np.sin(arg_vector)

    ### 200x speedup
    r, alfa = func1(dx, dy, boundary_x, boundary_y, height, width, numChannels) #with @jit

    nearest = np.ones((cell_size), np.int)
    nearest[0:cell_size/2] = -1

    w = np.zeros((cell_size, 2), np.float32)
    a_x = np.concatenate((cell_size/2 - np.arange(cell_size/2) - 0.5, np.arange(cell_size/2,cell_size) - cell_size/2 + 0.5)).astype(np.float32)
    b_x = np.concatenate((cell_size/2 + np.arange(cell_size/2) + 0.5, -np.arange(cell_size/2,cell_size) + cell_size/2 - 0.5 + cell_size)).astype(np.float32)
    w[:, 0] = 1.0 / a_x * ((a_x*b_x) / (a_x+b_x))
    w[:, 1] = 1.0 / b_x * ((a_x*b_x) / (a_x+b_x))

    ### 500x speedup
    mapp['map'] = func2(dx, dy, boundary_x, boundary_y, r, alfa, nearest, w, cell_size, height, width, sizeX, sizeY, p, stringSize) #with @jit

    map_tmp = np.reshape(mapp['map'], (sizeY, sizeX, 3*n_orients))
    # fhog = np.concatenate([map_tmp, np.random.randn(map_tmp.shape[0], map_tmp.shape[1], 4)], 2)
    fhog = map_tmp

    return fhog / fhog.max()

def get_pyhog(img, params, cell_size):
    img = img.astype(np.float64)/255.0
    img_c = img.copy("F")
    hog = pyhog.process(img_c, cell_size)
    return hog