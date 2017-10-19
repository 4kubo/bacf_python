from __future__ import division

import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import cm
from scipy.fftpack import fft2, ifft2
from scipy.signal import hann

from image_process.feature import get_pixels, get_pyhog
from special_operation.convertor import resize_DFT2
from special_operation.resp_newton import resp_newton
from utils.functions import get_subwindow_no_window


class BackgroundAwareCorrelationFilter(object):
    def __init__(self, feature, admm_lambda=0.01, cell_selection_thresh=0.5625,
                 dim_feature=31, filter_max_area=2500, feature_ratio=4, interpolate_response=4,
                 learning_rate=0.013, search_area_scale=5.0, reg_window_power=2,
                 n_scales=5, newton_iterations=5, output_sigma_factor=0.0625,
                 refinement_iterations=1, reg_lambda=0.01, reg_window_edge=3.0,
                 reg_window_min=0.1, scale_step=1.01, search_area_shape='square',
                 save_without_showing=False, debug=False, visualization=True):
        self.feature = feature
        self.admm_lambda = admm_lambda
        self.cell_selection_thresh = cell_selection_thresh
        self.feature_ratio = feature_ratio
        self.dim_feature = dim_feature
        self.filter_max_area = filter_max_area
        self.interpolate_response = interpolate_response
        self.learning_rate = learning_rate
        self.search_area_scale = search_area_scale
        self.reg_window_power = reg_window_power
        self.n_scales = n_scales
        self.newton_iterations = newton_iterations
        self.output_sigma_factor = output_sigma_factor
        self.refinement_iterations = refinement_iterations
        self.reg_lambda = reg_lambda
        self.reg_window_edge = reg_window_edge
        self.reg_window_min = reg_window_min
        self.scale_step = scale_step
        self.search_area_shape = search_area_shape
        self.save_without_showing = save_without_showing
        self.debug = debug
        self.visualization = visualization


    def init(self, img, init_rect):
        self.wsize = np.array((init_rect[3], init_rect[2]))
        self.init_pos = (init_rect[1]+np.floor(self.wsize[0]/2),
                         init_rect[0]+np.floor(self.wsize[1]/2))

        position = np.floor(self.init_pos)
        target_pix_sz = np.floor(self.wsize)
        init_target_sz = target_pix_sz

        # h*w*search_area_scale
        search_area = np.prod(init_target_sz / self.feature_ratio * self.search_area_scale)

        # When the number of cells are small, choose a smaller cell size
        if search_area < self.cell_selection_thresh * self.filter_max_area:
            tmp_cell_size = max(1, np.ceil(np.sqrt(np.prod(init_target_sz * self.search_area_scale) /
                                                   (self.cell_selection_thresh * self.filter_max_area))))
            self.feature_ratio = int(min(self.feature_ratio, tmp_cell_size))
            search_area = np.prod(init_target_sz / self.feature_ratio * self.search_area_scale)

        if search_area > self.filter_max_area:
            scale_factor = np.sqrt(search_area / self.filter_max_area)
        else:
            scale_factor = 1.0

        # Target size, or bounding box size at the initial scale
        base_target_pix_sz = target_pix_sz / scale_factor

        # Window size, taking padding into account
        if self.search_area_shape == 'proportional':
            # proportional area, same aspect ratio as the target
            search_pix_sz = np.floor(base_target_pix_sz * self.search_area_scale)
        elif self.search_area_shape == 'square':
            # square area, ignores the target aspect ratio
            search_pix_sz = np.tile(np.sqrt(np.prod(base_target_pix_sz * self.search_area_scale)), 2)
        elif self.search_area_shape == 'fix_padding':
            search_pix_sz = base_target_pix_sz \
                 + np.sqrt(np.prod(base_target_pix_sz * self.search_area_scale) \
                           + (base_target_pix_sz[0]) - base_target_pix_sz[1] / 4) \
                 - sum(base_target_pix_sz) / 2  # const padding
        else:
            print('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''')

        # Set the size to exactly match the cell size
        self.search_pix_sz = np.round(search_pix_sz / self.feature_ratio) * self.feature_ratio
        pixels = get_pixels(img, position, np.round(self.search_pix_sz * scale_factor), self.search_pix_sz)
        features = self._get_features(pixels)
        # The 2-D size of extracted feature
        self.feature_sz = np.float32(features.shape[:2])

        # Construct the label function
        output_sigma = np.sqrt(np.prod(np.floor(base_target_pix_sz / self.feature_ratio))) * self.output_sigma_factor
        rg = np.roll(np.arange(-np.floor((self.feature_sz[0] - 1) / 2) - 1,
                               np.ceil((self.feature_sz[0] - 1) / 2)),
                     -np.floor((self.feature_sz[0] - 1) / 2).astype(np.int64)) + 1
        cg = np.roll(np.arange(-np.floor((self.feature_sz[1] - 1) / 2) - 1,
                               np.ceil((self.feature_sz[1] - 1) / 2)),
                     -np.floor((self.feature_sz[1] - 1) / 2).astype(np.int64)) + 1
        [cs, rs] = np.meshgrid(cg, rg)
        y = np.exp(-0.5 * (((rs ** 2 + cs ** 2) / output_sigma ** 2)))
        self.yf = fft2(y)

        # Construct cosine window
        self.cos_window = np.dot(hann(int(self.feature_sz[0])).reshape(int(self.feature_sz[0]), 1),
                            hann(int(self.feature_sz[1])).reshape(1, int(self.feature_sz[1])))
        self.multi_cos_window = np.tile(self.cos_window[:, :, np.newaxis, np.newaxis],
                                   (1, 1, self.dim_feature, self.n_scales))

        if self.n_scales > 0:
            scale_exp = np.arange(-np.floor((self.n_scales - 1) / 2),\
                                  np.ceil((self.n_scales - 1) / 2 + 1))
            self.scale_factors = np.power(self.scale_step, scale_exp)

            # Force reasonable scale changes
            self.min_scale_factor = self.scale_step ** np.ceil(np.log(np.max(5 / self.search_pix_sz)) \
                                                                 / np.log(self.scale_step))
            self.max_scale_factor = self.scale_step ** \
                               np.floor(np.log(np.min(img.shape[0:2] / base_target_pix_sz)) \
                                        / np.log(self.scale_step))
        else:
            raise NotImplementedError

        if self.interpolate_response >= 3:
            # Pre-computes the grid that is used for socre optimization
            self.ky = np.roll(np.arange(-np.floor((self.feature_sz[0] - 1) / 2.), np.ceil((self.feature_sz[0] - 1) / 2. + 1))
                         , -np.floor((self.feature_sz[0] - 1) / 2).astype(np.int32), axis=0)
            self.kx = np.roll(np.arange(-np.floor((self.feature_sz[1] - 1) / 2.), np.ceil((self.feature_sz[1] - 1) / 2. + 1)),
                         -np.floor((self.feature_sz[1] - 1) / 2).astype(np.int32), axis=0)
            self.newton_iterations = self.newton_iterations
        else:
            raise NotImplementedError

        # Allocate memory for multi-scale tracking
        self.multires_pixel_template = np.zeros((int(self.search_pix_sz[0]),
                                                 int(self.search_pix_sz[1]),
                                                 img.shape[2], self.n_scales))
        self.small_filter_sz = np.floor(base_target_pix_sz / self.feature_ratio)
        self.base_target_pix_sz = base_target_pix_sz

        return position, scale_factor

    def _get_pixels(self, img, pos, search_pix_sz):
        # Square sub-window:
        if np.isscalar(search_pix_sz):
            search_pix_sz = np.array((search_pix_sz, search_pix_sz))

        # Make sure the size is not to small
        if search_pix_sz[0] < 1:
            search_pix_sz[0] = 2

        if search_pix_sz[1] < 1:
            search_pix_sz[1] = 2

        xs = np.floor(pos[1]) + np.arange(search_pix_sz[1]) - np.floor(search_pix_sz[1] / 2.)
        ys = np.floor(pos[0]) + np.arange(search_pix_sz[0]) - np.floor(search_pix_sz[0] / 2.)

        # Check for out-of-bounds coordinates, and set them to the values at the borders
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[img.shape[1] - 1 < xs] = img.shape[1] - 1
        ys[img.shape[0] - 1 < ys] = img.shape[0] - 1

        # Extract image
        img_patch = img[ys.astype(int), :, :]
        img_patch = img_patch[:, xs.astype(int), :]

        resized_patch = cv2.resize(img_patch, tuple(self.search_pix_sz.astype(int)))
        return resized_patch

    def _get_features(self, pixels):
        if pixels.ndim == 4:
            im_height, im_width, n_img_channel, n_image = pixels.shape
            features = np.stack([self.feature(pixels[..., n], self.feature_ratio)
                                 for n in range(n_image)], -1)
            return features
        else:
            features = self.feature(pixels, self.feature_ratio)
            return features

    def gen_tracker(self, images):
        rect_positions = []

        for i_image, image in enumerate(images):
            def track(position, scale_factor, model_xf, g_f):
                if self.interpolate_response == 1:
                    interp_sz = self.feature_sz * self.feature_ratio
                # Use dynamic interp size
                elif self.interpolate_response == 2:
                    interp_sz = np.floor(self.feature_sz * self.feature_ratio * scale_factor)
                else:
                    interp_sz = self.feature_sz

                # Estimate translation and scaling
                if i_image > 0:
                    old_pos = np.full(position.shape, np.inf)
                    iter = 1

                    # Translation search
                    while iter <= self.refinement_iterations and np.any(old_pos != position):
                        # Get multi - resolution image
                        for scale_ind in range(self.n_scales):
                            search_pix_sz = np.round(self.search_pix_sz * scale_factor * self.scale_factors[scale_ind])
                            self.multires_pixel_template[:,:,:, scale_ind] =\
                                self._get_pixels(image, position, search_pix_sz)

                        features = self._get_features(self.multires_pixel_template)

                        xt = features*self.multi_cos_window
                        del features
                        xtf = fft2(xt, axes=(0, 1))
                        responsef = np.sum(np.conj(g_f[..., np.newaxis])*xtf, axis=2)[..., None]

                        # If we undersampled features, we want to interpolate the
                        # response so it has the same size as the image patch
                        responsef_padded = resize_DFT2(responsef, interp_sz)

                        response = np.real(ifft2(responsef_padded, axes=(0, 1)))

                        if self.debug:
                            a = response[:,:,0]
                            ma, mi = a.max(), a.min()
                            im = (a-mi)/(ma-mi)
                            im = cv2.resize(im, (5*im.shape[0], 5*im.shape[1]))
                            cv2.imshow("resp", im)

                            a = self.multires_pixel_template[:,:,:,0]
                            ma, mi = a.max(), a.min()
                            im = (a-mi)/(ma-mi)
                            cv2.imshow("cropped", im)

                        # find maximum
                        if self.interpolate_response == 3:
                            print('Invalid parameter value for interpolate_response')
                        elif self.interpolate_response == 4:
                            disp_row, disp_col, sind =\
                                resp_newton(response, responsef_padded, self.newton_iterations, self.ky, self.kx, self.feature_sz)
                        else:
                            # Find index of which value is 0 in response
                            row, col, sind = np.unravel_index(np.argmax(response), response.shape,
                                                              order="F")
                            disp_row = np.mod(row - 1 + np.floor((interp_sz(1) - 1) / 2), interp_sz[0])\
                                       - np.floor((interp_sz[0] - 1) / 2)
                            disp_col = np.mod(col - 1 + np.floor((interp_sz(2) - 1) / 2), interp_sz[1])\
                                       - np.floor((interp_sz[1] - 1) / 2)

                        # Calculate translation
                        if self.interpolate_response in (0, 3, 4):
                            translation_vec = np.round(np.array([disp_row, disp_col])
                                                       *self.feature_ratio*scale_factor*self.scale_factors[sind])
                        elif self.interpolate_response == 1:
                            translation_vec = np.round(np.array([disp_row, disp_col])*scale_factor*self.scale_factors[sind])
                        else:
                            assert(self.interpolate_response==2)
                            translation_vec = np.round(np.array([disp_row, disp_col])*self.scale_factors[sind])

                        # Update the scale
                        scale_factor = scale_factor*self.scale_factors[sind]
                        # adjust to make sure we are not too larger or too small
                        if scale_factor < self.min_scale_factor:
                            scale_factor = self.min_scale_factor
                        elif scale_factor > self.max_scale_factor:
                            scale_factor = self.max_scale_factor

                        # Update position
                        old_pos = position
                        position = position + translation_vec
                        iter += 1

                # Extract training sample image region
                search_pix_sz = np.round(self.search_pix_sz*scale_factor)
                pixels = self._get_pixels(image, position, search_pix_sz)

                # Extract features and do windowing
                features = self._get_features(pixels)
                if features.ndim == 4:
                    xl = np.tile(self.cos_window[:, :, np.newaxis, np.newaxis],
                                 (1, 1, features.shape[2], features.shape[3]))
                elif features.ndim == 3:
                    xl = np.tile(self.cos_window[:, :, np.newaxis], (1, 1, features.shape[2]))
                else:
                    print('Strange feature shape!' + features.shape)
                xl = features*xl

                # Take the DFT
                xlf = fft2(xl, axes=(0, 1))

                if (i_image == 0):
                    model_xf = xlf
                else:
                    model_xf = (1 - self.learning_rate)*model_xf + self.learning_rate*xlf

                # ADMM
                g_f = np.zeros(xlf.shape)
                h_f = g_f
                l_f = g_f
                mu = 1
                beta = 10
                mu_max = 10000

                T = np.prod(self.feature_sz)
                S_xx = np.sum(np.conj(model_xf)*model_xf, 2)
                self.admm_iterations = 2
                for i in range(self.admm_iterations):
                    # Solve for G - please refer to the paper for more details
                    B = S_xx + (T * mu)
                    S_lx = np.sum(np.conj(model_xf)*l_f, 2)
                    S_hx = np.sum(np.conj(model_xf)*h_f, 2)
                    tmp_second_term = model_xf*(S_xx*self.yf)[..., None]/(T*mu)\
                                      - model_xf*S_lx[..., None] / mu \
                                      + model_xf*S_hx[..., None]
                    g_f = self.yf[..., None]*model_xf/(T*mu) - l_f/mu + h_f \
                            - tmp_second_term / B[..., None]

                    # Solve for H
                    h = (T / ((mu * T) + self.admm_lambda)) * ifft2(mu * g_f + l_f, axes=(0, 1))
                    [sy, sx, h] = get_subwindow_no_window(h, np.floor(self.feature_sz / 2), self.small_filter_sz)
                    t = np.zeros((int(self.feature_sz[0]), int(self.feature_sz[1]), h.shape[2]), dtype=np.complex128)
                    t[np.int16(sy), np.int16(sx), :] = h
                    h_f = fft2(t, axes=(0, 1))

                    # Update L
                    l_f = l_f + (mu * (g_f - h_f))

                    # Update mu - betha = 10.
                    mu = min(beta * mu, mu_max)

                # Get estimated bbox
                target_sz = np.floor(self.base_target_pix_sz*scale_factor)
                rect_pos = np.r_[position[[1, 0]] - np.floor(target_sz[[1, 0]] / 2), target_sz[[1, 0]]]
                rect_positions.append(rect_pos)
                return rect_pos, position, scale_factor, model_xf, g_f
            yield track

            # # Visualization
            # if self.visualization == 1:
            #     self._visualise(self, image, target_sz, i_image, response)

    def _visualise(self, img, target_sz, frame, response, scale_ind, pixels, g_f, scale_factor):
        xy = self.pos[:2] - target_sz[:2] / 2
        height, width = target_sz
        im_to_show = img
        if im_to_show.ndim == 2:
            im_to_show = np.matlab.repmat(im_to_show[:, :, np.newaxis], [1, 1, 3])

        if 0 < frame:
            resp_sz = np.round(self.search_pix_sz * scale_factor * self.scale_factors[scale_ind])
            sc_ind = int(np.floor((self.n_scales - 1) / 2) + 1)

            resp = np.fft.fftshift(response[:, :, sc_ind])
            m = resp.max()
            min_ = resp.min()
            normalized_resp = (resp - min_) / (m - min_)
            resized_resp = cv2.resize(normalized_resp, tuple(resp_sz.astype(int)))
            resized_resp = Image.fromarray((resized_resp * 255).astype(np.uint8))

            canv = Image.new("RGB", img.shape[:2][::-1])
            x_base = np.floor(self.pos[1]) - np.floor(resp_sz[1] / 2)
            y_base = np.floor(self.pos[0]) - np.floor(resp_sz[0] / 2)
            canv.paste(resized_resp, (int(x_base), int(y_base)))
            canv = np.asarray(canv)
            canv = (cm.jet(canv[:, :, 0])[:, :, :3] * 255).astype(np.uint8)
            main_img = cv2.addWeighted(im_to_show, 1.0, canv, 0.5, 0)

            resp_to_show = (cv2.resize(normalized_resp, self.search_pix_sz.astype(int)) * 255).astype(np.uint8)
            if self.save_without_showing:
                cv2.imwrite(target_dir + '/response/{}.png'.format(frame), resp_to_show)
            else:
                cv2.imshow("response", resp_to_show)
        else:
            target_dir = "{0}/{1}".format(self.run_id, self.target_name)
            self.target_dir = target_dir

            if self.save_without_showing:
                for i in ["/main_img", "/filters", "/image_with_bbox", "/response"]:
                    if not os.path.exists(target_dir + i):
                        os.makedirs(target_dir + i)
                    os.path.join(target_dir + i)
            main_img = im_to_show

        # in opencv, coordinate is in form of (x, y)
        cv2.rectangle(main_img, (int(xy[1]), int(xy[0])), (int(xy[1] + width), int(xy[0] + height)),
                      (255, 0, 0), 6)

        # Plotting 6 filters
        y0, x0 = (self.search_pix_sz / 2 - self.base_target_pix_sz / 2).astype(int)
        y1, x1 = (self.search_pix_sz / 2 + self.base_target_pix_sz / 2).astype(int)
        target_with_bbox = cv2.rectangle(pixels, (x0, y0), (x1, y1), (255, 0, 0), 3)

        filter = np.real(ifft2(g_f, axes=(0, 1)))
        filter = filter.sum(axis=2)
        max_ = filter.max()
        min_ = filter.min()
        f = (filter - min_) / (max_ - min_)
        f = (cv2.resize(f, (200, 200)) * 255).astype(np.uint8)

        if self.save_without_showing:
            cv2.imwrite(target_dir + '/filters/{}.png'.format(frame), f)
            cv2.imwrite(target_dir + '/image_with_bbox/{}.png'.format(frame), target_with_bbox)
            cv2.imwrite(target_dir + '/main_img/{}.png'.format(frame), main_img)
        else:
            # f = filter.sum(axis=2)
            cv2.imshow("filters", f)
            cv2.imshow("image_with_bbox", target_with_bbox)
            cv2.imshow('main', main_img)

        cv2.waitKey(1)