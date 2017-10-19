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
from utils.get_sequence_info import get_sequences


class BackgroundAwareCorrelationFilter(object):
    def __init__(self, params, target_seq):
        self.params = params
        self.params.admm_lambda = 0.01
        self.target_name = target_seq
        self.params.visualization = self.params.visualization or self.params.debug
        gt_label, frame_names, n_frame, init_rect = get_sequences(self.params.path_to_sequences, target_seq)
        self.n_frame = n_frame
        self.wsize = np.array((init_rect[3], init_rect[2]))
        self.init_pos = (init_rect[1] + np.floor(self.wsize[0] / 2),
                         init_rect[0] + np.floor(self.wsize[1] / 2))
        self.frame_names = frame_names
        self.t_features = [
            {'get_feature': get_pyhog, 'fparams': {'use_for_color': True, 'n_dim': self.params.dim_feature}}]

    def track(self):
        # parameters
        output_sigma_factor = self.params.output_sigma_factor
        pos = np.floor(self.init_pos)
        target_sz = np.floor(self.wsize)
        init_target_sz = target_sz
        learning_rate = self.params.learning_rate

        # set the feature ratio to the feature-cell size
        feature_ratio = self.params.cell_size

        # h*w*search_area_scale
        search_area = np.prod(init_target_sz / feature_ratio * self.params.search_area_scale)

        # when the number of cells are small, choose a smaller cell size
        if search_area < self.params.cell_selection_thresh * self.params.filter_max_area:
            tmp_cell_size = max(1, np.ceil(np.sqrt(np.prod(init_target_sz * self.params.search_area_scale) /
                                                   (self.params.cell_selection_thresh * self.params.filter_max_area))))
            self.params.cell_size = int(min(feature_ratio, tmp_cell_size))
            feature_ratio = self.params.cell_size
            search_area = np.prod(init_target_sz / feature_ratio * self.params.search_area_scale)

        if search_area > self.params.filter_max_area:
            current_scale_factor = np.sqrt(search_area / self.params.filter_max_area)
        else:
            current_scale_factor = 1.0

        # target size at the initial scale
        base_target_sz = target_sz / current_scale_factor

        # window size, taking padding into account
        if self.params.search_area_shape == 'proportional':
            # proportional area, same aspect ratio as the target
            sz = np.floor(base_target_sz * self.params.search_area_scale)
        elif self.params.search_area_shape == 'square':
            # square area, ignores the target aspect ratio
            sz = np.tile(np.sqrt(np.prod(base_target_sz * self.params.search_area_scale)), 2)
        elif self.params.search_area_shape == 'fix_padding':
            sz = base_target_sz \
                 + np.sqrt(np.prod(base_target_sz * self.params.search_area_scale) \
                           + (base_target_sz(1) - base_target_sz(2)) / 4) \
                 - sum(base_target_sz) / 2  # const padding
        else:
            print('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''')

        # Calculate feature dimension
        img = cv2.imread(self.frame_names[0])

        # set the size to exactly match the cell size
        sz = np.round(sz / feature_ratio) * feature_ratio
        # use_sz = np.floor(sz / feature_ratio)
        pixels = get_pixels(img, pos, np.round(sz * current_scale_factor), sz)
        features, _ = get_features(pixels, self.t_features, self.params.cell_size)
        use_sz = np.float64(features.shape[:2])

        # construct the label function
        output_sigma = np.sqrt(np.prod(np.floor(base_target_sz / feature_ratio))) * output_sigma_factor
        rg = np.roll(np.arange(-np.floor((use_sz[0] - 1) / 2) - 1,
                               np.ceil((use_sz[0] - 1) / 2)),
                     -np.floor((use_sz[0] - 1) / 2).astype(np.int64)) + 1
        cg = np.roll(np.arange(-np.floor((use_sz[1] - 1) / 2) - 1,
                               np.ceil((use_sz[1] - 1) / 2)),
                     -np.floor((use_sz[1] - 1) / 2).astype(np.int64)) + 1
        [cs, rs] = np.meshgrid(cg, rg)
        y = np.exp(-0.5 * (((rs ** 2 + cs ** 2) / output_sigma ** 2)))
        yf = fft2(y)

        if self.params.interpolate_response == 1:
            interp_sz = use_sz * feature_ratio
        else:
            interp_sz = use_sz

        # construct cosine window
        cos_window = np.dot(hann(int(use_sz[0])).reshape(int(use_sz[0]), 1),
                            hann(int(use_sz[1])).reshape(1, int(use_sz[1])))
        multi_cos_window = np.tile(cos_window[:, :, np.newaxis, np.newaxis],
                                   (1, 1, self.params.dim_feature, self.params.n_scales))

        # the search area size
        support_size = np.prod(use_sz)

        if img.shape[2] == 3:
            if np.all(img[:, :, 1] == img[:, :, 2]):
                color_image = False
            else:
                color_image = True
        else:
            color_image = False

        # compute feature dimensionality
        feature_dim = 0

        # for n in range(len(self.t_features)):
        #     if not 'use_for_color' in self.t_features[n]['fparams']:
        #         self.t_features[n]['fparams']['use_for_color'] = True
        #
        #     if self.t_features[n]['fparams']['use_for_color'] and color_image is True:
        #         feature_dim = feature_dim + self.t_features[n]['fparams']['n_dim']
        #
        #     if img.shape[2] > 1 and color_image is False:
        #         img = img[:,:, 1]

        feature_dim = self.t_features[0]['fparams']['n_dim']

        if self.params.n_scales > 0:
            scale_exp = np.arange(-np.floor((self.params.n_scales - 1) / 2), \
                                  np.ceil((self.params.n_scales - 1) / 2 + 1))
            scale_factors = np.power(self.params.scale_step, scale_exp)

            # force reasonable scale changes
            min_scale_factor = self.params.scale_step ** np.ceil(np.log(np.max(5 / sz)) \
                                                                 / np.log(self.params.scale_step))
            max_scale_factor = self.params.scale_step ** \
                               np.floor(np.log(np.min(img.shape[0:2] / base_target_sz)) \
                                        / np.log(self.params.scale_step))

        if self.params.interpolate_response >= 3:
            # Pre-computes the grid that is used for socre optimization
            ky = np.roll(np.arange(-np.floor((use_sz[0] - 1) / 2.), np.ceil((use_sz[0] - 1) / 2. + 1))
                         , -np.floor((use_sz[0] - 1) / 2).astype(np.int32), axis=0)
            kx = np.roll(np.arange(-np.floor((use_sz[1] - 1) / 2.), np.ceil((use_sz[1] - 1) / 2. + 1)),
                         -np.floor((use_sz[1] - 1) / 2).astype(np.int32), axis=0)
            newton_iterations = self.params.newton_iterations

        # Allocate memory for multi-scale tracking
        multires_pixel_template = np.zeros((int(sz[0]), int(sz[1]), img.shape[2], self.params.n_scales))
        small_filter_sz = np.floor(base_target_sz / feature_ratio)

        rect_positions = []

        for frame in range(self.n_frame):
            # load image
            img = cv2.imread(self.frame_names[frame])[:, :, ::-1]
            img = img[..., ::-1]
            # if img.shape[2] > 1 and color_image == False:
            #     img = img[:,:, 0]

            # do not estimate translation and scaling on the first frame,
            #  since we just want to initialize the tracker there
            if frame > 0:
                old_pos = np.full(pos.shape, np.inf)
                iter = 1

                # translation search
                while iter <= self.params.refinement_iterations and np.any(old_pos != pos):
                    # Get multi - resolution image
                    for scale_ind in range(self.params.n_scales):
                        multires_pixel_template[:,:,:, scale_ind] =\
                            get_pixels(img, pos,
                                       np.round(sz*current_scale_factor*scale_factors[scale_ind]), sz)

                    features, _ = get_features(multires_pixel_template, self.t_features,
                                               self.params.cell_size)

                    xt = features*multi_cos_window
                    del features
                    xtf = fft2(xt, axes=(0, 1))
                    responsef = np.sum(np.conj(g_f[..., np.newaxis])*xtf, axis=2)[..., None]
                    # if we undersampled features, we want to interpolate the
                    # response so it has the same size as the image patch
                    if self.params.interpolate_response == 2:
                        # use dynamic interp size
                        interp_sz = np.floor(y.shape*feature_ratio*current_scale_factor)

                    responsef_padded = resize_DFT2(responsef, interp_sz)

                    # response
                    response = np.real(ifft2(responsef_padded, axes=(0, 1)))

                    if self.params.debug:
                        a = response[:,:,0]
                        ma, mi = a.max(), a.min()
                        im = (a-mi)/(ma-mi)
                        im = cv2.resize(im, (5*im.shape[0], 5*im.shape[1]))
                        cv2.imshow("resp", im)

                        a = multires_pixel_template[:,:,:,0]
                        ma, mi = a.max(), a.min()
                        im = (a-mi)/(ma-mi)
                        cv2.imshow("cropped", im)

                    # find maximum
                    if self.params.interpolate_response == 3:
                        print('Invalid parameter value for interpolate_response')
                    elif self.params.interpolate_response == 4:
                        disp_row, disp_col, sind =\
                            resp_newton(response, responsef_padded, newton_iterations, ky, kx, use_sz)
                    else:
                        # Find index of which value is 0 in response
                        row, col, sind = np.unravel_index(np.argmax(response), response.shape,
                                                          order="F")
                        disp_row = np.mod(row - 1 + np.floor((interp_sz(1) - 1) / 2), interp_sz[0])\
                                   - np.floor((interp_sz[0] - 1) / 2)
                        disp_col = np.mod(col - 1 + np.floor((interp_sz(2) - 1) / 2), interp_sz[1])\
                                   - np.floor((interp_sz[1] - 1) / 2)

                    # calculate translation
                    if self.params.interpolate_response in (0, 3, 4):
                        translation_vec = np.round(np.array([disp_row, disp_col])*feature_ratio*current_scale_factor*scale_factors[sind])
                    elif self.params.interpolate_response == 1:
                        translation_vec = np.round(np.array([disp_row, disp_col])*current_scale_factor*scale_factors[sind])
                    elif self.params.interpolate_response == 2:
                        translation_vec = np.round(np.array([disp_row, disp_col])*scale_factors[sind])
                    else:
                        print('yabee')

                    # set the scale
                    current_scale_factor = current_scale_factor*scale_factors[sind]
                    # adjust to make sure we are not too larger or too small
                    if current_scale_factor < min_scale_factor:
                        current_scale_factor = min_scale_factor
                    elif current_scale_factor > max_scale_factor:
                        current_scale_factor = max_scale_factor

                    # update position
                    old_pos = pos
                    pos = pos + translation_vec
                    iter += 1

            # extract training sample image region
            pixels = get_pixels(img, pos, np.round(sz*current_scale_factor), sz)

            # extract features and do windowing
            features, _ = get_features(pixels, self.t_features, self.params.cell_size)

            if features.ndim == 4:
                xl = np.tile(cos_window[:, :, np.newaxis, np.newaxis],
                             (1, 1, features.shape[2], features.shape[3]))
            elif features.ndim == 3:
                xl = np.tile(cos_window[:, :, np.newaxis], (1, 1, features.shape[2]))
            else:
                print('Strange feature shape!' + features.shape)
            xl = features*xl

            # take the DFT and vectorize each feature dimension
            xlf = fft2(xl, axes=(0, 1))

            if (frame == 0):
                model_xf = xlf
            else:
                model_xf = (1 - learning_rate)*model_xf + learning_rate*xlf

            # ADMM
            g_f = np.zeros(xlf.shape)
            h_f = g_f
            l_f = g_f
            mu = 1
            betha = 10
            mumax = 10000

            T = np.prod(use_sz)
            S_xx = np.sum(np.conj(model_xf)*model_xf, 2)
            self.params.admm_iterations = 2
            for i in range(self.params.admm_iterations):
                # solve for G - please refer to the paper for more details
                B = S_xx + (T * mu)
                S_lx = np.sum(np.conj(model_xf)*l_f, 2)
                S_hx = np.sum(np.conj(model_xf)*h_f, 2)
                tmp_second_term = model_xf*(S_xx*yf)[..., None]/(T*mu)\
                                  - model_xf*S_lx[..., None] / mu \
                                  + model_xf*S_hx[..., None]
                g_f = yf[..., None]*model_xf/(T*mu) - l_f/mu + h_f \
                        - tmp_second_term / B[..., None]

                # solve for H
                h = (T / ((mu * T) + self.params.admm_lambda)) * ifft2(mu * g_f + l_f, axes=(0, 1))
                [sy, sx, h] = get_subwindow_no_window(h, np.floor(use_sz / 2), small_filter_sz)
                t = np.zeros((int(use_sz[0]), int(use_sz[1]), h.shape[2]), dtype=np.complex128)
                t[np.int16(sy), np.int16(sx), :] = h
                h_f = fft2(t, axes=(0, 1))

                # update L
                l_f = l_f + (mu * (g_f - h_f))

                # update mu - betha = 10.
                mu = min(betha * mu, mumax)

            target_sz = np.floor(base_target_sz*current_scale_factor)

            # save position and calculate FPS
            rect_pos = np.r_[pos[[1, 0]] - np.floor(target_sz[[1, 0]] / 2), target_sz[[1, 0]]]
            print("{} at {}".format(rect_pos, frame))
            rect_positions.append(rect_pos)

            # visualization
            if self.params.visualization == 1:
                xy = pos[:2] - target_sz[:2] / 2
                height, width = target_sz
                im_to_show = img
                if im_to_show.ndim == 2:
                    im_to_show = np.matlab.repmat(im_to_show[:, :, np.newaxis], [1, 1, 3])

                if 0 < frame:
                    resp_sz = np.round(sz * current_scale_factor * scale_factors[scale_ind])
                    sc_ind = int(np.floor((self.params.n_scales - 1) / 2) + 1)

                    resp = np.fft.fftshift(response[:, :, sc_ind])
                    m = resp.max()
                    min_ = resp.min()
                    normalized_resp = (resp - min_) / (m - min_)
                    resized_resp = cv2.resize(normalized_resp, tuple(resp_sz.astype(int)))
                    resized_resp = Image.fromarray((resized_resp * 255).astype(np.uint8))

                    canv = Image.new("RGB", img.shape[:2][::-1])
                    x_base = np.floor(pos[1]) - np.floor(resp_sz[1] / 2)
                    y_base = np.floor(pos[0]) - np.floor(resp_sz[0] / 2)
                    canv.paste(resized_resp, (int(x_base), int(y_base)))
                    canv = np.asarray(canv)
                    canv = (cm.jet(canv[:, :, 0])[:, :, :3] * 255).astype(np.uint8)
                    main_img = cv2.addWeighted(im_to_show, 1.0, canv, 0.5, 0)

                    resp_to_show = (cv2.resize(normalized_resp, pixels.shape[:2]) * 255).astype(np.uint8)
                    if self.params.save_without_showing:
                        cv2.imwrite(target_dir + '/response/{}.png'.format(frame), resp_to_show)
                    else:
                        cv2.imshow("response", resp_to_show)
                else:
                    target_dir = "{0}/{1}".format(self.params.run_id, self.target_name)
                    self.target_dir = target_dir

                    if self.params.save_without_showing:
                        for i in ["/main_img", "/filters", "/image_with_bbox", "/response"]:
                            if not os.path.exists(target_dir + i):
                                os.makedirs(target_dir + i)
                            os.path.join(target_dir + i)
                    main_img = im_to_show

                # in opencv, coordinate is in form of (x, y)
                cv2.rectangle(main_img, (int(xy[1]), int(xy[0])), (int(xy[1] + width), int(xy[0] + height)),
                              (255, 0, 0), 6)

                # Plotting 6 filters
                y0, x0 = (sz / 2 - base_target_sz / 2).astype(int)
                y1, x1 = (sz / 2 + base_target_sz / 2).astype(int)
                target_with_bbox = cv2.rectangle(pixels, (x0, y0), (x1, y1), (255, 0, 0), 3)

                filter = np.real(ifft2(g_f, axes=(0, 1)))
                filter = filter.sum(axis=2)
                max_ = filter.max()
                min_ = filter.min()
                f = (filter - min_) / (max_ - min_)
                f = (cv2.resize(f, (200, 200)) * 255).astype(np.uint8)

                if self.params.save_without_showing:
                    cv2.imwrite(target_dir + '/filters/{}.png'.format(frame), f)
                    cv2.imwrite(target_dir + '/image_with_bbox/{}.png'.format(frame), target_with_bbox)
                    cv2.imwrite(target_dir + '/main_img/{}.png'.format(frame), main_img)
                else:
                    # f = filter.sum(axis=2)
                    cv2.imshow("filters", f)
                    cv2.imshow("image_with_bbox", target_with_bbox)
                    cv2.imshow('main', main_img)

                cv2.waitKey(1)

        return rect_positions

class BackgroundAwareCorrelationFilte(object):
    def __init__(self, feature, target_seq, path_to_seq, admm_lambda=0.01, cell_selection_thresh=0.5625,
                 dim_feature=31, filter_max_area=2500, feature_ratio=4, interpolate_response=4,
                 learning_rate=0.013, search_area_scale=5.0, reg_window_power=2,
                 n_scales=5, newton_iterations=5, output_sigma_factor=0.0625,
                 refinement_iterations=1, reg_lambda=0.01, reg_sparsity_threshold=0.05,
                 reg_window_edge=3.0, reg_window_min=0.1, scale_step=1.01, search_area_shape='square',
                 save_without_showing=False, debug=False, visualization=True):
        self.feature = feature
        self.admm_lambda=admm_lambda
        self.cell_selection_thresh=cell_selection_thresh
        self.feature_ratio=feature_ratio
        self.dim_feature=dim_feature
        self.filter_max_area=filter_max_area
        self.interpolate_response=interpolate_response
        self.learning_rate=learning_rate
        self.search_area_scale=search_area_scale
        self.reg_window_power=reg_window_power
        self.n_scales=n_scales
        self.newton_iterations=newton_iterations
        self.output_sigma_factor=output_sigma_factor
        self.refinement_iterations=refinement_iterations
        self.reg_lambda=reg_lambda
        self.reg_sparsity_threshold=reg_sparsity_threshold
        self.reg_window_edge=reg_window_edge
        self.reg_window_min=reg_window_min
        self.scale_step = scale_step
        self.search_area_shape=search_area_shape
        self.save_without_showing=save_without_showing
        self.debug=debug
        self.visualization=visualization

        gt_label, self.frame_names, self.n_frame, init_rect = get_sequences(path_to_seq, target_seq)

        self.wsize = np.array((init_rect[3], init_rect[2]))
        self.init_pos = (init_rect[1]+np.floor(self.wsize[0]/2),
                         init_rect[0]+np.floor(self.wsize[1]/2))

    def init(self):
        # parameters
        self.pos = np.floor(self.init_pos)
        target_pix_sz = np.floor(self.wsize)
        init_target_sz = target_pix_sz

        # h*w*search_area_scale
        search_area = np.prod(init_target_sz / self.feature_ratio * self.search_area_scale)

        # when the number of cells are small, choose a smaller cell size
        if search_area < self.cell_selection_thresh * self.filter_max_area:
            tmp_cell_size = max(1, np.ceil(np.sqrt(np.prod(init_target_sz * self.search_area_scale) /
                                                   (self.cell_selection_thresh * self.filter_max_area))))
            self.feature_ratio = int(min(self.feature_ratio, tmp_cell_size))
            search_area = np.prod(init_target_sz / self.feature_ratio * self.search_area_scale)

        if search_area > self.filter_max_area:
            self.current_scale_factor = np.sqrt(search_area / self.filter_max_area)
        else:
            self.current_scale_factor = 1.0

        # target size, or bounding box size at the initial scale
        base_target_pix_sz = target_pix_sz / self.current_scale_factor

        # window size, taking padding into account
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

        # Calculate feature dimension
        img = cv2.imread(self.frame_names[0])

        # set the size to exactly match the cell size
        self.search_pix_sz = np.round(search_pix_sz / self.feature_ratio) * self.feature_ratio
        # use_sz = np.floor(sz / feature_ratio)
        pixels = get_pixels(img, self.pos, np.round(self.search_pix_sz * self.current_scale_factor), self.search_pix_sz)
        # features, _ = get_features(pixels, self.t_features, self.feature_ratio)
        features = self.get_features(pixels)
        # The 2-D size of extracted feature
        self.feature_sz = np.float32(features.shape[:2])

        # construct the label function
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

        # construct cosine window
        self.cos_window = np.dot(hann(int(self.feature_sz[0])).reshape(int(self.feature_sz[0]), 1),
                            hann(int(self.feature_sz[1])).reshape(1, int(self.feature_sz[1])))
        self.multi_cos_window = np.tile(self.cos_window[:, :, np.newaxis, np.newaxis],
                                   (1, 1, self.dim_feature, self.n_scales))

        if self.n_scales > 0:
            scale_exp = np.arange(-np.floor((self.n_scales - 1) / 2),\
                                  np.ceil((self.n_scales - 1) / 2 + 1))
            self.scale_factors = np.power(self.scale_step, scale_exp)

            # force reasonable scale changes
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
        self.multires_pixel_template = np.zeros((int(self.search_pix_sz[0]), int(self.search_pix_sz[1]), img.shape[2], self.n_scales))
        self.small_filter_sz = np.floor(base_target_pix_sz / self.feature_ratio)
        self.base_target_pix_sz = base_target_pix_sz

    def _get_pixels(self, img, pos, search_pix_sz):
        # square sub-window:
        if np.isscalar(search_pix_sz):
            search_pix_sz = np.array((search_pix_sz, search_pix_sz))

        # make sure the size is not to small
        if search_pix_sz[0] < 1:
            search_pix_sz[0] = 2

        if search_pix_sz[1] < 1:
            search_pix_sz[1] = 2

        xs = np.floor(pos[1]) + np.arange(search_pix_sz[1]) - np.floor(search_pix_sz[1] / 2.)
        ys = np.floor(pos[0]) + np.arange(search_pix_sz[0]) - np.floor(search_pix_sz[0] / 2.)

        # check for out-of-bounds coordinates, and set them to the values at
        # the borders
        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[img.shape[1] - 1 < xs] = img.shape[1] - 1
        ys[img.shape[0] - 1 < ys] = img.shape[0] - 1

        # extract imgage
        img_patch = img[ys.astype(int), :, :]
        img_patch = img_patch[:, xs.astype(int), :]

        # img_pil = Image.fromarray(np.uint8(img_patch))
        # resized_patch = np.asarray(img_pil.resize(self.search_pix_search_pix_sz.astype(int)))
        resized_patch = cv2.resize(img_patch, tuple(self.search_pix_sz.astype(int)))
        return resized_patch

    def get_features(self, pixels):
        if pixels.ndim == 4:
            im_height, im_width, n_img_channel, n_image = pixels.shape
            features = np.stack([self.feature(pixels[..., n], self.feature_ratio) for n in range(n_image)], -1)
            return features
        else:
            im_height, im_width, n_img_channel = pixels.shape
            features = self.feature(pixels, self.feature_ratio)
            return features

    def track(self):
        if self.interpolate_response == 1:
            interp_sz = self.feature_sz * self.feature_ratio
        else:
            interp_sz = self.feature_sz

        rect_positions = []

        for frame in range(self.n_frame):
            # load image
            img = cv2.imread(self.frame_names[frame])[:, :, ::-1]
            img = img[..., ::-1]
            # if img.shape[2] > 1 and color_image == False:
            #     img = img[:,:, 0]

            # do not estimate translation and scaling on the first frame,
            #  since we just want to initialize the tracker there
            if frame > 0:
                old_pos = np.full(self.pos.shape, np.inf)
                iter = 1

                # translation search
                while iter <= self.refinement_iterations and np.any(old_pos != self.pos):
                    # Get multi - resolution image
                    for scale_ind in range(self.n_scales):
                        search_pix_sz = np.round(self.search_pix_sz * self.current_scale_factor * self.scale_factors[scale_ind])
                        self.multires_pixel_template[:,:,:, scale_ind] =\
                            self._get_pixels(img, self.pos, search_pix_sz)

                    features = self.get_features(self.multires_pixel_template)

                    xt = features*self.multi_cos_window
                    del features
                    xtf = fft2(xt, axes=(0, 1))
                    responsef = np.sum(np.conj(g_f[..., np.newaxis])*xtf, axis=2)[..., None]
                    # if we undersampled features, we want to interpolate the
                    # response so it has the same size as the image patch
                    if self.interpolate_response == 2:
                        # Use dynamic interp size
                        interp_sz = np.floor(self.feature_sz*self.feature_ratio*self.current_scale_factor)

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

                    # calculate translation
                    if self.interpolate_response in (0, 3, 4):
                        translation_vec = np.round(np.array([disp_row, disp_col])
                                                   *self.feature_ratio*self.current_scale_factor*self.scale_factors[sind])
                    elif self.interpolate_response == 1:
                        translation_vec = np.round(np.array([disp_row, disp_col])*self.current_scale_factor*self.scale_factors[sind])
                    else:
                        assert(self.interpolate_response==2)
                        translation_vec = np.round(np.array([disp_row, disp_col])*self.scale_factors[sind])

                    # set the scale
                    self.current_scale_factor = self.current_scale_factor*self.scale_factors[sind]
                    # adjust to make sure we are not too larger or too small
                    if self.current_scale_factor < self.min_scale_factor:
                        self.current_scale_factor = self.min_scale_factor
                    elif self.current_scale_factor > self.max_scale_factor:
                        self.current_scale_factor = self.max_scale_factor

                    # update position
                    old_pos = self.pos
                    self.pos = self.pos + translation_vec
                    iter += 1

            # Extract training sample image region
            search_pix_sz = np.round(self.search_pix_sz*self.current_scale_factor)
            pixels = self._get_pixels(img, self.pos, search_pix_sz)

            # extract features and do windowing
            features = self.get_features(pixels)

            if features.ndim == 4:
                xl = np.tile(self.cos_window[:, :, np.newaxis, np.newaxis],
                             (1, 1, features.shape[2], features.shape[3]))
            elif features.ndim == 3:
                xl = np.tile(self.cos_window[:, :, np.newaxis], (1, 1, features.shape[2]))
            else:
                print('Strange feature shape!' + features.shape)
            xl = features*xl

            # take the DFT and vectorize each feature dimension
            xlf = fft2(xl, axes=(0, 1))

            if (frame == 0):
                model_xf = xlf
            else:
                model_xf = (1 - self.learning_rate)*model_xf + self.learning_rate*xlf

            # ADMM
            g_f = np.zeros(xlf.shape)
            h_f = g_f
            l_f = g_f
            mu = 1
            betha = 10
            mumax = 10000

            T = np.prod(self.feature_sz)
            S_xx = np.sum(np.conj(model_xf)*model_xf, 2)
            self.admm_iterations = 2
            for i in range(self.admm_iterations):
                # solve for G - please refer to the paper for more details
                B = S_xx + (T * mu)
                S_lx = np.sum(np.conj(model_xf)*l_f, 2)
                S_hx = np.sum(np.conj(model_xf)*h_f, 2)
                tmp_second_term = model_xf*(S_xx*self.yf)[..., None]/(T*mu)\
                                  - model_xf*S_lx[..., None] / mu \
                                  + model_xf*S_hx[..., None]
                g_f = self.yf[..., None]*model_xf/(T*mu) - l_f/mu + h_f \
                        - tmp_second_term / B[..., None]

                # solve for H
                h = (T / ((mu * T) + self.admm_lambda)) * ifft2(mu * g_f + l_f, axes=(0, 1))
                [sy, sx, h] = get_subwindow_no_window(h, np.floor(self.feature_sz / 2), self.small_filter_sz)
                t = np.zeros((int(self.feature_sz[0]), int(self.feature_sz[1]), h.shape[2]), dtype=np.complex128)
                t[np.int16(sy), np.int16(sx), :] = h
                h_f = fft2(t, axes=(0, 1))

                # update L
                l_f = l_f + (mu * (g_f - h_f))

                # update mu - betha = 10.
                mu = min(betha * mu, mumax)

            target_sz = np.floor(self.base_target_pix_sz*self.current_scale_factor)

            # save position and calculate FPS
            rect_pos = np.r_[self.pos[[1, 0]] - np.floor(target_sz[[1, 0]] / 2), target_sz[[1, 0]]]
            print("{} at {}".format(rect_pos, frame))
            rect_positions.append(rect_pos)

            # # Visualization
            # if self.visualization == 1:
            #     self._visualise(self, img, target_sz, frame, response)

        return rect_positions

    def _visualise(self, img, target_sz, frame, response, scale_ind, pixels, g_f):
        xy = self.pos[:2] - target_sz[:2] / 2
        height, width = target_sz
        im_to_show = img
        if im_to_show.ndim == 2:
            im_to_show = np.matlab.repmat(im_to_show[:, :, np.newaxis], [1, 1, 3])

        if 0 < frame:
            resp_sz = np.round(self.search_pix_sz * self.current_scale_factor * self.scale_factors[scale_ind])
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