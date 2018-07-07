from __future__ import division

import matplotlib as mpl
mpl.use("tkagg")

import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import cm
from scipy.fftpack import fft2, ifft2
from scipy.signal import hann, fftconvolve
import traceback

from image_process.feature import get_pixels, get_pixel
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
                 save_without_showing=False, debug=False, visualization=True,
                 fixed_size=None, is_redetection=False, is_entire_redection=False,
                 redetection_search_area_scale=2.0, psr_threshold=1.0):
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
        self._fixed_size = fixed_size
        self._is_redetction = is_redetection
        self._is_entire_redetction = is_entire_redection
        self._redetection_search_area_scale = redetection_search_area_scale
        self._psr_threshold = psr_threshold

    def init(self, img, init_rect):
        init_rect = np.array(init_rect)
        self.frame_i = 0
        self.wsize = np.array((init_rect[3], init_rect[2]))
        position = (init_rect[1]+np.floor(self.wsize[0]/2),
                         init_rect[0]+np.floor(self.wsize[1]/2))
        self._rect_pos = init_rect

        self._position = np.floor(position)
        target_pix_sz = np.floor(self.wsize)
        self.target_sz = target_pix_sz

        # h*w*search_area_scale
        search_area = np.prod(self.target_sz / self.feature_ratio * self.search_area_scale)

        # When the number of cells are small, choose a smaller cell size
        if search_area < self.cell_selection_thresh * self.filter_max_area:
            tmp_cell_size = max(1, np.ceil(np.sqrt(np.prod(self.target_sz * self.search_area_scale) /
                                                   (self.cell_selection_thresh * self.filter_max_area))))
            self.feature_ratio = int(min(self.feature_ratio, tmp_cell_size))
            search_area = np.prod(self.target_sz / self.feature_ratio * self.search_area_scale)

        if self._fixed_size is not None:
            # Scale factor of an extracted pixel area to the fixed size
            fixed_size = np.array(self._fixed_size)
            scale_factor = np.sqrt(search_area / np.prod(fixed_size/self.feature_ratio))

            # Target size, or bounding box size at the initial scale
            base_target_pix_sz = target_pix_sz / scale_factor

            self.search_pix_sz = fixed_size
        else:
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

        self._scale_factor = scale_factor

        pixel = get_pixel(img, self._position, np.round(self.search_pix_sz * scale_factor), self.search_pix_sz)
        features = self._get_features(pixel)
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
        # Initialization of inner parameters
        initial_g_f = np.zeros(features.shape)

        self._train(img, initial_g_f)

        self.frame = img
        self.pixel = pixel

        return pixel

    def track(self, frame):
        if self.interpolate_response == 1:
            interp_sz = self.feature_sz * self.feature_ratio
        # Use dynamic interp size
        elif self.interpolate_response == 2:
            interp_sz = np.floor(self.feature_sz * self.feature_ratio * self._scale_factor)
        else:
            interp_sz = self.feature_sz

        # Estimate translation and scaling
        old_pos = np.full(self._position.shape, np.inf)
        position = self._position
        iter = 1

        # Translation search
        while iter <= self.refinement_iterations and np.any(old_pos != position):
            # Get multi - resolution image
            target_sz = np.round(self.search_pix_sz)
            search_pix_sz = np.round(target_sz * self._scale_factor * self.scale_factors[:, None])
            multires_pixel_template = get_pixels(frame, position, self.n_scales, search_pix_sz, target_sz)

            features = self._get_features(multires_pixel_template)
            xt = features*self.multi_cos_window
            xtf = fft2(xt, axes=(0, 1))
            responsef = np.sum(np.conj(self._adaptive_g_f[..., np.newaxis])*xtf, axis=2)[..., None]

            # If we undersampled features, we want to interpolate the
            # response so it has the same size as the image patch
            responsef_padded = resize_DFT2(responsef, interp_sz)

            response = np.real(ifft2(responsef_padded, axes=(0, 1)))

            psr = self._get_psr(response)
            translation_vec, sind = \
                self._find_displacement(response, responsef_padded, frame, position, psr)

            # Update the scale
            scale_factor = self._scale_factor*self.scale_factors[sind]
            # Adjust to make sure we are not too larger or too small
            self._scale_factor = max(self.min_scale_factor,
                                     min(self.max_scale_factor, scale_factor))
            # Update position
            old_pos = position
            position = position + translation_vec
            iter += 1

        # Update of the latent state if translation search is finished within the limitation
        target_sz = np.floor(self.base_target_pix_sz * self._scale_factor)

        self.frame = frame
        self.frame_i += 1
        self.target_sz = target_sz
        self.pixel = multires_pixel_template[:, :, :, sind]
        self.features = features[:, :, :, sind]
        self.psr = psr
        self.response = response
        self._rect_pos = np.r_[position[[1, 0]] - np.floor(target_sz[[1, 0]] / 2), target_sz[[1, 0]]]
        self._position = position
        self._sind = sind

        return self.pixel, self.response

    def train(self, image):
        self._train(image, self._g_f, adaptive_g_f=self._adaptive_g_f)

    def get_state(self):
        return self._position, self._rect_pos, self._g_f, self._features

    def _get_features(self, pixels):
        if pixels.ndim == 4:
            im_height, im_width, n_img_channel, n_image = pixels.shape
            features = np.stack([self.feature(pixels[..., n], self.feature_ratio)
                                 for n in range(n_image)], -1)
            return features
        else:
            features = self.feature(pixels, self.feature_ratio)
            return features

    def _train(self, image, g_f, adaptive_g_f=None):
        """
        Train on current samples
        Args:
            image:
            scale_factor:
            position:
            model_xf:

        Returns:
            model_xf
            g_f
        """
        # Extract training sample image region
        xl = self._extract_feature(image, self._position, self._scale_factor)

        # Take the DFT
        xlf = fft2(xl, axes=(0, 1))

        # ADMM
        h_f = g_f
        l_f = g_f
        mu = 1
        beta = 10
        mu_max = 10000

        T = np.prod(self.feature_sz)
        S_xx = np.sum(np.conj(xlf) * xlf, 2)
        self.admm_iterations = 2
        for i in range(self.admm_iterations):
            # Solve for G - please refer to the paper for more details
            B = S_xx + (T * mu)
            S_lx = np.sum(np.conj(xlf) * l_f, 2)
            S_hx = np.sum(np.conj(xlf) * h_f, 2)
            tmp_second_term = xlf * (S_xx * self.yf)[..., None] / (T * mu) \
                              - xlf * S_lx[..., None] / mu \
                              + xlf * S_hx[..., None]
            g_f = self.yf[..., None] * xlf / (T * mu) - l_f / mu + h_f \
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

        # Online update of the classifier
        adaptive_g_f = g_f if adaptive_g_f is None\
            else (1 - self.learning_rate) * adaptive_g_f + self.learning_rate * g_f

        self._features = xl
        self._g_f = g_f
        self._adaptive_g_f = adaptive_g_f

    def _extract_feature(self, image, position, scale_factor):
        # Extract training sample image region
        search_pix_sz = np.round(self.search_pix_sz * scale_factor)
        raw_patch = get_pixel(image, position, search_pix_sz, self.search_pix_sz)
        pixels = cv2.resize(raw_patch, tuple(self.search_pix_sz.astype(int)))

        # Extract features and do windowing
        features = self._get_features(pixels)
        if features.ndim == 4:
            xl = np.tile(self.cos_window[:, :, np.newaxis, np.newaxis],
                         (1, 1, features.shape[2], features.shape[3]))
        elif features.ndim == 3:
            xl = np.tile(self.cos_window[:, :, np.newaxis], (1, 1, features.shape[2]))
        else:
            print('Strange feature shape!' + features.shape)
            raise NotImplementedError
        xl = features * xl
        return xl

    def _find_displacement(self, response, responsef_padded, image, old_pos, psr=None):
        # Redetect the target using PSR value
        # Currently, redetction funciont doesn't work well
        if self._is_redetction:
            # Check detection failure
            if psr is not None and psr < self._psr_threshold:
                # Redetection over an entire input image
                if self._is_entire_redetction:
                    feature = self._get_features(image)

                    feature_ratio = np.array(image.shape[:2]).astype(float) \
                                    / np.array(feature.shape[:2])
                    feature_ratio = np.sqrt(np.prod(feature_ratio))

                    relative_position = self._correlate_with_filter(feature)
                    position = relative_position * feature_ratio
                    translation = old_pos - position

                # Redetection over an extended patch region
                else:
                    target_sz = self.search_pix_sz * self._redetection_search_area_scale
                    search_pix_sz = np.round(target_sz * self._scale_factor)
                    target_sz = np.round(target_sz)
                    raw_patch = get_pixels(image, old_pos, self.n_scales, search_pix_sz, target_sz)
                    feature = self._get_features(raw_patch)

                    feature_ratios = np.array(raw_patch.shape[:2]) / np.array(feature.shape[:2]).astype(float)
                    feature_ratio = np.sqrt(np.prod(feature_ratios))

                    relative_position = self._correlate_with_filter(feature)
                    relative_translation = relative_position - np.array(feature.shape[0:2]) / 2.0
                    translation = relative_translation * feature_ratio

                sind = int(self.n_scales / 2.)
                return translation, sind
            else:
                disp_row, disp_col, sind = \
                    resp_newton(response, responsef_padded, self.newton_iterations,
                                self.ky, self.kx, self.feature_sz)
        else:
            disp_row, disp_col, sind = \
                resp_newton(response, responsef_padded, self.newton_iterations,
                            self.ky, self.kx, self.feature_sz)

        # Calculate translation
        if self.interpolate_response in (0, 3, 4):
            translation = np.round(np.array([disp_row, disp_col])
                                       * self.feature_ratio * self._scale_factor * self.scale_factors[sind])
        elif self.interpolate_response == 1:
            translation = np.round(np.array([disp_row, disp_col]) * self._scale_factor * self.scale_factors[sind])
        else:
            assert (self.interpolate_response == 2)
            translation = np.round(np.array([disp_row, disp_col]) * self.scale_factors[sind])
        return translation, sind

    def _correlate_with_filter(self, feature):
        adaptive_g_f = np.real(self._adaptive_g_f)
        adaptive_g = ifft2(adaptive_g_f, axes=(0, 1))
        # Because result of using abs() is better
        response_map = \
            np.mean([fftconvolve(feature[:, :, i], adaptive_g[::-1, ::-1, i],
                                 mode="same")
                     for i in range(self.dim_feature)], axis=0)
        response_map = np.abs(response_map)
        relative_position =\
            np.array(np.unravel_index(np.argmax(response_map), response_map.shape[0:2]))
        return relative_position

    def _get_psr(self, response):
        """
        Get Peak-to-sidelobe Ratio
        Args:
            response:

        Returns:

        """
        shifted_resp = np.fft.fftshift(response)
        shape = np.array(response.shape)
        # Use only the maximal scale
        x, y, c = np.unravel_index(np.argmax(shifted_resp), shape)

        img_patch = shifted_resp[:, :, c:c + 1]
        position = np.array((x, y))
        search_pix_sz = shape[0:2] / 5
        raw_resp = get_pixel(img_patch, position, search_pix_sz)
        cropped_resp = cv2.resize(raw_resp, tuple(self.search_pix_sz.astype(int)))
        psr = (cropped_resp.max() - cropped_resp.mean()) / (cropped_resp.std() + 1e-10)
        return psr

    def visualise(self, report_id, is_simplest=False, is_detailed=False,
                  save_without_showing=""):
        """
        Visualize current tracking state
        Args:
            report_id(str): Strings which specifies input frames
            is_simplest(bool): If True, visualize frame only with bounding box.
                 This argument has priority over `is_detailed`
            is_detailed(bool): If True, show some visualization
            save_without_showing(bool):
        """
        if is_simplest:
            tl = (int(self._rect_pos[0]), int(self._rect_pos[1]))
            br = (int(self._rect_pos[0] + self._rect_pos[2]),
                  int(self._rect_pos[1] + self._rect_pos[3]))
            image_with_bbox = cv2.rectangle(self.frame, tl, br, (255, 0, 0), 3)
            if save_without_showing:
                if self.frame_i == 0:
                    path_to_save = "{0}/frames_with_bbox".format(save_without_showing)
                    if not os.path.exists(path_to_save):
                        os.makedirs(path_to_save)
                        os.path.join(path_to_save)
                path_i_to_save = "{0}/frames_with_bbox/{1}/{2}.png"\
                    .format(save_without_showing, report_id, self.frame_i)
                cv2.imwrite(path_i_to_save, image_with_bbox)
            else:
                cv2.imshow("frame_with_bbox", image_with_bbox)
            cv2.waitKey(1)
        else:
            xy = self._position[:2] - self.target_sz[:2] / 2
            height, width = self.target_sz
            im_to_show = self.frame
            if im_to_show.ndim == 2:
                im_to_show = np.matlab.repmat(im_to_show[:, :, np.newaxis], [1, 1, 3])

            if 0 < self.frame_i:
                resp_sz = np.round(self.search_pix_sz
                                   * self._scale_factor
                                   * self.scale_factors[self._sind])
                sc_ind = int(np.floor((self.n_scales - 1) / 2) + 1)

                resp = np.fft.fftshift(self.response[:, :, sc_ind])
                m = resp.max()
                min_ = resp.min()
                normalized_resp = (resp - min_) / (m - min_)
                resized_resp = cv2.resize(normalized_resp, tuple(resp_sz.astype(int)))
                resized_resp = Image.fromarray((resized_resp * 255).astype(np.uint8))

                canv = Image.new("RGB", self.frame.shape[:2][::-1])
                x_base = np.floor(self._position[1]) - np.floor(resp_sz[1] / 2)
                y_base = np.floor(self._position[0]) - np.floor(resp_sz[0] / 2)
                canv.paste(resized_resp, (int(x_base), int(y_base)))
                canv = np.asarray(canv)
                canv = (cm.jet(canv[:, :, 0])[:, :, :3] * 255).astype(np.uint8)
                main_img = cv2.addWeighted(im_to_show, 1.0, canv, 0.5, 0)
                if is_detailed:
                    target_sz = tuple(self.search_pix_sz.astype(int))
                    resp_to_show = (cv2.resize(normalized_resp, target_sz) * 255).astype(np.uint8)
                    if save_without_showing:
                        path_i_to_save = "{0}/responses/{1}/{2}.png"\
                            .format(save_without_showing, report_id, self.frame_i)
                        cv2.imwrite(path_i_to_save, resp_to_show)
                    else:
                        cv2.imshow("response", resp_to_show)
            else:
                if save_without_showing:
                    for dir_name in ["/regions_with_bbox", "/filters", "/frames_with_response", "/responses"]:
                        path_to_save = "{0}/{1}/{2}".format(save_without_showing, dir_name, report_id)
                        if not os.path.exists(path_to_save):
                            os.makedirs(path_to_save)
                        os.path.join(path_to_save)
                main_img = im_to_show

            # in opencv, coordinate is in form of (x, y)
            cv2.rectangle(main_img, (int(xy[1]), int(xy[0])), (int(xy[1] + width), int(xy[0] + height)),
                          (255, 0, 0), 6)

            if is_detailed:
                # Extracted region from a given frame
                y0, x0 = (self.search_pix_sz / 2 - self.base_target_pix_sz / 2).astype(int)
                y1, x1 = (self.search_pix_sz / 2 + self.base_target_pix_sz / 2).astype(int)
                frame = self.pixel.astype(np.uint8)
                target_with_bbox = cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 3)
                # Correlation filter
                filter = np.real(ifft2(self._g_f, axes=(0, 1)))
                filter = filter.sum(axis=2)
                max_ = filter.max()
                min_ = filter.min()
                f = (filter - min_) / (max_ - min_)
                f = (cv2.resize(f, (200, 200)) * 255).astype(np.uint8)

                if save_without_showing:
                    path_i_to_save = \
                        "{0}/filters/{1}/{2}.png"\
                            .format(save_without_showing, report_id, self.frame_i)
                    cv2.imwrite(path_i_to_save, f)
                    path_i_to_save = \
                        "{0}/regions_with_bbox/{1}/{2}.png"\
                            .format(save_without_showing, report_id, self.frame_i)
                    cv2.imwrite(path_i_to_save, target_with_bbox)
                else:
                    # f = filter.sum(axis=2)
                    cv2.imshow("filter", f)
                    cv2.imshow("region_with_bbox", target_with_bbox)

            if save_without_showing:
                path_i_to_save = "{0}/frames_with_response/{1}/{2}.png"\
                    .format(save_without_showing, report_id, self.frame_i)
                cv2.imwrite(path_i_to_save, main_img)
            else:
                cv2.imshow('frame_with_response', main_img)

            cv2.waitKey(1)