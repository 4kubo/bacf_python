from image_process import features_pedro_py as pyhog
import numpy as np
from PIL import Image
import cv2


def get_pixels(img, pos, n_scales, search_pix_sz, target_sz=None, mode="edge"):
    """
    Crop pixels from input image `img` with the center of `pos` and resize to `target_sz`
    Args:
        img:
        pos:
        search_pix_sz:
        target_sz:
        mode: A type of padding.
    """
    search_pix_sz[search_pix_sz < 1] = 2
    # About coordination of top left side
    yx0 = (pos[None, :] - (search_pix_sz / 2.)).astype(int)
    m_yx0 = - np.minimum(0, yx0)
    yx0 = np.maximum(0, yx0)
    # About coordination of bottom right side
    yx1 = (pos[None, :] + (search_pix_sz / 2.)).astype(int)
    yx1_max = np.array([img.shape[:2]] * n_scales) - 1
    m_yx1 = np.maximum(0, yx1 - yx1_max)
    yx1 = np.minimum(yx1, yx1_max)
    # Pad cropped pixels
    padded_pixs = [np.pad(img[y0: y1, x0: x1, :], ((m_y0, m_y1), (m_x0, m_x1), (0, 0)), mode)
                   for (y0, x0), (y1, x1), (m_y0, m_x0), (m_y1, m_x1) in zip(yx0, yx1, m_yx0, m_yx1)]
    cropped_pixs = [cv2.resize(padded_pix, tuple(target_sz.astype(int))) for padded_pix in padded_pixs]
    cropped_pixs = np.stack(cropped_pixs, axis=-1)
    return cropped_pixs

def get_pixel(img, pos, search_pix_sz, target_sz=None, mode="edge"):
    """
    Crop pixel from input image `img` with the center of `pos` and resize to `target_sz`
    Args:
        img:
        pos:
        search_pix_sz:
        target_sz:
        mode: A type of padding.
    """
    # About coordination of top left side
    yx0 = (pos - (search_pix_sz / 2.)).astype(int)
    m_yx0 = - np.minimum(0, yx0)
    yx0 = np.maximum(0, yx0)
    # About coordination of bottom right side
    yx1 = (pos + (search_pix_sz / 2.)).astype(int)
    yx1_max = np.array(img.shape[:2]) - 1
    m_yx1 = np.maximum(0, yx1 - yx1_max)
    yx1 = np.minimum(yx1, yx1_max)

    y0, x0 = yx0
    y1, x1 = yx1
    m_y0, m_x0 = m_yx0
    m_y1, m_x1 = m_yx1
    # Pad cropped pixels
    cropped_pix = np.pad(img[y0: y1, x0: x1, :], ((m_y0, m_y1), (m_x0, m_x1), (0, 0)), mode)
    if target_sz is not None:
        cropped_pix = cv2.resize(cropped_pix, tuple(target_sz.astype(int)))
    return cropped_pix


def get_pyhog(image, cell_size):
    """
    This function is borrowed from dimatura's implementation of HoG
    reference : https://github.com/dimatura/pyhog
    """
    image = image.astype(np.float64)/255.0
    image_c = image.copy("F")
    hog = pyhog.process(image_c, cell_size)
    return hog