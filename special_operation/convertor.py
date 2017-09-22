import numpy as np


def resize_DFT2(dft_input, size_desired):
    imh, imw, n1, n2 = dft_input.shape
    size_image = np.array((imh, imw))
    
    if np.any(size_desired != size_image):
        size_min = np.min(size_image, size_desired)
    
        scaling = np.prod(size_desired) / np.prod(size_image)

        dft_resized = np.zeroes([size_desired, n1, n2], dtype=np.complex128)

        mids = np.ceil(size_min / 2)
        mide = np.floor((size_min - 1) / 2) - 1

        dft_resized[:mids[0], np.arange(mids[1]),: ,:] =\
            scaling * dft_input[np.arange(mids[0]), np.arange(mids[1]),:,:]
        dft_resized[:mids[0], -1  - mide[1]:-1 , :, :] =\
                scaling * dft_input[:mids[0], -1 - mide[1]:-1, :, :]
        dft_resized[-1 - mide[0]:-1, :mids[1], :, :] =\
                scaling * dft_input[-1 - mide[0]:-1, :mids[1], :, :]
        dft_resized[-1 - mide[0]:-1, -1 - mide[1]:-1, :, :] =\
                scaling * dft_input[-1 - mide[0]:-1, -1 - mide[1]:-1, :, :]
    else:
        dft_resized = dft_input
    dft_resized = np.squeeze(dft_resized, 3)
    return dft_resized