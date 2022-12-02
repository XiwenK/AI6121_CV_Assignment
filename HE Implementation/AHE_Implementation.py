import math
import numpy as np

from PIL import Image
from HE_Implementation import histogram_equalization, rgb_to_hsv, hsv_to_rgb, image_compare


def adaptive_histogram_equalization(im, window_size=32, affect_size=16):
    n_im = np.zeros_like(im)

    m, n = im.shape
    rows = math.ceil((m - window_size) / affect_size) + 1
    cols = math.ceil((n - window_size) / affect_size) + 1
    offset = int((window_size - affect_size) / 2)

    for i in range(rows):
        for j in range(cols):
            # block segment
            affect_beg_i, affect_end_i = offset + i * affect_size, offset + (i + 1) * affect_size
            affect_beg_j, affect_end_j = offset + j * affect_size, offset + (j + 1) * affect_size

            window_beg_i, window_end_i = i * affect_size, window_size + i * affect_size
            window_beg_j, window_end_j = j * affect_size, window_size + j * affect_size

            window_arr = im[window_beg_i: window_end_i, window_beg_j: window_end_j]
            n_window_arr = histogram_equalization(window_arr)

            # boarder padding
            if i == 0:
                n_im[window_beg_i: affect_beg_i, window_beg_j: window_end_j] \
                    = n_window_arr[: affect_beg_i - window_beg_i, :]
            elif i == rows - 1:
                n_im[affect_end_i: window_end_i, window_beg_j: window_end_j] \
                    = n_window_arr[affect_end_i - window_beg_i: window_end_i - window_beg_i, :]

            if j == 0:
                n_im[window_beg_i: window_end_i, window_beg_j: affect_beg_j] \
                    = n_window_arr[:, : affect_beg_j - window_beg_j]
            elif j == cols - 1:
                n_im[window_beg_i: window_end_i, affect_end_j: window_end_j] \
                    = n_window_arr[:, affect_end_j - window_beg_j: window_end_j - window_beg_j]

            n_im[affect_beg_i: affect_end_i, affect_beg_j: affect_end_j] \
                = n_window_arr[affect_beg_i - window_beg_i: affect_end_i - window_beg_i,
                  affect_beg_j - window_beg_j: affect_end_j - window_beg_j]

    return n_im


# def execute_func_AHE(src_PATH, tgt_PATH):
#     im = Image.open(src_PATH)
#     im = np.array(im)
#     print(im.shape)
#
#     n_im = np.empty_like(im)
#     if len(im.shape) == 2:
#         n_im = adaptive_histogram_equalization(im)
#     elif len(im.shape) == 3:
#         imr = im[:, :, 0]
#         img = im[:, :, 1]
#         imb = im[:, :, 2]
#
#         n_imr = adaptive_histogram_equalization(imr)
#         n_img = adaptive_histogram_equalization(img)
#         n_imb = adaptive_histogram_equalization(imb)
#
#         m, n = imr.shape[0], imr.shape[1]
#         for i in range(m):
#             for j in range(n):
#                 if n_imr[i][j] > 255:
#                     n_imr[i][j] = 255
#                 if n_img[i][j] > 255:
#                     n_img[i][j] = 255
#                 if n_imb[i][j] > 255:
#                     n_imb[i][j] = 255
#
#         n_im[:, :, 0] = n_imr
#         n_im[:, :, 1] = n_img
#         n_im[:, :, 2] = n_imb
#
#     n_im = Image.fromarray(n_im)
#     image_compare(im, n_im, tgt_PATH)


def execute_func_AHE(src_PATH, tgt_PATH):
    im = Image.open(src_PATH)
    im = np.array(im)
    print(im.shape)

    n_im = np.empty_like(im)
    if len(im.shape) == 2:
        n_im = adaptive_histogram_equalization(im)
    elif len(im.shape) == 3:
        n_im = rgb_to_hsv(im)

        imv = n_im[:, :, 2]
        imv = adaptive_histogram_equalization(imv.astype('uint8'))
        m, n = imv.shape[0], imv.shape[1]
        for i in range(m):
            for j in range(n):
                if imv[i][j] > 255:
                    imv[i][j] = 255

        n_im[:, :, 2] = imv.astype('float64')
        n_im = hsv_to_rgb(n_im)

    n_im = Image.fromarray(n_im.astype('uint8'))
    image_compare(im, n_im, tgt_PATH)
