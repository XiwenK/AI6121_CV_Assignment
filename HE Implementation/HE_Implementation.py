import colorsys

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.color import rgb2hsv


def calculate_hist(arr):
    arr = arr.flatten()
    hist = np.zeros(256, dtype=float)

    for i in arr:
        hist[i] += 1

    return hist / len(arr)


def histogram_equalization(im):
    hist = calculate_hist(im)
    # plt.hist(im.flatten(), 256, [0, 256], density=True, stacked=True)
    # plt.show()

    n_hist = np.ceil(np.cumsum(hist) * 255)

    n_im = np.zeros(im.shape)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            n_im[i][j] = n_hist[im[i][j]]

    # plt.hist(n_im.flatten(), 256, [0, 256], density=True, stacked=True)
    # plt.show()

    return n_im


def image_compare(img1, img2, tgt_PATH):
    plt.figure()
    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.savefig(tgt_PATH)
    plt.show()


def rgb_to_hsv(im):
    dim = im.shape
    n_im = np.zeros(im.shape)
    for i in range(dim[0]):
        for j in range(dim[1]):
            n_im[i][j][0], n_im[i][j][1], n_im[i][j][2] \
                = colorsys.rgb_to_hsv(im[i][j][0], im[i][j][1], im[i][j][2])
    return n_im


def hsv_to_rgb(im):
    dim = im.shape
    n_im = np.zeros(im.shape)
    for i in range(dim[0]):
        for j in range(dim[1]):
            n_im[i][j][0], n_im[i][j][1], n_im[i][j][2] \
                = colorsys.hsv_to_rgb(im[i][j][0], im[i][j][1], im[i][j][2])
    return n_im


def execute_func_HE(src_PATH, tgt_PATH):
    im = Image.open(src_PATH)
    im = np.array(im)
    print(im.shape)

    n_im = np.empty_like(im)
    if len(im.shape) == 2:
        n_im = histogram_equalization(im)
    elif len(im.shape) == 3:
        imr = im[:, :, 0]
        img = im[:, :, 1]
        imb = im[:, :, 2]

        n_imr = histogram_equalization(imr)
        n_img = histogram_equalization(img)
        n_imb = histogram_equalization(imb)

        m, n = imr.shape[0], imr.shape[1]
        for i in range(m):
            for j in range(n):
                if n_imr[i][j] > 255:
                    n_imr[i][j] = 255
                if n_img[i][j] > 255:
                    n_img[i][j] = 255
                if n_imb[i][j] > 255:
                    n_imb[i][j] = 255

        n_im[:, :, 0] = n_imr
        n_im[:, :, 1] = n_img
        n_im[:, :, 2] = n_imb

    n_im = Image.fromarray(n_im)
    image_compare(im, n_im, tgt_PATH)

# def execute_func_HE(src_PATH, tgt_PATH):
#     im = Image.open(src_PATH)
#     im = np.array(im)
#     print(im.shape)
#
#     n_im = np.empty_like(im)
#     if len(im.shape) == 2:
#         n_im = histogram_equalization(im)
#     elif len(im.shape) == 3:
#         n_im = rgb_to_hsv(im)
#
#         imv = n_im[:, :, 2]

#         imv = histogram_equalization(imv.astype('uint8'))
#         m, n = imv.shape[0], imv.shape[1]
#         for i in range(m):
#             for j in range(n):
#                 if imv[i][j] > 255:
#                     imv[i][j] = 255

#         n_im[:, :, 2] = imv.astype('float64')
#         n_im = hsv_to_rgb(n_im)
#
#     n_im = Image.fromarray(n_im.astype('uint8'))
#     image_compare(im, n_im, tgt_PATH)


# file = 'sample02.jpeg'
# execute_func_HE('source pictures/' + file, 'HE_rgb_target pictures/' + file)
