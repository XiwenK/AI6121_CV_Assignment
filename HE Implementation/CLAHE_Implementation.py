import numpy as np

from PIL import Image

from HE_Implementation import rgb_to_hsv, hsv_to_rgb, calculate_hist, image_compare


def clip_histogram(hist, threshold):
    exceed_sum = sum([i - threshold for i in hist if i > threshold])
    exceed_mean = exceed_sum / 256

    clip_hist = np.zeros_like(hist)
    for i in range(256):
        if hist[i] <= threshold:
            clip_hist[i] = hist[i] + exceed_mean
        else:
            clip_hist[i] = threshold + exceed_mean

    return clip_hist


def contrast_limited_adaptive_histogram_equalization(im, block_num=8, threshold=10.0):
    m, n = im.shape
    block_m = int(m / block_num)
    block_n = int(n / block_num)

    hists = []
    for i in range(block_num):
        hists_row = []
        for j in range(block_num):
            beg_i, end_i = i * block_m, (i + 1) * block_m
            beg_j, end_j = j * block_n, (j + 1) * block_n

            block_im = im[beg_i: end_i, beg_j: end_j]

            hist = calculate_hist(block_im)
            clip_hist = clip_histogram(hist, threshold / 256)
            n_hist = np.ceil(np.cumsum(clip_hist) * 255)

            hists_row.append(n_hist)

        hists.append(hists_row)

    n_im = np.zeros_like(im)
    for i in range(m):
        for j in range(n):
            r = int((i - 0.5 * block_m) / block_m)
            c = int((j - 0.5 * block_n) / block_n)

            delta_x = (i - (r + 0.5) * block_m) / block_m
            delta_y = (j - (c + 0.5) * block_n) / block_n

            if r < 0 and c < 0:
                n_im[i][j] = hists[r + 1][c + 1][im[i][j]]
            elif r < 0 and c >= block_num - 1:
                n_im[i][j] = hists[r + 1][c][im[i][j]]
            elif r >= block_num - 1 and c < 0:
                n_im[i][j] = hists[r][c + 1][im[i][j]]
            elif r >= block_num - 1 and c >= block_num - 1:
                n_im[i][j] = hists[r][c][im[i][j]]
            elif r < 0 or r >= block_num - 1:
                if r < 0:
                    r = 0
                elif r > block_num - 1:
                    r = block_num - 1
                left = hists[r][c][im[i][j]]
                right = hists[r][c + 1][im[i][j]]
                n_im[i][j] = (1 - delta_y) * left + delta_y * right
            elif c < 0 or c >= block_num - 1:
                if c < 0:
                    c = 0
                elif c > block_num - 1:
                    c = block_num - 1
                up = hists[r][c][im[i][j]]
                down = hists[r + 1][c][im[i][j]]
                n_im[i][j] = (1 - delta_x) * up + delta_x * down
            else:
                lu = hists[r][c][im[i][j]]
                ld = hists[r + 1][c][im[i][j]]
                ru = hists[r][c + 1][im[i][j]]
                rd = hists[r + 1][c + 1][im[i][j]]
                n_im[i][j] = (1 - delta_y) * ((1 - delta_x) * lu + delta_x * ld) + delta_y * (
                        (1 - delta_x) * ru + delta_x * rd)

    n_im = n_im.astype('uint8')
    return n_im


def execute_func_CLAHE(src_PATH, tgt_PATH):
    im = Image.open(src_PATH)
    im = np.array(im)
    print(im.shape)

    n_im = np.empty_like(im)
    if len(im.shape) == 2:
        n_im = contrast_limited_adaptive_histogram_equalization(im)
    elif len(im.shape) == 3:
        imr = im[:, :, 0]
        img = im[:, :, 1]
        imb = im[:, :, 2]

        n_imr = contrast_limited_adaptive_histogram_equalization(imr)
        n_img = contrast_limited_adaptive_histogram_equalization(img)
        n_imb = contrast_limited_adaptive_histogram_equalization(imb)

        # m, n = imr.shape[0], imr.shape[1]
        # for i in range(m):
        #     for j in range(n):
        #         if n_imr[i][j] > 255:
        #             n_imr[i][j] = 255
        #         if n_img[i][j] > 255:
        #             n_img[i][j] = 255
        #         if n_imb[i][j] > 255:
        #             n_imb[i][j] = 255

        n_im[:, :, 0] = n_imr
        n_im[:, :, 1] = n_img
        n_im[:, :, 2] = n_imb

    n_im = Image.fromarray(n_im)
    image_compare(im, n_im, tgt_PATH)


# def execute_func_CLAHE(src_PATH, tgt_PATH):
#     im = Image.open(src_PATH)
#     im = np.array(im)
#     print(im.shape)
#
#     n_im = np.empty_like(im)
#     if len(im.shape) == 2:
#         n_im = contrast_limited_adaptive_histogram_equalization(im)
#     elif len(im.shape) == 3:
#         n_im = rgb_to_hsv(im)
#
#         imv = n_im[:, :, 2]
#         n_im[:, :, 2] = contrast_limited_adaptive_histogram_equalization(imv.astype('uint8')).astype('float64')
#         n_im = hsv_to_rgb(n_im).astype('uint8')
#
#     n_im = Image.fromarray(n_im)
#     image_compare(im, n_im, tgt_PATH)
