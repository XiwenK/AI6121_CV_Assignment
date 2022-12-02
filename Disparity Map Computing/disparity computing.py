import os
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np

# fx = 942.8  # lense focal length透镜焦距
# baseline = 54.8  # distance in mm between the two cameras两台摄像机之间的距离（mm）
# disparities = 128  # num of disparities to consider要考虑的差异数
# block = 31  # block size to match要匹配的块大小
# units = 0.512  # depth units, adjusted for the output to fit in one byte深度单位，调整为适合一个字节的输出
from PIL import Image
from skimage import filters


def sobel(img, threshold=0):
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    rows, cols = img.shape
    output = np.zeros(img.shape)

    for i in range(0, rows - 2):
        for j in range(0, cols - 2):
            v = sum(sum(G_x * img[i:i + 3, j:j + 3]))  # vertical
            h = sum(sum(G_y * img[i:i + 3, j:j + 3]))  # horizon
            output[i + 1, j + 1] = np.sqrt((v ** 2) + (h ** 2))

    for p in range(0, rows):
        for q in range(0, cols):
            if output[p, q] < threshold:
                output[p, q] = 0

    return output


def disparity_compute(left_im,  right_im, block, disparities):
    disparity_map = np.zeros(shape=left_im.shape)
    height, width = left_im.shape
    h_block = block // 2

    for i in range(0, height):
        for j in range(0, width):
            r_l, r_r = max(0, i - h_block), min(height - 1, i + h_block)
            c_l, c_r = max(0, j - h_block), min(width - 1, j + h_block)
            l = left_im[r_l: r_r, c_l: c_r]

            # calc SSD at all possible disparities
            h_dis = disparities // 2
            dis_l, dis_r = max(-h_dis, -c_l), min(h_dis, width - 1 - c_r)
            ssd = np.empty([dis_r - dis_l + 1, 1])

            for d in range(dis_l, dis_r + 1):
                r = right_im[r_l: r_r, c_l + d: c_r + d]
                ssd[d - dis_l] = np.sum((l[:, :] - r[:, :]) ** 2)

            # select the best match
            disparity_map[i, j] = np.argmin(ssd)

    return disparity_map

    # Convert disparity to depth
    # depth = np.zeros(shape=left_im.shape).astype(float)
    # depth[disparity_map > 0] = (fx * baseline) / (units * disparity_map[disparity_map > 0])


im_l = Image.open(os.getcwd() + '/src_image' + '/corridorl.jpg').convert('L')
im_r = Image.open(os.getcwd() + '/src_image' + '/corridorr.jpg').convert('L')
im_l = np.array(im_l)
im_r = np.array(im_r)
# im_l = sobel(im_l)
# im_r = sobel(im_r)

disparity_map = disparity_compute(im_l, im_r, block=5, disparities=23)

im_map = Image.fromarray(disparity_map)
plt.imshow(im_map)
plt.savefig(os.getcwd() + '/disparity_map' + '/disparity_map_corridor_small_search_space')

plt.show()
