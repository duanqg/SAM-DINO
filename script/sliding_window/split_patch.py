

import cv2

def split_image(img, rows, cols):
    """
    将图像切分为指定数量的行和列。

    :param img: 要切分的原始图像
    :param rows: 切分的行数
    :param cols: 切分的列数
    :return: 切分后的图像块列表
    """
    # 获取原始图像的尺寸
    height, width = img.shape[:2]

    # 计算每个切分块的尺寸
    block_width, block_height = width // cols, height // rows

    # 切分图像
    blocks = []
    for y in range(0, height, block_height):
        for x in range(0, width, block_width):
            block = img[y:y+block_height, x:x+block_width]
            blocks.append(block)

    return blocks

import numpy as np
import cv2

def split_image_np_into_blocks(image, block_size):
    """
    将图像（NumPy 数组）切分成指定大小的块。

    :param image: 要切分的 NumPy 数组表示的图像
    :param block_size: 每个块的大小 (height, width)
    :return: 切分后的图像块列表
    """
    # 获取图像的高度和宽度
    img_height, img_width = image.shape[:2]
    m, n = block_size

    # 确保块大小不大于图像尺寸
    if m > img_height or n > img_width:
        raise ValueError("块大小不能大于图像大小")

    # 切分图像
    blocks = [image[x:x+m, y:y+n] for x in range(0, img_height, m) for y in range(0, img_width, n)]

    return blocks
