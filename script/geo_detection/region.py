# -*- coding: utf-8 -*-
import os
import argparse
import pointcld2Img
import img2Pointcld
import ctypes
import cv2
import numpy as np

def region_growing():
    print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--type", help="map_type", type=str, default='geometric',
    )
    parser.add_argument(
        "-n", "--name", help="map_name", type=str, default='P1180160',
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    normal_path = os.path.abspath(r'../data/normal/' + args.name + '.JPG.geometric.bin')
    normal_map = img2Pointcld.img2Pointcld.read_array(normal_path)

    normal_value = normal_map.astype(np.float32) / 255.0
    # 显示结果
    cv2.imshow('Original Image', normal_map)
    # cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
