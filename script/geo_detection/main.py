# -*- coding: utf-8 -*-
import os
import argparse
import pointcld2Img
import img2Pointcld
import ctypes


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-t", "--type", help="map_type", type=str, default='geometric',
    # )
    parser.add_argument(
        "-n", "--name", help="map_name", type=str, required=True
    )
    args = parser.parse_args()
    return args

def camera_info (depth_path):
    col = len(depth_path[0])
    row = len(depth_path)
    scale_x = 3094 / col
    scale_y = 2312 / row
    # print(len(depth[0]), len(depth))
    fx = 2559.729675 / scale_x
    fy = 2559.72967 / scale_y
    cx = 1547.000000 / scale_x
    cy = 1156.000000 / scale_y
    return col, row, fx, fy, cx, cy
def main():
    args = parse_args()

    depth_path = os.path.abspath(r'../../data/depth/' + args.name + '.JPG.geometric.bin')
    normal_path = os.path.abspath(r'../../data/normal/' + args.name + '.JPG.geometric.bin')
    pcd_path = os.path.abspath(r'../../data/result_pcd/' + args.name)
    output_img_path = os.path.abspath(r'../../data/geomask/' + args.name + '.png')
    exe = os.path.abspath(r'../../superVoxel/build/libsupervoxel.so')
    _depth_map = img2Pointcld.img2Pointcld.read_array(depth_path)

    col, row, fx, fy, cx, cy = camera_info(_depth_map)
    if not os.path.exists(depth_path):
        raise FileNotFoundError("File not found: {}".format(depth_path))

    if not os.path.exists(normal_path):
        raise FileNotFoundError("File not found: {}".format(normal_path))

    img2Pointcld.img2Pointcld.img_to_pointcld(depth_path, normal_path, pcd_path + '.ply')
    superVoxel = ctypes.cdll.LoadLibrary(exe)
    superVoxel.superVoxel(ctypes.c_char_p(pcd_path.encode('utf-8')), 16, ctypes.c_double(0.1))
    pointcld2Img.pointcld2Img.point_cloud_image(pcd_path + ".xyz", output_img_path, col, row, fx, fy, cx, cy)

if __name__ == "__main__":
    main()
