#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pointCloud

class img2Pointcld:
    def read_array(path):
        with open(path, "rb") as fid:
            width, height, channels = np.genfromtxt(
                fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int
            )
            fid.seek(0)
            num_delimiter = 0
            byte = fid.read(1)
            while True:
                if byte == b"&":
                    num_delimiter += 1
                    if num_delimiter >= 3:
                        break
                byte = fid.read(1)
            array = np.fromfile(fid, np.float32)
        array = array.reshape((width, height, channels), order="F")
        return np.transpose(array, (1, 0, 2)).squeeze()

    def depth_img_to_pointcld(depth_map, normal_map, min_depth, max_depth):
        scale_x = 3094 / len(depth_map[0])
        scale_y = 2312 / len(depth_map)
        # print(len(depth[0]), len(depth))
        fx = 2559.729675 / scale_x
        fy = 2559.72967 / scale_y
        cx = 1547.000000 / scale_x
        cy = 1156.000000 / scale_y
        points = []
        colors = []
        normals = []
        uvl = []
        for v in range(0, depth_map.shape[0]):
            for u in range(0, depth_map.shape[1]):
                zw = depth_map[v, u]
                # 去掉奇异值
                if (zw / max_depth < 1.5) & (zw > min_depth):
                    xw = (u - cx) * zw / fx
                    yw = (v - cy) * zw / fy
                    r = 255
                    g = 0
                    b = 255
                    point = [xw, yw, zw]
                    rgb = [r, g, b]
                    n = normal_map[v][u]
                    _uvl = [u, v, 0]
                    points.append(point)
                    normals.append(n)
                    colors.append(rgb)
                    uvl.append(_uvl)
        points=np.array(points)
        normals=np.array(normals)
        colors=np.array(colors)
        pcd = o3d.geometry.PointCloud()  # pcd类型的数据。
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # pcd.uvl = o3d.utility.Vector3dVector(uvl)
        return pcd

    def img_to_pointcld(depth_path, normal_path, output_path):
        depth_map = img2Pointcld.read_array(depth_path)
        normal_map = img2Pointcld.read_array(normal_path)
        min_depth, max_depth = np.percentile(depth_map, [5, 90])

        pcd = img2Pointcld.depth_img_to_pointcld(depth_map, normal_map, min_depth, max_depth)
        o3d.io.write_point_cloud(output_path, pcd, write_ascii=True)

        pointCloud.pointCloud.ply_double2float(output_path)
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)