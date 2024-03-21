
import numpy as np
from PIL import Image
class pointcld2Img:
    def read_array_xyz(path, col, row, fx, fy, cx, cy):
        uv = np.zeros((row, col, 3))
        with open(path, "r") as fid:
            array = np.genfromtxt(
                fid, delimiter=None, usecols=(0, 1, 2, 3, 4, 5)
            )
        for a in array:
            xw = a[0]
            yw = a[1]
            zw = a[2]
            r = a[3]
            g = a[4]
            b = a[5]
            v = int((yw * fy / zw) + cy)
            u = int((xw * fx / zw) + cx)

            uv[v, u] = [r, g, b]
            # if(u <= len(depth[0]) & v <= len(depth)):
            #     # print(u,v)
            #     uv[v, u] = [r, g, b]
        return uv

    def point_cloud_image(pcd_path, output_img_path, col, row, fx, fy, cx, cy):
        pcl_map = pointcld2Img.read_array_xyz(pcd_path, col, row, fx, fy, cx, cy)
        im = Image.fromarray(np.uint8(pcl_map))
        im.save(output_img_path)