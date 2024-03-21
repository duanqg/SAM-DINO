import numpy as np
import cv2
import os
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
def is_similar(normal1, normal2, threshold = 4):
    """Check if two normals are similar based on a threshold."""
    # 定义两个向量

    # 计算点积
    dot_product = np.dot(normal1, normal2)
    # # 计算向量的模
    norm_a = np.linalg.norm(normal1)
    norm_b = np.linalg.norm(normal2)
    angle_rad = np.arccos(dot_product / (norm_a * norm_b))
    # # 转换为度
    angle_deg = np.degrees(angle_rad)
    return angle_deg < threshold

def seed_region_growing(normal_map, seeds):
    height, width, _ = normal_map.shape
    label_matrix = np.zeros((height, width), dtype=np.int32)
    label = 1

    for seed in seeds:
        points_to_check = [seed]

        while points_to_check:
            point = points_to_check.pop(0)
            x, y = point

            if label_matrix[y, x] != 0:
                continue

            label_matrix[y, x] = label

            # Check neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and label_matrix[ny, nx] == 0:
                        if is_similar(normal_map[ny, nx], normal_map[y, x]):
                            points_to_check.append((nx, ny))

        label += 1
    return label_matrix

def region_growing(h, w, file, seeds):
    normal_map = read_array(file)
    segmented = seed_region_growing(normal_map, seeds)
    segmented_normalized = (segmented / segmented.max()) * 255
    segmented_normalized = segmented_normalized.astype(np.uint8)
    new_size = (w, h)
    # Resize the image using bilinear interpolation
    upscaled_image = cv2.resize(segmented_normalized, new_size, interpolation=cv2.INTER_LINEAR)
    upscaled_image = np.expand_dims(upscaled_image, axis=0)
    # print(file)
    # cv2.imshow('Original Image', normal_map)
    # cv2.imshow('Segmented Image', segmented_normalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return upscaled_image

def region_growing_cut(h, w, normal_map, seeds):
    segmented = seed_region_growing(normal_map, seeds)
    segmented_normalized = (segmented / segmented.max()) * 255
    segmented_normalized = segmented_normalized.astype(np.uint8)
    # new_size = (w, h)
    # Resize the image using bilinear interpolation
    # upscaled_image = cv2.resize(segmented_normalized, new_size, interpolation=cv2.INTER_LINEAR)

    return np.expand_dims(segmented_normalized, axis=0)