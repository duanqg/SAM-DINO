import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def create_coco_json(mask_path, categories, width, height):
    # 初始化COCO数据结构
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": categories
    }
    # 添加图像信息
    image_info = {
        "id": 1,  # 假设只有一个图像
        "file_name": mask_path.split('/')[-1],
        "width": width,
        "height": height
    }
    coco_output["images"].append(image_info)

    return coco_output

def add_coco_masks(coco_output, annotation_id, mask, value):
    annotation_infos = []

    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = (np.pad(mask, pad_width=1, mode='constant', constant_values=0) * 255).astype('uint8')
    contours, _ = cv2.findContours(padded_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours = np.subtract(contours, 0)

    for i, contour in enumerate(contours):
        if len(contour) < 3:  # filter unenclosed objects
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bounding_box = [x, y, w, h]
        seg_area = cv2.contourArea(contour)
        bbox_area = w * h

        if bbox_area < 4:  # filter small objects
            continue

        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]

        annotation_info = {
            "id": annotation_id,
            "image_id": 1,
            "category_id": value,
            "area": seg_area,  # it's float
            "bbox": bounding_box,
            "segmentation": [segmentation],
            "width": mask.shape[1],
            "height": mask.shape[0],
        }

        annotation_infos.append(annotation_info)
    # print(annotation_infos)
    coco_output["annotations"].extend(annotation_infos)
    return coco_output


def create_annotation_infos(annotation_id, image_id, category_info, binary_mask):
    is_crowd = category_info['is_crowd']
    annotation_infos = []

    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = (np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0) * 255).astype('uint8')
    contours, _ = cv2.findContours(padded_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = np.subtract(contours, 1)

    for i, contour in enumerate(contours):
        if len(contour) < 3:  # filter unenclosed objects
            continue

        x, y, w, h = cv2.boundingRect(contour)
        bounding_box = [x, y, w, h]
        seg_area = cv2.contourArea(contour)
        bbox_area = w * h

        if bbox_area < 4:  # filter small objects
            continue

        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]

        annotation_info = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_info["id"],
            "iscrowd": is_crowd,
            "area": seg_area,  # it's float
            "bbox": bounding_box,
            "segmentation": [segmentation],
            "width": binary_mask.shape[1],
            "height": binary_mask.shape[0],
        }

        annotation_id += 1

        annotation_infos.append(annotation_info)

    return annotation_infos, annotation_id


def json_convert(obj):
    """
    递归转换对象中的所有numpy int64对象为Python int类型。
    """
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, list):
        return [json_convert(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: json_convert(value) for key, value in obj.items()}
    else:
        return obj


def visualize_coco_image(image_path, annotation_path,show_segmentation=True):
    # 读取JSON文件
    with open(annotation_path) as f:
        data = json.load(f)

    # 读取图像
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 通过图像ID找到对应的标注
    for ann in data['annotations']:
        # 绘制边界框
        bbox = ann['bbox']
        bbox_coords = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        draw.rectangle(bbox_coords, outline='red', width=10)
        # print(dino_xxyy)

        # 如果需要，绘制分割掩码
        if show_segmentation and 'segmentation' in ann:
            for segment in ann['segmentation']:
                if isinstance(segment, list):  # 检查segment是否为多边形格式
                    draw.polygon(segment, outline='blue', width=10)

    # 显示图像
    plt.imshow(image)
    plt.axis('off')
    plt.show()