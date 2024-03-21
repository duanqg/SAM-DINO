import os
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from script.cocojson import create_coco_json, add_coco_masks, json_convert
# bbox是否太大.如果占比80%,要舍去


def is_bbox_big(box, img):
    scale_h = (box[3] - box[1]) /img.shape[0]
    scale_w = (box[2] - box[0]) /img.shape[1]

    if((scale_w > 0.75) & (scale_h >0.75)):
        return 0
    else:
        return 1



# 计算交并比IOU
def iou_sam_dino(mask, box2):
    np_seg = np.array(mask)
    seg_value = 1
    segmentation = np.where(np_seg == seg_value)
    # Bounding Box
    if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
        x_min = np.min(segmentation[1])
        x_max = np.max(segmentation[1])
        y_min = np.min(segmentation[0])
        y_max = np.max(segmentation[0])
        box1 = np.array([x_min, y_min, x_max, y_max])

        bxmin = max(box1[0], box2[0])
        bymin = max(box1[1], box2[1])
        bxmax = min(box1[2], box2[2])
        bymax = min(box1[3], box2[3])
        bwidth = bxmax - bxmin
        bhight = bymax - bymin
        inter = bwidth * bhight
        union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
        return inter / union

def dino_sam_mask(detection, output_dir, suffix):
    if len(detection.mask) == 0:
        return
    i = 0
    full_img = None
    color = np.array([1.0,1.0,1.0])
    for id in detection.class_id:
        m = detection.mask[i]
        iou = iou_sam_dino(m, detection.xyxy[i])
        ibg = is_bbox_big(detection.xyxy[i], m)
        if id == 0:
            color = np.array([1.0, 1.0, 1.0])
        elif id == 1:
            color = np.array([0.9, 0.9, 0.9])
        elif id == 2:
            color = np.array([0.8, 0.8, 0.8])
        elif id == 3:
            color = np.array([0.7, 0.7, 0.7])
        elif id == 4:
            color = np.array([0.6, 0.6, 0.6])
        elif id == 5:
            color = np.array([0.5, 0.5, 0.5])
        else:
            color = np.array([0.4, 0.4, 0.4])

        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
            map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
        if ((iou > 0.3) & (ibg == 1)):
            full_img[m != 0] = color
        else:
            if(iou <= 0.3) : print("舍弃iou过小的值,iou = ", iou)
            else: print("舍弃范围过大的bbox,该bbox与图片的长宽比都大于0.75")
        i = i + 1
        # color_mask = np.random.random((1, 3)).tolist()[0]
    full_img = full_img * 255
    # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], map.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    full_img = Image.fromarray(np.uint8(full_img))
    full_img.save(os.path.join(output_dir, suffix + ".jpg"))


def normal_sam_mask(anns, output_dir, suffix):
    full_img = None

    # for ann in sorted_anns:
    # for i in range(len(anns)):
        # ann = anns[i]
    m = anns[0]
    if full_img is None:
        full_img = np.zeros((m.shape[0], m.shape[1], 3))
        map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
    #
    color_mask = np.random.random((1, 3)).tolist()[0]

    # color_mask = np.array([1.0, 1.0, 1.0])
    full_img[m != 0] = color_mask
    full_img = full_img* 255
    # # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], m.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    full_img = Image.fromarray(np.uint8(full_img))
    full_img.save(os.path.join(output_dir, suffix + ".jpg"))

def auto_sam_mask(anns, output_dir, suffix):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    full_img = None

    # for ann in sorted_anns:
    for i in range(len(sorted_anns)):
        ann = anns[i]
        m = ann['segmentation']
        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
            map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
        map[m != 0] = i + 1
        color_mask = np.random.random((1, 3)).tolist()[0]
        full_img[m != 0] = color_mask
    full_img = full_img*255
    # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], map.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    full_img = Image.fromarray(np.uint8(full_img))

    full_img.save(os.path.join(output_dir, suffix + ".jpg"))
    return full_img, res



def output_json(detection, output_dir, suffix, categories):
    if len(detection.mask) == 0:
        return
    i = 0
    full_img = None
    coco_json = create_coco_json(os.path.join(output_dir, suffix + ".jpg"), categories, detection.mask[0].shape[1],detection.mask[0].shape[0])
    color = np.array([1.0,1.0,1.0])
    for id in detection.class_id:
        m = detection.mask[i]
        iou = iou_sam_dino(m, detection.xyxy[i])
        ibg = is_bbox_big(detection.xyxy[i], m)
        if id == 0:
            color = np.array([1.0, 1.0, 1.0])
        elif id == 1:
            color = np.array([0.9, 0.9, 0.9])
        elif id == 2:
            color = np.array([0.8, 0.8, 0.8])
        elif id == 3:
            color = np.array([0.7, 0.7, 0.7])
        elif id == 4:
            color = np.array([0.6, 0.6, 0.6])
        elif id == 5:
            color = np.array([0.5, 0.5, 0.5])
        else:
            color = np.array([0.4, 0.4, 0.4])

        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
            map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
        if ((iou > 0.3) & (ibg == 1)):
            full_img[m != 0] = color
            coco_json = add_coco_masks(coco_json, i, m, id)
        else:
            if(iou <= 0.3) : print("舍弃iou过小的值,iou = ", iou)
            else: print("舍弃范围过大的bbox,该bbox与图片的长宽比都大于0.75")
        i = i + 1
    full_img = full_img * 255
    # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], map.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    full_img = Image.fromarray(np.uint8(full_img))
    full_img.save(os.path.join(output_dir, suffix + ".jpg"))
    with open(os.path.join(output_dir, suffix + ".json"), 'w') as f:
        coco_output = json_convert(coco_json)
        json.dump(coco_output, f)