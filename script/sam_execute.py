
import torch
import torchvision
import numpy as np
from segment_anything import SamPredictor, SamAutomaticMaskGenerator
import random, math
def rg_segment_predictor(sam_predictor: SamPredictor , image: np.ndarray, seed: np.ndarray,
            mask_input: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    masks, scores, logits, attentions, upscaled_embedding = sam_predictor.predict(
        point_coords=np.array(seed),
        point_labels= np.array([1]),
        mask_input= mask_input,
        multimask_output=False)
    return masks, scores

def point_segment_predictor(sam_predictor: SamPredictor, image: np.ndarray, seed: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    masks, scores, logits, attentions, upscaled_embedding = sam_predictor.predict(
        point_coords=np.array(seed),
        point_labels= np.array([1]),
        multimask_output=False)
    return masks, scores


import random

def is_point_in_box(point, box):
    """ 检查点是否在边界框内 """
    x, y = point
    xmin, ymin, xmax, ymax = box
    return xmin <= x <= xmax and ymin <= y <= ymax

def generate_random_point_in_B_not_in_A(box_A, box_B):
    """
    在边界框B内生成一个随机点P，该点不在边界框A内
    """
    while True:
        # 在边界框B内生成随机点
        x = random.uniform(box_B[0], box_B[2])
        y = random.uniform(box_B[1], box_B[3])

        point = (math.floor(x), math.floor(y))

        # 检查点是否在边界框A内
        if not is_point_in_box(point, box_A):
            return point



def segment_auto_mask(sam_auto_mask_generator: SamAutomaticMaskGenerator, image: np.ndarray) -> np.ndarray:
    mask, scores, logits, attentions, upscaled_embedding = sam_auto_mask_generator.generate(image)
    return mask

def segment_predictor(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray, label: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    result_score = []
    i = 0
    # bbox计算优势：计算速度快
    # bbox计算劣势：不能顺应角度，容易多选
    # seg anything的劣势，没有鲁棒性
    # 优化：bbox框为2D输入，不能表示3D，应使用bbox中心点，做注意力的方式优化
    # 该优化方法问题：bbox中心点，容易误选择多选择
    # bbox与SAM结果的计算相交区域面积与两边占比
    for box in xyxy:
        # 示例
        box_A = (box[0], box[1], box[2], box[3])  # dino边界框A
        box_B = (1, 1, image.shape[0]-1, image.shape[1]-1)  # 边界框B
        point_prompts = []
        point_prompts.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        # 生成随机点
        for i in range(0, 4):
            random_point = generate_random_point_in_B_not_in_A(box_A, box_B)
            point_prompts.append([random_point[0], random_point[1]])
        # if(label[i] == 0):
        #     masks, scores, logits, attentions, upscaled_embedding = sam_predictor.predict(
        #         point_coords=np.array(point_prompts),
        #         # point_labels= np.array([1, 0, 0, 0, 0]),
        #         point_labels= np.array([1]),
        #         multimask_output=False)
        # if(label[i] == 0):
        #     masks, scores, logits, attentions, upscaled_embedding = sam_predictor.predict(
        #         point_coords=np.array([[(box[0]+box[2])/ 2,(box[1]+box[3]) / 2]]),
        #         box=box,
        #         point_labels= np.array([1]),
        #         multimask_output=False)
        # else:
        masks, scores, logits, attentions, upscaled_embedding = sam_predictor.predict(box=box,multimask_output=True)
        index = np.argmax(scores)
        result_score.append(scores[index])
        #记录id的位置在这里
        result_masks.append(masks[index])
        i += 1

    return result_masks, result_score

def detection_boxes(dino, input_image, classes):
    detection = dino.predict_with_classes(
        image=input_image,
        classes=classes,
        box_threshold=0.3,
        text_threshold=0.25
    )
    # nms：
    # 1.将所有的boxes按照置信度从小到大排序，然后从boxes中删除置信度最大的box
    # 2.将剩下的boxes与置信度最大的box，分别计算iou，去掉iou大于阈值(iou_threshold)的boxes
    # 重复1，2;直到索引为空

    # NMS post process
    print(f"Before NMS: {len(detection.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detection.xyxy),
        torch.from_numpy(detection.confidence),
        0.8
    ).numpy().tolist()
    detection.xyxy = detection.xyxy[nms_idx]
    detection.confidence = detection.confidence[nms_idx]
    detection.class_id = detection.class_id[nms_idx]

    print(f"After NMS: {len(detection.xyxy)} boxes")
    return detection