import cv2

import os
import glob

from script.output import dino_sam_mask
from script.init_checkpoints import init_dino, init_sam_predictor
from script.sam_execute import segment_predictor, detection_boxes


if __name__ == "__main__":
    # parser = argparse.ArgumentParser("Segment Building Struct", add_help=True)

    # parser.add_argument(
    #     "--output_dir", "-o", type=str, default="output", required=True, help="output directory"
    # )
    # parser.add_argument(
    #     "--input_dir", "-i", type=str, default="input", required=True, help="input directory"
    # )

    # args = parser.parse_args()

    # cfg
    output_dir = "./data/result_test_point/"
    input_dir = './data/debug/'

    dino = init_dino()
    sam_predictor = init_sam_predictor()

    # Predict classes and hyper-param for GroundingDINO
    CLASSES = ["window","wall","door","car","handrail","tree","grass","plant"]
    # CLASSES = ["window"]
    image_type = '.JPG'
    images_path = glob.glob(os.path.join(input_dir + '*' + image_type))  # 所有图片路径
    for image_path in images_path:
        image_name = os.path.basename(image_path)
        suffix = image_name.split(".")[0]
        image = cv2.imread(image_path)
        detections = detection_boxes(dino, image, CLASSES)
        detections.mask, detections.scores = segment_predictor(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
            label=detections.class_id
        )
        image_mask = dino_sam_mask(detections, output_dir, suffix)
