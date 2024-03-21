import cv2

import os
import glob

from script.output import output_json
from script.init_checkpoints import init_dino, init_sam_predictor
from script.sam_execute import segment_predictor, detection_boxes
from script.cocojson import visualize_coco_image

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
    # output_dir = "data/result/"
    # input_dir = './data/debug/'
    output_dir = "/980pro/results"
    input_dir = '/home/kemove/Videos/xinya_fly_video/xinya_simple_radial/images/'

    # output_dir ="/home/kemove/Documents/colmap_dataset/south-building/sparse/stereo/segmentation"
    # input_dir = '/home/kemove/Documents/colmap_dataset/south-building/sparse/images/'

    dino = init_dino()
    sam_predictor = init_sam_predictor()

    # Predict classes and hyper-param for GroundingDINO

    CLASSES = ["window","door","tree","stair","grass","air conditioner"]
    # 示例类别
    categories = [
        {"id": 0, "name": "window"},
        {"id": 1, "name": "door"},
        {"id": 2, "name": "tree"},
        {"id": 3, "name": "stair"},
        {"id": 4, "name": "grass"},
        {"id": 5, "name": "air conditioner"}
    ]
    image_type = '.jpg'
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
        image_mask = output_json(detections, output_dir, suffix, categories)

        # visualize_coco_image(image_path, os.path.join(output_dir, suffix + ".json"))
