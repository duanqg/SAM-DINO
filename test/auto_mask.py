import cv2

import os
import glob
from script.init_checkpoints import init_sam_auto_mask_generator
from script.sam_execute import segment_auto_mask
from script.output import auto_sam_mask


if __name__ == "__main__":
    # parser = argparse.ArgumentParser("Segment Building Struct", add_help=True)
    #
    # parser.add_argument(
    #     "--output_dir", "-o", type=str, default="output", required=True, help="output directory"
    # )
    # parser.add_argument(
    #     "--input_dir", "-i", type=str, default="input", required=True, help="input directory"
    # )
    # args = parser.parse_args()
    #
    # # cfg
    # output_dir = args.output_dir
    # input_dir = args.input_dir

    output_dir = "./data/result_test_point/"
    input_dir = '../data/debug/'

    sam_auto_mask_generator = init_sam_auto_mask_generator()

    image_type = '.JPG'
    images_path = glob.glob(os.path.join(input_dir + '*' + image_type))  # 所有图片路径
    for image_path in images_path:
        image_name = os.path.basename(image_path)
        suffix = image_name.split(".")[0]
        image = cv2.imread(image_path)
        print(image.shape)
        image_mask = segment_auto_mask(
            sam_auto_mask_generator=sam_auto_mask_generator,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )
        auto_sam_mask(image_mask, output_dir, suffix)

