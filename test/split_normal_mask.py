import cv2
import numpy as np
import torch

from torch.nn import functional as F
import os
import glob

from typing import Tuple
from script.sliding_window.region_growing import region_growing_cut, read_array
from script.output import normal_sam_mask
from script.init_checkpoints import init_sam_predictor, delete_all_files_in_folder
from script.sam_execute import rg_segment_predictor, point_segment_predictor
from script.sliding_window.split_patch import split_image

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def rg(H,W, normal_map, seeds):
    mask = region_growing_cut(H, W, normal_map, seeds)
    prompt = mask.astype(np.float32)

    target_size = get_preprocess_shape(H, W, 256)
    prompt = torch.from_numpy(prompt)
    prompt = torch.nn.functional.interpolate(
        prompt.float().unsqueeze(0).permute([0,1,2,3]),
        target_size,
        mode='bilinear')

    h,w = prompt.shape[-2:]
    padh = 256 - h
    padw = 256 - w
    prompt = F.pad(prompt, (0, padw, 0, padh))
    prompt = (prompt / 255) * 40 - 20
    prompt = prompt.squeeze(1).cpu().numpy()
    return prompt

def split_image_depth(image_file, depth_file):
    rows, cols = 5, 6
    normal_map = read_array(depth_file) # 读取深度图
    w,h = normal_map.shape[:2]
    rgb_image = cv2.imread(image_file)
    rgb_image = cv2.resize(rgb_image, (h, w), interpolation=cv2.INTER_LINEAR)
    print(rgb_image.shape[:2])
    normal_blocks = split_image(normal_map, rows, cols) # 切分深度图
    rgb_blocks = split_image(rgb_image, rows, cols)

    return rgb_blocks, normal_blocks

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
    output_dir = "../data/result/"
    input_dir = '../data/debug/'
    input_dir_normal = '../data/normal/'
    sam_predictor = init_sam_predictor()
    image_type = '.JPG'
    seed = [(120, 120)]
    # 清空文件夹
    delete_all_files_in_folder(output_dir)

    images_path = glob.glob(os.path.join(input_dir + '*' + image_type))  # 所有图片路径
    for image_path in images_path:
        image_name = os.path.basename(image_path)
        suffix = image_name.split(".")[0]
        img_blocks, normal_blocks = split_image_depth(image_path, input_dir_normal + suffix + '.JPG.geometric.bin')
        image = cv2.imread(image_path)
        os.mkdir(output_dir + suffix)

        # 保存或处理每个图像块
        for i, block in enumerate(img_blocks):
            enable = True
            img_height, img_width = block.shape[:2]
            print(block.shape, normal_blocks[i].shape)
            if img_height < 120 or img_width < 120:
                enable = False
            if(enable != False):
                geometry_mask = rg(block.shape[0], block.shape[1], normal_blocks[i], seeds=seed)
                print(geometry_mask.shape, block.shape, normal_blocks[i].shape)
                sn_mask, sn_scores = rg_segment_predictor(
                    sam_predictor=sam_predictor,
                    image=cv2.cvtColor(block, cv2.COLOR_BGR2RGB),
                    seed=seed,
                    mask_input=geometry_mask
                )
                s_mask, s_scores = point_segment_predictor(
                    sam_predictor=sam_predictor,
                    image=cv2.cvtColor(block, cv2.COLOR_BGR2RGB),
                    seed=seed
                )
                normal_sam_mask(sn_mask, output_dir, suffix+ f'/block_{i}_normal')
                normal_sam_mask(s_mask, output_dir, suffix + f'/block_{i}_rgb')
                cv2.imwrite(output_dir + suffix + f'/block_{i}.jpg', block)