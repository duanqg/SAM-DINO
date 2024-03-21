import cv2
import numpy as np
import torch

from torch.nn import functional as F
import os
import glob

from typing import Tuple
from script.sliding_window.region_growing import region_growing
from script.output import normal_sam_mask
from script.init_checkpoints import init_sam_predictor
from script.sam_execute import rg_segment_predictor

def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
    """
    Compute the output size given input size and target long side length.
    """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def rg(H,W, normal_file, seeds):
    normal_file = '/home/kemove/Documents/sam-dino/data/normal/' + normal_file + '.JPG.geometric.bin'

    seeds = seeds
    mask = region_growing(H, W, normal_file, seeds)
    prompt = mask.astype(np.float32)

    # prompt = np.transpose(prompt, (2, 0, 1))
    target_size = get_preprocess_shape(H, W, 256)
    prompt = torch.from_numpy(prompt)
    prompt = torch.nn.functional.interpolate(
        prompt.float().unsqueeze(0).permute([0,1,2,3]),
        target_size,
        mode='bilinear')

    h,w = prompt.shape[-2:]
    print("interpolate: ", prompt.shape)
    padh = 256 - h
    padw = 256 - w
    prompt = F.pad(prompt, (0, padw, 0, padh))
    prompt = (prompt / 255) * 40 - 20
    prompt = prompt.squeeze(1).cpu().numpy()
    print(prompt.shape)
    return prompt

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
    sam_predictor = init_sam_predictor()
    image_type = '.JPG'
    seed = [(1200, 500)]

    images_path = glob.glob(os.path.join(input_dir + '*' + image_type))  # 所有图片路径
    for image_path in images_path:
        image_name = os.path.basename(image_path)
        suffix = image_name.split(".")[0]
        image = cv2.imread(image_path)
        geometry_mask = rg(image.shape[0], image.shape[1], suffix, seeds=seed)
        mask, scores = rg_segment_predictor(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            seed=seed,
            mask_input=geometry_mask
        )
        normal_sam_mask(mask, output_dir, suffix)
