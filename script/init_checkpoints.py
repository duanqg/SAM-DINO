import torch
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


DEVICE = torch.device('cuda')  # 也可以使用cuda

# GroundingDINO config and checkpoint
# Segment-Anything checkpoint
# GROUNDING_DINO_CONFIG_PATH = "/home/kemove/Documents/sam-dino/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# GROUNDING_DINO_CHECKPOINT_PATH = "/home/kemove/Documents/sam-dino/model/groundingdino_swint_ogc.pth"
GROUNDING_DINO_CONFIG_PATH = "/home/kemove/Documents/sam-dino/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/home/kemove/Documents/sam-dino/model/groundingdino_swinb_cogcoor.pth"

SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/home/kemove/Documents/sam-dino/model/sam_vit_h_4b8939.pth"


def init_dino():
    # Building GroundingDINO inference
    grounding_dino_model = Model(
        model_config_path=GROUNDING_DINO_CONFIG_PATH,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    return grounding_dino_model


def init_sam_predictor():
    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    return sam_predictor


def init_sam_auto_mask_generator():
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_auto_mask_generator = SamAutomaticMaskGenerator(
        model=sam
        # points_per_side=32,
        # points_per_batch=64,
        # pred_iou_thresh=0.88,
        # stability_score_thresh=0.95,
        # stability_score_offset=1.0,
        # box_nms_thresh=0.7,
        # crop_n_layers=0,
        # crop_nms_thresh=0.7,
        # crop_overlap_ratio=512 / 1500,
        # crop_n_points_downscale_factor=1,
        # point_grids=None,
        # min_mask_region_area=0,
        # output_mode="binary_mask"
    )
    return sam_auto_mask_generator


import os
import shutil


def delete_all_files_in_folder(folder):
    """
    删除指定文件夹中的所有文件和子文件夹。

    :param folder: 要清空的文件夹路径
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'无法删除 {file_path}. 原因: {e}')
