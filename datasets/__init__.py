# ------------------------------------------------------------------------
# Modified by Wei-Jie Huang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, cfg,strong_aug = False):
    if cfg.DATASET.DATASET_FILE == 'coco':
        return build_coco(image_set, cfg)
    if cfg.DATASET.DATASET_FILE == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, cfg)
    DAOD_dataset = [
        'cityscapes_to_foggy_cityscapes',
        'sim10k_to_cityscapes_caronly',
        'cityscapes_to_bdd_daytime',
        #----------遥感场景------------------
        'xView3c_to_DOTA3c',
        'xView3c_small_to_DOTA3c_small',
        'optical_to_infrared',
        'AOD_to_UVA',
        'UCASAOD_to_CARPK',
        'CARPK_to_UCASAOD',
        'HRRSD_to_SSDD',
        'clear_to_cloudy',

    ]
    if cfg.DATASET.DATASET_FILE in DAOD_dataset:
        from .DAOD import build
        return build(image_set, cfg,strong_aug)
    raise ValueError(f'dataset {cfg.DATASET.DATASET_FILE} not supported')
