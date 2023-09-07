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

import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import util.misc as utils
from models import build_model,EMA

from config import get_cfg_defaults

import torchvision.transforms as T
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from util import box_ops

def setup(args):
    cfg = get_cfg_defaults()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    utils.init_distributed_mode(cfg)
    cfg.defrost()

    if cfg.OUTPUT_DIR:
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        os.system(f'cp {args.config_file} {cfg.OUTPUT_DIR}')
        ddetr_src = 'models/deformable_detr.py'
        ddetr_des = Path(cfg.OUTPUT_DIR) / 'deformable_detr.py.backup'
        dtrans_src = 'models/deformable_transformer.py'
        dtrans_des = Path(cfg.OUTPUT_DIR) / 'deformable_transformer.py.backup'
        main_src = 'main.py'
        main_des = Path(cfg.OUTPUT_DIR) / 'main.py.backup'
        os.system(f'cp {ddetr_src} {ddetr_des}')
        os.system(f'cp {dtrans_src} {dtrans_des}')
        os.system(f'cp {main_src} {main_des}')

    return cfg


def main(cfg):
    align = cfg.MODEL.BACKBONE_ALIGN or cfg.MODEL.SPACE_ALIGN or cfg.MODEL.CHANNEL_ALIGN or cfg.MODEL.INSTANCE_ALIGN
    assert align == (cfg.DATASET.DA_MODE == 'uda')

    print("git:\n  {}\n".format(utils.get_sha()))
    print(cfg)

    if cfg.MODEL.FROZEN_WEIGHTS is not None:
        assert cfg.MODEL.MASKS, "Frozen training is meant for segmentation only"

    device = torch.device(cfg.DEVICE)


    model, criterion, criterion_ssod,postprocessors = build_model(cfg)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


#-------------------2023.03.17-------创建EMAmodel---------------------
    # EMA
    ema_model = EMA.ModelEMA(model)


    #------加载推理模型-----
    if cfg.SSOD.RESUME_EMA:
        checkpoint_ema = torch.load(cfg.SSOD.RESUME_EMA, map_location='cpu')
        ema_model.ema.load_state_dict(checkpoint_ema['semi_ema_model'], strict=True)
#---------------------------------------------------------------------


    #---------inference------------------
    # standard PyTorch mean-std input image normalization
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  #inference
    for n,img_name in enumerate(os.listdir(args.img_dir)): #取前300个

        print('已处理:', n)
        img_p = os.path.join(args.img_dir,img_name)
        im = Image.open(img_p).convert('RGB')
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0)

        img = img.cuda()
        # propagate through the model
        outputs = ema_model.ema(img)

        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        keep = scores[0] > args.visual_score
        boxes = boxes[0, keep]
        labels = labels[0, keep]

        # and from relative [0, 1] to absolute [0, height] coordinates
        im_h, im_w = im.size
        # print('im_h,im_w',im_h,im_w)
        target_sizes = torch.tensor([[im_w, im_h]])
        target_sizes = target_sizes.cuda()
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # plot_results

        save_img_p = os.path.join(args.output_dir,'%s.png'%img_name.split('.')[0])
        source_img = Image.open(img_p).convert("RGB")

        draw = ImageDraw.Draw(source_img)

        for n,(xmin, ymin, xmax, ymax) in enumerate(boxes[0].tolist()):
            c = str(args.label_list[labels[0].tolist()])
            if c == 'Plane':
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="yellow",width=3)
            elif c == 'Ship':
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="red",width=3)
            elif c == 'Storage-tank':
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="blue",width=3)
            else:
                print('others')

        source_img.save(save_img_p, "png")
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR Detector')
    parser.add_argument('--config_file', default='', type=str)
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--img_dir", default='', type=str)
    parser.add_argument("--output_dir", default='', type=str)
    parser.add_argument("--label_list", default=['Plane','Storage-tank','Ship'], type=str)
    parser.add_argument("--visual_score", default=0.2, type=float)
    args = parser.parse_args()
    cfg = setup(args)
    main(cfg)
