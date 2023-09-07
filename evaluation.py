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
from torch.utils.data import DataLoader
import datasets
import datasets.DAOD as DAOD
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch,train_one_epoch_with_ssod
from models import build_model,EMA


from config import get_cfg_defaults


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

    # fix the seed for reproducibility
    seed = cfg.SEED + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True



    model, criterion, criterion_ssod,postprocessors = build_model(cfg)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', cfg=cfg)
    dataset_val = build_dataset(image_set='val', cfg=cfg)
    if cfg.SSOD.strong_aug:    #是否创建半监督强增广dataset
        dataset_train_strong_aug = build_dataset(image_set='train', cfg=cfg,strong_aug = True)
    else:
        dataset_train_strong_aug = None

    if cfg.DIST.DISTRIBUTED:
        if cfg.CACHE_MODE:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
            if dataset_train_strong_aug is not None:  # 半监督强增广使用
                sampler_train_strong_aug = samplers.NodeDistributedSampler(dataset_train_strong_aug)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
            if dataset_train_strong_aug is not None:  # 半监督强增广使用
                sampler_train_strong_aug = samplers.DistributedSampler(dataset_train_strong_aug)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if dataset_train_strong_aug is not None: #半监督强增广使用
            sampler_train_strong_aug = torch.utils.data.RandomSampler(dataset_train_strong_aug)

    if cfg.DATASET.DA_MODE == 'uda':
        assert cfg.TRAIN.BATCH_SIZE % 2 == 0, f'cfg.TRAIN.BATCH_SIZE {cfg.TRAIN.BATCH_SIZE} should be a multiple of 2'
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, cfg.TRAIN.BATCH_SIZE//2, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=DAOD.collate_fn, num_workers=cfg.NUM_WORKERS,
                                       pin_memory=True)

        if dataset_train_strong_aug is not None: #半监督强增广使用
            batch_sampler_train_strong_aug = torch.utils.data.BatchSampler(
                sampler_train_strong_aug, cfg.TRAIN.BATCH_SIZE // 2, drop_last=True)
            data_loader_train_strong_aug = DataLoader(dataset_train_strong_aug, batch_sampler=batch_sampler_train_strong_aug,
                                           collate_fn=DAOD.collate_fn, num_workers=cfg.NUM_WORKERS,
                                           pin_memory=True)
        else:
            data_loader_train_strong_aug = None


    else:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, cfg.TRAIN.BATCH_SIZE, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                       pin_memory=True)
    data_loader_val = DataLoader(dataset_val, cfg.TRAIN.BATCH_SIZE, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=cfg.NUM_WORKERS,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, cfg.TRAIN.LR_BACKBONE_NAMES) and not match_name_keywords(n, cfg.TRAIN.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
            "lr": cfg.TRAIN.LR,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.LR_BACKBONE_NAMES) and p.requires_grad],
            "lr": cfg.TRAIN.LR_BACKBONE,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, cfg.TRAIN.LR_LINEAR_PROJ_NAMES) and p.requires_grad],
            "lr": cfg.TRAIN.LR * cfg.TRAIN.LR_LINEAR_PROJ_MULT,
        }
    ]
    if cfg.TRAIN.SGD:
        optimizer = torch.optim.SGD(param_dicts, lr=cfg.TRAIN.LR, momentum=0.9,
                                    weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP)

    if cfg.DIST.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.DIST.GPU],find_unused_parameters=True)
        model_without_ddp = model.module

    if cfg.DATASET.DATASET_FILE == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", cfg)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if cfg.MODEL.FROZEN_WEIGHTS is not None:
        checkpoint = torch.load(cfg.MODEL.FROZEN_WEIGHTS, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(cfg.OUTPUT_DIR)
    if cfg.RESUME: # [BUG] write after freezing cfgs
        print('加载训练好的模型权重'.center(50,'*'))
        if cfg.RESUME.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.RESUME, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(cfg.RESUME, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not cfg.EVAL and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']

            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            override_resumed_lr_drop = True
            if override_resumed_lr_drop:
                print('Warning: (hack) override_resumed_lr_drop is set to True, so cfg.TRAIN.LR_DROP would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = cfg.TRAIN.LR_DROP
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            cfg.START_EPOCH = checkpoint['epoch'] + 1 

    # EMA
    ema_model = EMA.ModelEMA(model)

    #创建semi_ema
    if cfg.SSOD.cosine_ema:
        semi_ema = EMA.CosineEMA(ema_model.ema, decay_start=cfg.SSOD.ema_rate,
                                  total_epoch= cfg.TRAIN.EPOCHS - cfg.SSOD.burn_epochs)
    else:
        semi_ema = EMA.SemiSupModelEMA(ema_model.ema, cfg.SSOD.ema_rate)

#---------------------------------------------------------------------

    print("Start Evaluation")

    #(1)评估原版模型
    if cfg.RESUME :
        print('Evaluation Model'.center(50,'*'))
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, cfg.OUTPUT_DIR
        )

    #(2)评估 EMA模型
    if cfg.SSOD.RESUME_EMA:
        print('Evaluation EMA Model'.center(50, '*'))
        checkpoint_ema = torch.load(cfg.SSOD.RESUME_EMA, map_location='cpu')
        if 'semi_ema' in cfg.SSOD.RESUME_EMA: #评估semi_ema模型
            semi_ema.ema.load_state_dict(checkpoint_ema['semi_ema_model'], strict=True)
            test_stats_semi_ema, coco_evaluator_semi_ema = evaluate(
                semi_ema.ema, criterion, postprocessors, data_loader_val, base_ds, device, cfg.OUTPUT_DIR
            )
        else: #评估ema模型
            ema_model.ema.load_state_dict(checkpoint_ema['ema_model'], strict=True)
            test_stats_ema, coco_evaluator_ema = evaluate(
                ema_model.ema, criterion, postprocessors, data_loader_val, base_ds, device, cfg.OUTPUT_DIR
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR Detector')
    parser.add_argument('--config_file', default='', type=str)
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = setup(args)
    main(cfg)
