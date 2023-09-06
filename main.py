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
        # check the resumed model
        if not cfg.EVAL:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, cfg.OUTPUT_DIR
            )
            print('测试完毕'.center(50, '*'))

    # if cfg.EVAL:
    #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                           data_loader_val, base_ds, device, cfg.OUTPUT_DIR)
    #     if cfg.OUTPUT_DIR:
    #         utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    #     return

#-------------------2023.03.17-------创建EMAmodel---------------------
    # EMA
    ema_model = EMA.ModelEMA(model)

    semi_ema = None
    #评估时使用
    #原版模型
    best_checkpoint_fitness = 0
    #semi_ema
    best_semi_ema_fitness = 0
    cache_best_semi_ema_epoch = 0
    #ema
    best_ema_fitness = 0
    cache_best_ema_epoch = 0

    #---记录评估指标---
    ema_model_eval = []
    semi_ema_model_eval = []
    if cfg.SSOD.RESUME_EMA:
        checkpoint_ema = torch.load(cfg.SSOD.RESUME_EMA, map_location='cpu')
        ema_model.ema.load_state_dict(checkpoint_ema['ema_model'], strict=True)
#---------------------------------------------------------------------

#===============adaptive thoushold 20230406=================
    th = cfg.SSOD.thoushold_value
    AT = None
    if cfg.SSOD.adaptive_thoushold:
        from models.label_match import AdaptiveThreshold
        #自适应阈值网络初始化
        AT = AdaptiveThreshold(cfg.DATASET.NUM_CLASSES - 1,th)
#==========================================================

    print("Start training")
    cfg.START_EPOCH = 0    #修改初始位置用于半监督训练
    start_time = time.time()
    for epoch in range(cfg.START_EPOCH, cfg.TRAIN.EPOCHS):
        #--------------原版对抗训练----------------
        # epoch从 0 开始
        if epoch < cfg.SSOD.burn_epochs:
            if cfg.DIST.DISTRIBUTED:
                sampler_train.set_epoch(epoch)
            #--------------------------------------
            #---由于训练波动较大，将最佳模型赋予model用于40epoch之后的训练，以进一步提升最佳模型结果

            if epoch == cfg.TRAIN.LR_DROP:
               # if utils.is_main_process():
                print(
                    '加载最佳模型'
                )

                checkpoint_ema = torch.load(os.path.join(output_dir,'best_ema.pth'), map_location='cpu')
                if not cfg.DIST.DISTRIBUTED:
                    model_without_ddp.load_state_dict(checkpoint_ema['ema_model'], strict=True)
                else:#DDP模型
                    state_dict = {k.replace("module.",""):v for k, v in checkpoint_ema['ema_model'].items()}
                    model_without_ddp.load_state_dict(state_dict, strict=True)

                #评估，用于debug
                test_stats, coco_evaluator = evaluate(
                    model, criterion, postprocessors, data_loader_val, base_ds, device, cfg.OUTPUT_DIR
                )            #----------------------------------------


            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch, cfg.TRAIN.CLIP_MAX_NORM)


        else:
            if cfg.DIST.DISTRIBUTED:
               sampler_train.set_epoch(epoch)
               sampler_train_strong_aug.set_epoch(epoch)


            # -------------------2023.03.17-------创建EMAmodel---------------------
            #（1）ema权重赋值
            # 1.1 将ema的权重赋值给model
            if epoch == cfg.SSOD.burn_epochs:
                    #重新设置学习率，学习率*10
                    for p in optimizer.param_groups:
                        p['lr'] *= 10

                    print(
                        '加载最佳模型，半监督训练'
                    )

                    checkpoint_ema = torch.load(os.path.join(output_dir,'best_ema.pth'), map_location='cpu')
                    if not cfg.DIST.DISTRIBUTED:
                        model_without_ddp.load_state_dict(checkpoint_ema['ema_model'], strict=True)
                        ema_model.ema.load_state_dict(checkpoint_ema['ema_model'], strict=True)
                    else: #DDP
                        state_dict = {k.replace("module.", ""): v for k, v in checkpoint_ema['ema_model'].items()}
                        model_without_ddp.load_state_dict(state_dict, strict=True)
                        ema_model.ema.load_state_dict(state_dict, strict=True)

                    # 评估，用于debug
                    # test_stats, coco_evaluator = evaluate(
                    #     ema_model.ema, criterion, postprocessors, data_loader_val, base_ds, device, cfg.OUTPUT_DIR
                    # )  # ---------------------


                    #1.2 创建semi_ema
                    if cfg.SSOD.cosine_ema:
                        semi_ema = EMA.CosineEMA(ema_model.ema, decay_start=cfg.SSOD.ema_rate,
                                                  total_epoch= cfg.TRAIN.EPOCHS - cfg.SSOD.burn_epochs)
                    else:
                        semi_ema = EMA.SemiSupModelEMA(ema_model.ema, cfg.SSOD.ema_rate)
            #（2）训练
            train_stats = train_one_epoch_with_ssod(
                model, criterion, criterion_ssod,data_loader_train, data_loader_train_strong_aug,optimizer, device, epoch, ema_model,output_dir,
                AT,cfg.DATASET.NUM_CLASSES,th,cfg.TRAIN.CLIP_MAX_NORM)

        #（3）训练后
        #需要更新 ema 和 semi_ema
        #1)更新ema
        ema_model.update(model)
        #2)更新semi_ema
        if semi_ema:
            semi_ema.update_decay(epoch - cfg.SSOD.burn_epochs)  #更新semi_ema的decay
            semi_ema.update(ema_model.ema)

        lr_scheduler.step()

        #（4）评估
        #4.1原版模型评估
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, cfg.OUTPUT_DIR
        )
        #4.2
        #1）存在semi_ema则评估semi_ema
        if epoch >= cfg.SSOD.burn_epochs:
            test_stats_semi_ema, coco_evaluator_semi_ema = evaluate(
                semi_ema.ema, criterion, postprocessors, data_loader_val, base_ds, device, cfg.OUTPUT_DIR
            )
        #2)不存在则评估ema_model
        else:
            test_stats_ema, coco_evaluator_ema = evaluate(
                ema_model.ema, criterion, postprocessors, data_loader_val, base_ds, device, cfg.OUTPUT_DIR
            )

        #（5）存储
        if cfg.OUTPUT_DIR and utils.is_main_process():
            #5.1存储最佳原版模型（IOU=0.5）
            if test_stats['coco_eval_bbox'][1] >= best_checkpoint_fitness:
                best_checkpoint_fitness = test_stats['coco_eval_bbox'][1]
                checkpoint_path = output_dir / 'best_checkpoint.pth'
                cache_best_checkpoint_epoch = epoch #记录epoch数
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'cfg': cfg,
                }, checkpoint_path)

            #5.2 存储最佳ema模型（IOU=0.5）
            #1）存储最佳semi_ema
            if epoch >= cfg.SSOD.burn_epochs:
                semi_ema_model_eval.append(test_stats_semi_ema['coco_eval_bbox'][1])
                #记录结果
                with open(output_dir / "semi_ema_model_eval.txt", 'w') as f:
                    for i in semi_ema_model_eval:
                        f.write('%s\n'%i)

                if test_stats_semi_ema['coco_eval_bbox'][1] >= best_semi_ema_fitness:
                    best_semi_ema_fitness = test_stats_semi_ema['coco_eval_bbox'][1]
                    checkpoint_path = output_dir / 'best_semi_ema.pth'
                    cache_best_semi_ema_epoch = epoch  # 记录epoch数
                    utils.save_on_master({
                        'semi_ema_model': semi_ema.ema.state_dict(),
                        'epoch': epoch,
                    }, checkpoint_path)

            #2）存储最佳ema
            if epoch < cfg.SSOD.burn_epochs:
                ema_model_eval.append(test_stats_ema['coco_eval_bbox'][1])
                #记录结果
                with open(output_dir / "ema_model_eval.txt", 'w') as f:
                    for i in ema_model_eval:
                        f.write('%s\n'%i)

                if test_stats_ema['coco_eval_bbox'][1] >= best_ema_fitness:
                    best_ema_fitness = test_stats_ema['coco_eval_bbox'][1]
                    checkpoint_path = output_dir / 'best_ema.pth'
                    cache_best_ema_epoch = epoch  # 记录epoch数
                    utils.save_on_master({
                        'ema_model': ema_model.ema.state_dict(),
                        'epoch': epoch,
                    }, checkpoint_path)

            #（6）记录日志
            with open(output_dir / "log_best.txt",'w') as f:
                f.write('best_checkpoint -->  map50:%s , epoch:%s\n'%(best_checkpoint_fitness,cache_best_checkpoint_epoch))
                f.write('best_semi_ema -->  map50:%s , epoch:%s\n'%(best_semi_ema_fitness,cache_best_semi_ema_epoch))
                f.write('best_ema -->  map50:%s , epoch:%s\n'%(best_ema_fitness,cache_best_ema_epoch))



        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if cfg.OUTPUT_DIR and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR Detector')
    parser.add_argument('--config_file', default='', type=str)
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = setup(args)
    main(cfg)
