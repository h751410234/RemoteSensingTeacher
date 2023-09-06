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

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import warnings
from typing import Iterable

import numpy as np
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher,get_unlabel_img,to_cuda
from models.deformable_detr import PostProcess,NMSPostProcess
from models.label_match import get_pseudo_label_via_threshold,deal_pesudo_label,spilt_output,\
    get_vaild_output,show_pesudo_label_with_gt,rescale_pseudo_targets









def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets, _,_= prefetcher.next()
    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):

        #outputs:dict_keys(['pred_logits', 'pred_boxes', 'aux_outputs', 'da_output'])
        outputs = model(samples,SSOD_flag = False)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets, _,_= prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}





def train_one_epoch_with_ssod(model: torch.nn.Module, criterion: torch.nn.Module, criterion_SSOD: torch.nn.Module,
                    data_loader: Iterable,data_loader_train_strong_aug: Iterable ,optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, EMA_model: torch.nn.Module,output_dir,AT = None,class_num = 4, th = 0.5,max_norm: float = 0):

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if data_loader_train_strong_aug is not None: #开启强增广
        prefetcher = data_prefetcher(data_loader_train_strong_aug, device, prefetch=True)
    else:
        prefetcher = data_prefetcher(data_loader, device, prefetch=True)

    postprocessors = {'bbox': PostProcess()}
    samples, targets,unlabel_targets,samples_strong_aug = prefetcher.next()
    cache_loss_array = []
    cache_ssod_loss_array = []
    cache_th_list = []

    for _ in metric_logger.log_every(range(len(data_loader_train_strong_aug)), print_freq, header):
        #outputs:dict_keys(['pred_logits', 'pred_boxes', 'aux_outputs', 'da_output'])
        #-------------1.半监督推理，得到伪标签-----------------------------
        #1.1得到输入图像（unlabel） weak_aug
        unlabel_samples_img = get_unlabel_img(samples)
        # if samples_strong_aug is not None:
          #   unlabel_samples_img_strong_aug = get_unlabel_img(samples_strong_aug)  #强增广图像
        with torch.no_grad():
            #1.2--------得到输出结果
            outputs_unlabel_ema = EMA_model.ema(unlabel_samples_img,SSOD_flag = True) #老师模型预测全部结果，使用weak_aug，img
            #（1）预测结果后处理，得到分类+回归结果,其中box 为中心点+宽高格式（与target保持一致）
            unlabel_targets = [{k: v.to(device) for k, v in t.items()} for t in unlabel_targets]
            orig_unlabel_target_sizes = torch.stack([torch.tensor([1,1]).to(device) for i in range(len(unlabel_targets))], dim=0)  #保证坐标归一化
            results = postprocessors['bbox'](outputs_unlabel_ema, orig_unlabel_target_sizes,SSOD=True)  #[{'scores': s, 'labels': l, 'boxes': b}]
            #（2）筛选伪标签，得到可靠的伪标签 （待创新）
            #2.1 自适应阈值方法
            if AT: #自适应阈值
                threshold = AT.masking(results)
                cache_th_list.append(threshold)
            else: #固定阈值方法
                threshold = np.asarray([th] * (class_num -1))
                cache_th_list.append(threshold)
            idx_list,labels_dict,boxes_dict,scores_dcit = get_pseudo_label_via_threshold(results,threshold = threshold)    #卡阈值以得到可靠伪标签
            #(3)将伪标签结果处理为计算损失的格式
            unlabel_pseudo_targets = deal_pesudo_label(unlabel_targets,idx_list,labels_dict,boxes_dict,scores_dcit)
            #(4)对pseudo_label坐标 进行  nms 和 比例缩放，保证与unlabel_targets一致
            unlabel_pseudo_targets = rescale_pseudo_targets(unlabel_samples_img,unlabel_pseudo_targets)
            #(5)可视化，用于debug
           # print('可视化')
           # show_pesudo_label_with_gt(unlabel_samples_img,unlabel_pseudo_targets,unlabel_targets,idx_list,unlabel_samples_img_strong_aug)
        #---------------------2.学生模型对全图进行推理---------------------------
        #outputs = model(samples,SSOD_flag =True)   #对弱增广样本进行推理
        outputs = model(samples_strong_aug,SSOD_flag =True)   #对强增广样本进行推理
        #---------------------3.拆分预测结果为 （源域，目标域）以方便后续分别计算损失---------------------------
        #(1)拆分结果
        target_outputs,pesudo_outputs = spilt_output(outputs)

        #（2）得到对应伪标签的预测结果 + 格式处理后的标签
        vaild_pesudo_outputs,unlabel_pseudo_targets = get_vaild_output(pesudo_outputs,unlabel_pseudo_targets,idx_list)

        #---------------------4.计算损失(注意需要单独计算伪标签损失)--------------------------------
        #（1）源域目标检测损失 + 域适应损失
        weight_dict = criterion.weight_dict
        loss_dict = criterion(target_outputs, targets,ssod_flag = False)

        #（2）目标域伪标签计算损失

        # 计算SSOD损失
        loss_dict_unlabel = criterion_SSOD(vaild_pesudo_outputs, unlabel_pseudo_targets, ssod_flag=True)  # DDP计算有问题

        #(3)合并损失
        #(3.1)监督损失
        losses_sup = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # #(3.2)半监督损失
        losses_ssod = sum(loss_dict_unlabel[k] * weight_dict[k] for k in loss_dict_unlabel.keys() if k in weight_dict)
        if losses_ssod == 0:
            losses_ssod = torch.tensor(0)

        #(3.3)合并损失
        losses = losses_sup + losses_ssod * weight_dict['teacher_loss_weight']

       # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)

        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}

        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()

        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets,unlabel_targets,samples_strong_aug = prefetcher.next()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # -----记录损失-------------
    if utils.is_main_process():
        cache_loss_array.append(losses_sup.detach().cpu().numpy())
        cache_ssod_loss_array.append(losses_ssod.detach().cpu().numpy())
        cache_loss_mean = np.asarray(cache_loss_array).mean()
        cache_ssod_loss_mean = np.asarray(cache_ssod_loss_array).mean()
        with open(os.path.join(output_dir,'loss_txt'),'a') as f:
            f.write('sup_loss: %s , ssod_loss: %s \n'%(cache_loss_mean,cache_ssod_loss_mean))

        with open(os.path.join(output_dir,'th_txt'),'a') as f2:
            f2.write('********************************\n')
            for line in cache_th_list:
                line = [str(i) for i in line]
                f2.write('%s \n'%(' ').join(line))
        #-----------------------

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, _,targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
