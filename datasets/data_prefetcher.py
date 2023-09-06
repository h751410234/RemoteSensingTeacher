# ------------------------------------------------------------------------
# Modified by Wei-Jie Huang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch




def to_cuda(samples, targets,unlabel_targets,target_img_strong_aug, device):
    samples = samples.to(device, non_blocking=True)
    if target_img_strong_aug is not None:  #判断是否为None
        target_img_strong_aug = target_img_strong_aug.to(device, non_blocking=True)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    unlabel_targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in unlabel_targets]
    return samples, targets,unlabel_targets,target_img_strong_aug





class data_prefetcher():
    def __init__(self, loader, device, prefetch=True):
        self.loader = iter(loader)
        self.prefetch = prefetch
        self.device = device
        if prefetch:
            self.stream = torch.cuda.Stream()
            self.preload()

    def preload(self):
        try:
            self.next_samples, self.next_targets,self.next_unlabel_targets,self.next_target_img_strong_aug = next(self.loader)
        except StopIteration:
            self.next_samples = None
            self.next_targets = None
            self.next_unlabel_targets = None
            self.next_target_img_strong_aug = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_samples, self.next_targets,self.next_unlabel_targets,self.next_target_img_strong_aug = to_cuda(self.next_samples, self.next_targets,self.next_unlabel_targets,self.next_target_img_strong_aug,self.device)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:

    def next(self):
        if self.prefetch:
            torch.cuda.current_stream().wait_stream(self.stream)
            samples = self.next_samples
            targets = self.next_targets
            unlabel_targets = self.next_unlabel_targets
            target_img_strong_aug = self.next_target_img_strong_aug
            if samples is not None :
                samples.record_stream(torch.cuda.current_stream())
            if target_img_strong_aug is not None:
                target_img_strong_aug.record_stream(torch.cuda.current_stream())
            if targets is not None:
                for t in targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            if unlabel_targets is not None:
                for t in unlabel_targets:
                    for k, v in t.items():
                        v.record_stream(torch.cuda.current_stream())
            self.preload()
        else:
            try:
                samples, targets,unlabel_targets,target_img_strong_aug = next(self.loader)
                samples, targets,unlabel_targets,target_img_strong_aug = to_cuda(samples, targets,unlabel_targets, target_img_strong_aug,self.device)
            except StopIteration:
                samples = None
                targets = None
                unlabel_targets = None
                target_img_strong_aug = None
        return samples, targets,unlabel_targets,target_img_strong_aug


def get_unlabel_img(nestedtensor):
    images,masks = nestedtensor.decompose()
    b,c,h,w = images.shape
    unlabel_b = b // 2
    unlabel_img = images[unlabel_b:,:,:,:]
    return unlabel_img


