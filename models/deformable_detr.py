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
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .deformable_transformer import build_deforamble_transformer
from .utils import GradientReversal
from .F_attention import FSAS,LFFFN

import copy





def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False,
                 backbone_align=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels  #在config.yaml设置，默认为4
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)  #默认[8,16,32]
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.uda = backbone_align
        self.backbone_align = backbone_align

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        if backbone_align:

            #---------fuse module
            self.f_attention = FSAS(dim = 256) #频域注意力
            #----Learn FFN
            self.LF_FFN = LFFFN(dim=256)
            self.grl = GradientReversal()
            self.backbone_D = MLP(hidden_dim, hidden_dim, 1, 3)
            for layer in self.backbone_D.layers:
                nn.init.xavier_uniform_(layer.weight, gain=1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, samples: NestedTensor,SSOD_flag = False):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)#s=输出为3层特征（8、16、32）及其对应位置编码
        srcs = []
        masks = []
        #-----将8、16、32尺度的特征通道变为 HIDDEN_DIM：256----------
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        #----默认设置num_feature_levels：4，因此对32倍下采样特征图继续进行下采样,得到倍数为64的特征图及其配置编码
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:#只采样一层的话，使用features的最后一层特征
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                # 生成与原图大小一致的mask[b,h,w]
                m = samples.mask
                # resize为与上一步骤生成特征图大小一致的mask [b,h,w]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                # 生成对应特征图的位置编码
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.two_stage:
            #如果不是two_stage，则使用nn.Embedding生成query_embeds [300,512] 512为HIDDEN_DIM * 2 包括：位置编码query_pos和query两部分
            query_embeds = self.query_embed.weight

        #------transformer encoder + decoder 部分
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact,memory,spatial_shapes = self.transformer(srcs, masks, pos, query_embeds)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        if self.training and self.uda:
            B = outputs_class.shape[1]
            da_output = {}

            if SSOD_flag: #是否开启半监督模式
                outputs_class_unlabel = outputs_class[:,B // 2:]
                outputs_coord_unlabel = outputs_coord[:,B // 2:]
                if self.two_stage:
                    enc_outputs_class_unlabel = enc_outputs_class[B // 2:]
                    enc_outputs_coord_unact_unlabel = enc_outputs_coord_unact[B // 2:]

            # --由于域适应，一半的batch为 目标域，没有label,因此只计算源域的 预测结果
            outputs_class = outputs_class[:, :B // 2]
            outputs_coord = outputs_coord[:, :B // 2]
            if self.two_stage:
                enc_outputs_class = enc_outputs_class[:B // 2]
                enc_outputs_coord_unact = enc_outputs_coord_unact[:B // 2]

            #-------域适应部分---------
            if self.backbone_align:
                # backbone对齐,（8、16、32、64倍）
                # （1）维度变换:[b,c,h,w]->[b,h*w,c]
                # （2）输入特征图过grl层
                # （3）过鉴别器得到最终得分 [b,h*w,c] -> [b,h*w,1] ,鉴别器为简单的MLP
                # （4）torch.cat将多尺度特征拼接[b,h*w,1] -> [b,多层级 h*w,1]

                out_srcs_da = []
                memory_list = memory.split([H_ * W_ for H_, W_ in spatial_shapes], dim=1)
                for idx, src in enumerate(srcs):
                    pos_embed = pos[idx].flatten(2).transpose(1, 2)  #位置编码
                    src_da = src.flatten(2).transpose(1, 2)
                    sub_memory = memory_list[idx]
                    #--- frequency cross-attention
                    src_da = self.f_attention((src_da + pos_embed), (sub_memory + pos_embed), src_da)
                    #--- 可学习 L_FFN
                    src_da = self.LF_FFN(src_da)
                    out_srcs_da.append(src_da)


                da_output['backbone'] = torch.cat(
                    [self.backbone_D(self.grl(src)) for src in out_srcs_da], dim=1)


        #记录结果
        #（1）最后一层预测结果
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.aux_loss:
            #中间层预测结果
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}

        if self.training and self.uda:
            out['da_output'] = da_output

        if self.training and SSOD_flag:
            out['pred_logits_unlabel'] =  outputs_class_unlabel[-1]
            out['pred_boxes_unlabel']  =  outputs_coord_unlabel[-1]
            if self.aux_loss:
                out['aux_outputs_unlabel'] = self._set_aux_loss(outputs_class_unlabel, outputs_coord_unlabel)
            if self.two_stage:
                enc_outputs_coord_unlabel = enc_outputs_coord_unact_unlabel.sigmoid()
                out['enc_outputs_unlabel'] = {'pred_logits': enc_outputs_class_unlabel, 'pred_boxes': enc_outputs_coord_unlabel}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25, da_gamma=2):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.da_gamma = da_gamma

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def loss_da(self, outputs, use_focal=False):
        B = outputs.shape[0]
        assert B % 2 == 0

        targets = torch.empty_like(outputs)
        targets[:B//2] = 0
        targets[B//2:] = 1

        loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')

        if use_focal:
            prob = outputs.sigmoid()
            p_t = prob * targets + (1 - prob) * (1 - targets)
            loss = loss * ((1 - p_t) ** self.da_gamma)

        return loss.mean()

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets,ssod_flag = False):

        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             #outputs:dict_keys(['pred_logits', 'pred_boxes', 'aux_outputs', 'da_output'])
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if ssod_flag:
            outputs_without_aux = {k.replace('_unlabel',''): v for k, v in outputs.items() if k != 'aux_outputs_unlabel' and k != 'enc_outputs_unlabel'}
            #改变pseudo_outputs命名，与非半监督训练命名规则保持一致
            outputs.update({'pred_boxes': outputs.pop('pred_boxes_unlabel')})
            outputs.update({'pred_logits': outputs.pop('pred_logits_unlabel')})
        else:
            outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        # Retrieve the matching between the outputs of the last layer and the targets

        #使用匈牙利匹配得到对应索引
        if len(targets) > 0: #存在伪标签
            indices = self.matcher(outputs_without_aux, targets)
            # Compute the average number of target boxes accross all nodes, for normalization purposes
            num_boxes = sum(len(t["labels"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)
        else:   #设置虚假值，以保证DDP同步
            indices = None
            num_boxes = torch.as_tensor([1], dtype=torch.float, device=outputs['pred_logits'].device)

        #------导致DDP卡住,因此需要设置indices以保证同步----------
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)

        if indices is None:
            num_boxes = num_boxes - 1
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        #---------------------------
        if indices is None: #不存在为标签
            return {}

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' or 'aux_outputs_unlabel' in outputs:
            if ssod_flag: #计算半监督损失
                key_aux = 'aux_outputs_unlabel'
            else:
                key_aux = 'aux_outputs'
            for i, aux_outputs in enumerate(outputs[key_aux]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' or 'enc_outputs_unlabel' in outputs:
            if ssod_flag:  # 计算半监督损失
                key_enc = 'enc_outputs_unlabel'
            else:
                key_enc = 'enc_outputs'
            enc_outputs = outputs[key_enc]
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'da_output' in outputs:
            for k, v in outputs['da_output'].items():
                losses[f'loss_{k}'] = self.loss_da(v, use_focal= False)  #对于query的均为focal loss

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes,SSOD = False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)  #(b,num_cls * num_query)

        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        if not SSOD:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        else:  #SSOD 保证后处理坐标为 中心点+宽高 格式
            boxes = out_bbox
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        #---记录prob，自适应阈值使用
        prob = torch.gather(prob, 1, topk_boxes.unsqueeze(-1).repeat(1,1,prob.shape[-1]))
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = [{'scores': s, 'labels': l, 'boxes': b,'prob':p} for s, l, b,p in zip(scores, labels, boxes,prob)]

        return results

#--------add nms---------------------------
from torchvision.ops.boxes import batched_nms
class NMSPostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes,SSOD = False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        print('NMSPostProcess处理')
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        bs, n_queries, n_cls = out_logits.shape

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()

        all_scores = prob.view(bs, n_queries * n_cls).to(out_logits.device)
        all_indexes = torch.arange(n_queries * n_cls)[None].repeat(bs, 1).to(out_logits.device)
        all_boxes = all_indexes // out_logits.shape[2]
        all_labels = all_indexes % out_logits.shape[2]

        if not SSOD:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        else:  #SSOD 保证后处理坐标为 中心点+宽高 格式
            boxes = out_bbox
        boxes = torch.gather(boxes, 1, all_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        results = []
        for b in range(bs):
            box = boxes[b]
            score = all_scores[b]
            lbls = all_labels[b]

            pre_topk = score.topk(100).indices
            box = box[pre_topk]
            score = score[pre_topk]
            lbls = lbls[pre_topk]

            keep_inds = batched_nms(box, score, lbls, 0.1)[:100]
            results.append({
                'scores': score[keep_inds],
                'labels': lbls[keep_inds],
                'boxes':  box[keep_inds],
            })

        return results

#------------------------

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        #input_dim:256,hidden_dim:256,output_dim:1,num_layers:3
        super().__init__()
        self.num_layers = num_layers #3
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    device = torch.device(cfg.DEVICE)

    backbone = build_backbone(cfg) #resnet50 + 对应每层级特征图的位置编码

    transformer = build_deforamble_transformer(cfg)   #
    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=cfg.DATASET.NUM_CLASSES,
        num_queries=cfg.MODEL.NUM_QUERIES,
        num_feature_levels=cfg.MODEL.NUM_FEATURE_LEVELS,
        aux_loss=cfg.LOSS.AUX_LOSS,
        with_box_refine=cfg.MODEL.WITH_BOX_REFINE,
        two_stage=cfg.MODEL.TWO_STAGE,
        backbone_align=cfg.MODEL.BACKBONE_ALIGN,
    )
    if cfg.MODEL.MASKS:
        model = DETRsegm(model, freeze_detr=(cfg.MODEL.FROZEN_WEIGHTS is not None))
    matcher = build_matcher(cfg)
    weight_dict = {'loss_ce': cfg.LOSS.CLS_LOSS_COEF, 'loss_bbox': cfg.LOSS.BBOX_LOSS_COEF}
    weight_dict['loss_giou'] = cfg.LOSS.GIOU_LOSS_COEF
    if cfg.MODEL.MASKS:
        weight_dict["loss_mask"] = cfg.LOSS.MASK_LOSS_COEF
        weight_dict["loss_dice"] = cfg.LOSS.DICE_LOSS_COEF
    # TODO this is a hack
    if cfg.LOSS.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(cfg.MODEL.DEC_LAYERS - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    weight_dict['loss_backbone'] = cfg.LOSS.BACKBONE_LOSS_COEF

    # --------SSOD
    weight_dict['teacher_loss_weight'] = cfg.SSOD.teacher_loss_weight

    losses = ['labels', 'boxes', 'cardinality']
    if cfg.MODEL.MASKS:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(cfg.DATASET.NUM_CLASSES, matcher, weight_dict, losses, focal_alpha=cfg.LOSS.FOCAL_ALPHA, da_gamma=cfg.LOSS.DA_GAMMA)
    criterion.to(device)
    #for ssod
    criterion_ssod = SetCriterion(cfg.DATASET.NUM_CLASSES, matcher, weight_dict, losses, focal_alpha=cfg.LOSS.FOCAL_ALPHA, da_gamma=cfg.LOSS.DA_GAMMA)
    criterion_ssod.to(device)
    postprocessors = {'bbox': PostProcess()}
    if cfg.MODEL.MASKS:
        postprocessors['segm'] = PostProcessSegm()
        if cfg.DATASET.DATASET_FILE == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion,criterion_ssod, postprocessors
