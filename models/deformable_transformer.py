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

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 ):
        super().__init__()

        self.d_model = d_model #HIDDEN_DIM
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        #----创建encoder部分
        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)

        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers) #创建6层
        #----创建decoder部分
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)
        #-----层级编码
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))  #[4,256]


        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        #************开始处理，默认4层特征
        #（1）数据预处理，将特征(h,w)维度合并，生成层级特征编码 并于 位置编码合并（相加）
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)): #0,1,2,3
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            #合并h,w后交换维度 ，[b,c:256,h,w] -> [b,h*w,c]
            src = src.flatten(2).transpose(1, 2)
            #mask 由于没有c通道 直接展平即可 [b,h*w]
            mask = mask.flatten(1)
            #pos_embed  同理，[b,c:256,h,w] -> [b,h*w,c]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            #由于使用多尺度特征，self.level_embed用于区分不同尺度特征（使用torch.Tensor初始化，是可学习的参数）
            #self.level_embed[lvl].view(1, 1, -1):[1,1,256] lvl_pos_embed:[b,h*w,c:256]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1) #位置编码+层级编码
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        #----特征cat，在dim=1(h*w)维度拼接(concat)
        src_flatten = torch.cat(src_flatten, 1)   #(b,多尺度concath*w,256)
        mask_flatten = torch.cat(mask_flatten, 1)  #(b,多尺度concath*w)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) #(b,多尺度concath*w,256)
        #spatial_shapes:记录每个层级特征图h，w  维度：[4,2] = [[100,167],[50,84],[25,42],[13,21]]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        #level_start_index:用于记录每个层级特征的开头元素索引维度[4] = [    0, 16700, 20900, 21950]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        #valid_ratios各个层级特征中 有效的宽高比 [B, num_levels, 2],由于有padding，目的  作用：XXX待补充
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        #执行encoder阶段
        memory = self.encoder(
            src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten
        )

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:  #两阶段方法（暂时未看）
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            #query_embed ：[300, 512] 512 为HIDDEN_DIM * 2
            # 包括：tgt（预测目标类别和坐标时使用）和query_embed（生成reference_point使用）两部分
            query_embed, tgt = torch.split(query_embed, c, dim=1)  #c=256，query_embed 和tgt
            #分别扩展维度 [300,256] -> [b,300,256]
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            #通过linear[256,2]转换参考点，过sigmoid实现归一化->[b,256,2]
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(
            tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten
        )

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact,memory,spatial_shapes
        return hs, init_reference_out, inter_references_out, None, None,memory,spatial_shapes


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4
                 ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        """
        src_flatten:(b,多尺度concath*w,256)
        #-------------------
        spatial_shapes:记录每个层级特征图h，w  维度：[4,2] = [[100,167],[50,84],[25,42],[13,21]]
        level_start_index:用于记录每个层级特征的开头元素索引维度[4] = [    0, 16700, 20900, 21950]
        valid_ratios:valid_ratios各个层级特征中 有效的宽高比 [B, num_levels, 2],由于有padding，目的  作用：XXX待补充
        lvl_pos_embed_flatten:位置编码+层级编码 : (b,多尺度concath*w,256)
        mask_flatten: 记录有效像素 (b,多尺度concath*w)
        """
        # self attention
        #（1）self.with_pos_embed：将特征图与位置编码相加
        #*(2)计算  多头可变形注意力，输出维度：[b,多层级h*w,256]
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        #(3)残差连接 + dropout （P=0.1） 0.1的概率 变为0
        src = src + self.dropout1(src2)
        #（4）过 normlayer1(256)
        src = self.norm1(src)

        # ffn
        # src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        # src = src + self.dropout3(src2)
        # src = self.norm2(src)
        #(1)使用linear1(256,1024)将多头注意力机制得到的output进行线性变换[b,多层级h*w,256]->[b,多层级h*w,1024]
        #(2)加入非线性： 过relu 和drouput2
        #(3)使用linear2(1024,256)将output转换为[b,多层级h*w,1024]->[b,多层级h*w,256]
        #(4)残差连接+normlayer2(256)
        src = self.forward_ffn(src) #[b,多层级h*w,256]
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        #获得reference_points，
        # spatial_shapes：维度：[4,2] = [[100,167],[50,84],[25,42],[13,21]]
        # valid_ratios各个层级特征中 有效的宽高比 [B, num_levels, 2],由于有padding
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            #对于每一层feature map初始化每个参考点中心横纵坐标，加减0.5是确保每个初始点是在每个pixel的中心
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            #对网格坐标进行归一化处理,无效值（padding的值）在此处大于1
            #ref_y.reshape(-1)[None]：变换维度 [h,w]->[1,h*w]
            #(valid_ratios[:, None, lvl, 1] * H_)找到最大有效高度值
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            #合并宽高[b,h*w,(x,y):2]
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        #将多层级的特征参考点合并(concat)，[b,多层级h*w,(x,y):2]
        reference_points = torch.cat(reference_points_list, 1)
        #reference_points[:, :, None] :[b,多层级h*w,1,1]
        #valid_ratios[:, None]        :[b,1,num_levels:4,2]
        #(1)扩展维度为[b,多层级h*w,num_levels:4,2]
        #(2)相乘类似于复制操作，由于实际每个点特征计算时 需要聚合 4个层级的对应位置特征，  故将该点特征的 参考点复制4份 分别对应不同层级的参考点
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        #输入的多层级特征
        output = src
        #计算 [可变形 多头 注意力]时需要使用的参考点坐标 [b,多层级h*w,num_levels:4,2]
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        #总共使用6层DeformableTransformerEncoderLayer处理
        for _, layer in enumerate(self.layers):
            output = layer(
                output, pos, reference_points, spatial_shapes, level_start_index, padding_mask
            )

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4
                 ):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt



    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        #src:memory   encoder处理后的特征
        #src_spatial_shapes:spatial_shapes  维度：[4,2] = [[100,167],[50,84],[25,42],[13,21]]
        #src_level_start_index:level_start_index 用于记录每个层级特征的开头元素索引维度[4] = [    0, 16700, 20900, 21950]
        #src_valid_ratios:valid_ratios 各个层级特征中 有效的宽高比 [B, num_levels, 2],由于有padding，目的
        #query_pos:query_embed  #生成的query_embed [b,300,256]
        #src_padding_mask:mask_flatten 记录padding的位置 (b,多尺度concath*w)

        # self attention
        q = k = self.with_pos_embed(tgt, query_pos) #[b,300,256]
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt
        #src:memory   encoder处理后的特征
        #src_spatial_shapes:spatial_shapes  维度：[4,2] = [[100,167],[50,84],[25,42],[13,21]]
        #src_level_start_index:level_start_index 用于记录每个层级特征的开头元素索引维度[4] = [    0, 16700, 20900, 21950]
        #src_valid_ratios:valid_ratios 各个层级特征中 有效的宽高比 [B, num_levels, 2],由于有padding，目的
        #query_pos:query_embed  #生成的query_embed [b,300,256]
        #src_padding_mask:mask_flatten 记录padding的位置 (b,多尺度concath*w)
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                #非 two stage时
                assert reference_points.shape[-1] == 2
                #去除padding后的无效像素
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(
                output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask
            )

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        # https://github.com/fundamentalvision/Deformable-DETR/issues/43
        return [output], [reference_points]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(cfg):
    return DeformableTransformer(
        d_model=cfg.MODEL.HIDDEN_DIM,
        nhead=cfg.MODEL.NHEADS,
        num_encoder_layers=cfg.MODEL.ENC_LAYERS,
        num_decoder_layers=cfg.MODEL.DEC_LAYERS,
        dim_feedforward=cfg.MODEL.DIM_FEEDFORWARD,
        dropout=cfg.MODEL.DROPOUT,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=cfg.MODEL.NUM_FEATURE_LEVELS,
        dec_n_points=cfg.MODEL.DEC_N_POINTS,
        enc_n_points=cfg.MODEL.ENC_N_POINTS,
        two_stage=cfg.MODEL.TWO_STAGE,
        two_stage_num_proposals=cfg.MODEL.NUM_QUERIES
    )


