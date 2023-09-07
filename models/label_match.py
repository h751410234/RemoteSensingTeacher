import numpy as np
import os
import cv2
import torch
import torchvision.transforms as transforms
from util import box_ops
import time

def _make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

#==============自适应阈值================20230406
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output

class AdaptiveThreshold():
    """
    reference SAT in FreeMatch
    """
    def __init__(self, num_classes, th=0.5,momentum=0.90):
        self.num_classes = num_classes
        self.m = momentum
        self.p_model = torch.ones((self.num_classes)) * th# / self.num_classes
        self.label_hist = torch.ones((self.num_classes))  # / self.num_classes
        self.time_p = self.p_model.mean()
        self.class_th = self.p_model.mean()
        self.clip_thresh = False

    @torch.no_grad()
    def update(self,scores_result,probs_result):


        #1.计算  time_p
        mean_scores = scores_result.mean()
        self.time_p = self.time_p * self.m + (1 - self.m) * mean_scores  # 更新  根据预测的平均置信度 更新全局,单一数值

        if self.clip_thresh:  # 默认不使用
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        #2，计算 计算 p_model，针对每一类的阈值
        probs_result = probs_result[:,:-1]  #去除背景类
        max_probs, max_idx = torch.max(probs_result, dim=-1, keepdim=True)

        #calculate_probs each class
        cache_probs = torch.zeros(probs_result.shape[-1])
        for c in range(probs_result.shape[-1]):
            c_idx = (max_idx == c)
            c_max_probs = max_probs[c_idx]
            if int(c_max_probs.shape[0]) > 0: #exist class in img
                c_max_probs_mean = c_max_probs.mean(dim = 0)
                cache_probs[c] = c_max_probs_mean
            else:
                cache_probs[c] = self.p_model[c]
        # print(cache_probs)
        # print('*************')
        # print(self.p_model )
        self.p_model = self.p_model * self.m + (1 - self.m) * cache_probs.to(self.p_model.device) #计算每类阈值


    @torch.no_grad()
    def masking(self, predict_unlabel_result_list):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(predict_unlabel_result_list[0]['scores'].device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(predict_unlabel_result_list[0]['scores'].device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(predict_unlabel_result_list[0]['scores'].device)
        #1.处理预测结果
        pur_scores_all = []
        pur_probs_all = []

        for pul in predict_unlabel_result_list:
            pur_scores_all.append(pul['scores'].detach())
            pur_probs_all.append(pul['prob'].detach())

        pur_scores_all = torch.cat(pur_scores_all,dim = 0)
        pur_probs_all = torch.cat(pur_probs_all,dim = 0)

        pur_scores_all = pur_scores_all.detach()
        pur_probs_all = pur_probs_all.detach()


        #去除阈值小于0.1的预测结果
        probs_result = pur_probs_all[pur_scores_all >= 0.1]
        scores_result = pur_scores_all[pur_scores_all >= 0.1]  # 考虑加入

        #2.根据预测结果更新自适应阈值权重
        if len(probs_result) > 0:
            self.update(scores_result,probs_result)

            #3.得到各个类别自适应阈值大小
            mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
            self.class_th = self.time_p * mod
            return self.class_th.cpu().numpy()
        else:
            return self.class_th.cpu().numpy()

#========================================================

def get_pseudo_label_via_threshold(results,threshold = 0.8):
    cache_idx_list = []
    cache_labels_dict = {}
    cache_boxes_dict = {}
    cache_scores_dict = {}

    #---取消多尺度训练偶尔出现labels为背景的情况,添加一维用于除去
    threshold = np.append(threshold,1.0)
   # print(threshold)
    for n,result in enumerate(results):  #每张图的预测结果
        #{'scores': s, 'labels': l, 'boxes': b}
        # 对应每个类别的 置信度阈值
        threshold_for_class = torch.from_numpy(threshold[result['labels'].cpu().numpy()]).to(result['scores'].device)
        #print(threshold_for_class)
        scores = result['scores']
        vaild_idx = scores >= threshold_for_class
        vaild_labels = result['labels'][vaild_idx]
        vaild_boxes = result['boxes'][vaild_idx]
        vaild_scores = result['scores'][vaild_idx]
        if len(vaild_labels) > 0 :
            cache_idx_list.append(n)
            cache_labels_dict[n] = vaild_labels
            cache_boxes_dict[n] = vaild_boxes
            cache_scores_dict[n] = vaild_scores
    return cache_idx_list,cache_labels_dict,cache_boxes_dict,cache_scores_dict

def deal_pesudo_label(unlabel_target_list,idx_list,pesudo_labels_dict,pesudo_boxes_dict,scores_dcit):
    unlabel_target_format_dict = {}
    for i in idx_list:
        #-----target记录
        cache_unlabel_target_format = {}
        unlabel_target = unlabel_target_list[i]
        cache_unlabel_target_format['labels'] = pesudo_labels_dict[i]
        cache_unlabel_target_format['boxes'] = pesudo_boxes_dict[i]
        cache_unlabel_target_format['scores'] = scores_dcit[i]
        cache_unlabel_target_format['image_id'] = unlabel_target['image_id']
        cache_unlabel_target_format['area'] = unlabel_target['area']
        cache_unlabel_target_format['iscrowd'] = unlabel_target['iscrowd']
        cache_unlabel_target_format['orig_size'] = unlabel_target['orig_size']
        cache_unlabel_target_format['size'] = unlabel_target['size']
        unlabel_target_format_dict[i] = cache_unlabel_target_format
    return unlabel_target_format_dict

def spilt_output(output_dict):
    source_dict = {}
    pesudo_dict = {}
    for k,v in output_dict.items():
        if 'unlabel' in k :
            pesudo_dict[k] = v
        else:
            source_dict[k] = v
    return source_dict,pesudo_dict

def get_vaild_output(pesudo_outputs,unlabel_pseudo_targets_dict,idx):
    vaild_pesudo_outputs = {}
    for k,v in pesudo_outputs.items():
        if 'pred' in k:
            vaild_pesudo_outputs[k] = v[idx,:,:]
        elif 'aux_outputs_unlabel' in k:
            cache_list = []
            for sub_v_dict in v:
                cache_dict = {}
                cache_dict['pred_logits'] = sub_v_dict['pred_logits'][idx,:,:]
                cache_dict['pred_boxes'] = sub_v_dict['pred_boxes'][idx,:,:]
                cache_list.append(cache_dict)
            vaild_pesudo_outputs[k] = cache_list
        elif 'enc_outputs_unlabel' in k:
            cache_dict = {}
            cache_dict['pred_logits'] = v['pred_logits'][idx,:,:]
            cache_dict['pred_boxes'] = v['pred_boxes'][idx,:,:]
            vaild_pesudo_outputs[k] = cache_dict
        else:
            assert '不存在的输出结果'

    #处理伪标签格式，用于后续损失计算
    unlabel_pseudo_targets = []
    for k,v in unlabel_pseudo_targets_dict.items():
        unlabel_pseudo_targets.append(v)

    return vaild_pesudo_outputs,unlabel_pseudo_targets


from torchvision.ops.boxes import batched_nms

def rescale_pseudo_targets(unlabel_samples_img,unlabel_pseudo_targets,nms_th = 0.5):
    _b,_c,_h,_w = unlabel_samples_img.shape

    for k,v in unlabel_pseudo_targets.items():
        _h_real, _w_real = unlabel_pseudo_targets[k]['size'].cpu().numpy()

        unlabel_pseudo_targets[k]['boxes'] = box_ops.box_cxcywh_to_xyxy(unlabel_pseudo_targets[k]['boxes'])

        #（1）恢复为原图坐标
        unlabel_pseudo_targets[k]['boxes'][:,[0,2]] = unlabel_pseudo_targets[k]['boxes'][:,[0,2]] * _w
        unlabel_pseudo_targets[k]['boxes'][:,[1,3]] = unlabel_pseudo_targets[k]['boxes'][:,[1,3]] * _h
        #（2）NMS
        keep_inds = batched_nms(unlabel_pseudo_targets[k]['boxes'],
                                unlabel_pseudo_targets[k]['scores'],
                                unlabel_pseudo_targets[k]['labels'], nms_th)[:100]
        unlabel_pseudo_targets[k]['boxes'] = unlabel_pseudo_targets[k]['boxes'][keep_inds]
        unlabel_pseudo_targets[k]['scores'] = unlabel_pseudo_targets[k]['scores'][keep_inds]
        unlabel_pseudo_targets[k]['labels'] = unlabel_pseudo_targets[k]['labels'][keep_inds]
        #（3）比例缩放
        unlabel_pseudo_targets[k]['boxes'] = box_ops.box_xyxy_to_cxcywh(unlabel_pseudo_targets[k]['boxes'])
        unlabel_pseudo_targets[k]['boxes'][:,[0,2]] = unlabel_pseudo_targets[k]['boxes'][:,[0,2]]  / _w_real
        unlabel_pseudo_targets[k]['boxes'][:,[1,3]] = unlabel_pseudo_targets[k]['boxes'][:,[1,3]]  / _h_real
    return unlabel_pseudo_targets


#======================可视化DEBUG使用=============================

def Denormalize(img):
    # 这是归一化的 mean 和std
    channel_mean = torch.tensor([0.485, 0.456, 0.406])
    channel_std = torch.tensor([0.229, 0.224, 0.225])
    # 这是反归一化的 mean 和std
    MEAN = [-mean / std for mean, std in zip(channel_mean, channel_std)]
    STD = [1 / std for std in channel_std]
    # 归一化和反归一化生成器
    denormalizer = transforms.Normalize(mean=MEAN, std=STD)
    de_img = denormalizer(img)
    return de_img

def draw_img(img,unlabel_samples_img_strong_aug,data_dict):
    _h, _w = data_dict['size'].cpu().numpy()
    boxes = data_dict['boxes'].cpu().numpy()  #中心点+宽高
    boxes[:,[0,2]] *= _w
    boxes[:,[1,3]] *= _h
    img = img.copy()
    if unlabel_samples_img_strong_aug is not None:
        unlabel_samples_img_strong_aug = unlabel_samples_img_strong_aug.copy()
        for i,box in enumerate(boxes):
            cls = data_dict['labels'][i].cpu().numpy()
            x_c,y_c,w,h = [int(i) for i in box]
            x1,y1,x2,y2 = x_c - w//2,y_c - h // 2 ,x_c + w//2,y_c + h // 2
            img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            unlabel_samples_img_strong_aug = cv2.rectangle(unlabel_samples_img_strong_aug,(x1,y1),(x2,y2),(0,0,255),2)
       # cv2.imshow('a',img)
       # cv2.imshow('b',unlabel_samples_img_strong_aug)
        cv2.imwrite('a.jpg',img)
        cv2.imwrite('b.jpg',unlabel_samples_img_strong_aug)
        print('停止')
        time.sleep(5000000)



def show_pesudo_label_with_gt(unlabel_img_array,unlabel_pseudo_targets,unlabel_targets,idx_list,unlabel_samples_img_strong_aug_array,save_dir = './show_pseudo'):
    _make_dir(save_dir)

    for n,idx in enumerate(idx_list):
        unlabel_img = unlabel_img_array[idx].detach().cpu()  #根据索引找
        unlabel_samples_img_strong_aug = unlabel_samples_img_strong_aug_array[idx].detach().cpu()  #根据索引找
        unlabel_pseudo_target = unlabel_pseudo_targets[idx] #根据索引找
        unlabel_target = unlabel_targets[idx]  #根据索引找

        # 对图像进行反归一化
        unlabel_img = Denormalize(unlabel_img).numpy()
        unlabel_img *= 255.0
        unlabel_img = unlabel_img.transpose(1,2,0).astype(np.uint8)
        if unlabel_samples_img_strong_aug is not None:
            unlabel_samples_img_strong_aug = Denormalize(unlabel_samples_img_strong_aug).numpy()
            unlabel_samples_img_strong_aug *= 255.0
            unlabel_samples_img_strong_aug = unlabel_samples_img_strong_aug.transpose(1, 2, 0).astype(np.uint8)

        draw_img(unlabel_img,unlabel_samples_img_strong_aug,unlabel_pseudo_target)









