# ----------------------------------------------
# Created by Wei-Jie Huang
# ----------------------------------------------


from pathlib import Path
from torch.utils.data import Dataset
from datasets.coco import CocoDetection, make_coco_transforms,make_coco_strong_transforms
from util.misc import get_local_rank, get_local_size, nested_tensor_from_tensor_list

import random

def get_paths(root):
    root = Path(root)

    return {
        #---------添加遥感相关-------------------
        'xView': {
            'train_img': '',
            'train_anno': '',
        },
        'DOTA': {
            'train_img': '',
            'train_anno': '',
            'val_img': '',
            'val_anno': '',
        },
        'UCASAOD': {
            'train_img': '',
            'train_anno': '',
            'val_img': '',
            'val_anno': '',
        },
        'CARPK': {
            'train_img': '',
            'train_anno': '',
            'val_img': '',
            'val_anno': '',
        },

        'HRRSD': {
            'train_img': '',
            'train_anno': '',
        },
        'SSDD': {
            'train_img': '',
            'train_anno': '',
            'val_img': '',
            'val_anno': '',
        },
    }


class DADataset(Dataset):
    def __init__(self, source_img_folder, source_ann_file, target_img_folder, target_ann_file,
                 transforms,transforms_strong_aug, return_masks, cache_mode=False, local_rank=0, local_size=1):
        self.source = CocoDetection(
            img_folder=source_img_folder,
            ann_file=source_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )
        self.target = CocoDetection(
            img_folder=target_img_folder,
            ann_file=target_ann_file,
            transforms=transforms,
            strong_transforms=transforms_strong_aug,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )

    def __len__(self):
        return max(len(self.source), len(self.target))

    def __getitem__(self, idx):
        source_img,_,source_target = self.source[idx % len(self.source)]
        target_img,target_img_strong_aug ,target_label = self.target[idx % len(self.target)]
        return source_img, target_img, source_target,target_label,target_img_strong_aug


def collate_fn(batch):
    source_imgs, target_imgs, source_targets,target_label,target_img_strong_aug = list(zip(*batch))
    samples = nested_tensor_from_tensor_list(source_imgs + target_imgs)

    #判断是否经过强增广处理
    if target_img_strong_aug[0] is not None:
        samples_strong_aug = nested_tensor_from_tensor_list(source_imgs + target_img_strong_aug) #与弱增广保持相同处理，以保证维度统一，方便后续操作
    else:
        samples_strong_aug = None
    return samples, source_targets,target_label,samples_strong_aug


def build(image_set, cfg,strong_aug):
    paths = get_paths(cfg.DATASET.COCO_PATH)
    source_domain, target_domain = cfg.DATASET.DATASET_FILE.split('_to_')
    if image_set == 'val':
        return CocoDetection(
            img_folder=paths[target_domain]['val_img'],
            ann_file=paths[target_domain]['val_anno'],
            transforms=make_coco_transforms(image_set),
            return_masks=cfg.MODEL.MASKS,
            cache_mode=cfg.CACHE_MODE,
            local_rank=get_local_rank(),
            local_size=get_local_size()
        )
    elif image_set == 'train':
        if cfg.DATASET.DA_MODE == 'source_only':
            return CocoDetection(
                img_folder=paths[source_domain]['train_img'],
                ann_file=paths[source_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        elif cfg.DATASET.DA_MODE == 'oracle':
            return CocoDetection(
                img_folder=paths[target_domain]['train_img'],
                ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'uda':
            if strong_aug: #半监督时  学生训练使用强增广
                return DADataset(
                    source_img_folder=paths[source_domain]['train_img'],
                    source_ann_file=paths[source_domain]['train_anno'],
                    target_img_folder=paths[target_domain]['train_img'],
                    target_ann_file=paths[target_domain]['train_anno'],
                    transforms=make_coco_transforms(image_set),
                    transforms_strong_aug = make_coco_strong_transforms(image_set), #半监督方法使用
                    return_masks=cfg.MODEL.MASKS,
                    cache_mode=cfg.CACHE_MODE,
                    local_rank=get_local_rank(),
                    local_size=get_local_size()
                )
            else:
                return DADataset(
                    source_img_folder=paths[source_domain]['train_img'],
                    source_ann_file=paths[source_domain]['train_anno'],
                    target_img_folder=paths[target_domain]['train_img'],
                    target_ann_file=paths[target_domain]['train_anno'],
                    transforms=make_coco_transforms(image_set),
                    transforms_strong_aug = None, #半监督方法使用
                    return_masks=cfg.MODEL.MASKS,
                    cache_mode=cfg.CACHE_MODE,
                    local_rank=get_local_rank(),
                    local_size=get_local_size()
                )
        else:
            raise ValueError(f'Unknown argument cfg.DATASET.DA_MODE {cfg.DATASET.DA_MODE}')
    raise ValueError(f'unknown image set {image_set}')
