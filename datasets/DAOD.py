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
        'cityscapes': {
            'train_img': root / 'cityscapes/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_train.json',
            'val_img': root / 'cityscapes/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_val.json',
        },
        'cityscapes_caronly': {
            'train_img': root / 'cityscapes/leftImg8bit/train',
            'train_anno': root / 'cityscapes/annotations/cityscapes_caronly_train.json',
            'val_img': root / 'cityscapes/leftImg8bit/val',
            'val_anno': root / 'cityscapes/annotations/cityscapes_caronly_val.json',
        },
        'foggy_cityscapes': {
            'train_img': root / 'cityscapes/leftImg8bit_foggy/train',
            'train_anno': root / 'cityscapes/annotations/foggy_cityscapes_train.json',
            'val_img': root / 'cityscapes/leftImg8bit_foggy/val',
            'val_anno': root / 'cityscapes/annotations/foggy_cityscapes_val.json',
        },
        'sim10k': {
            'train_img': root / 'sim10k/VOC2012/JPEGImages',
            'train_anno': root / 'sim10k/annotations/sim10k_caronly.json',
        },
        'bdd_daytime': {
            'train_img': root / 'bdd_daytime/train',
            'train_anno': root / 'bdd_daytime/annotations/bdd_daytime_train.json',
            'val_img': root / 'bdd_daytime/val',
            'val_anno': root / 'bdd_daytime/annotations/bdd_daytime_val.json',
        },
        #---------添加遥感相关-------------------
        'xView3c': {
            'train_img': '/data/jianhonghan/遥感域适应数据集/xView/crop_800_100_3c/train_images',
            'train_anno': '/data/jianhonghan/遥感域适应数据集/xView/crop_800_100_3c/coco_annotations/train_3c.json',
            'val_img': '',
            'val_anno': '',
        },
        'DOTA3c': {
            'train_img': '/data/jianhonghan/遥感域适应数据集/DOTA1.0/crop_800_100_3c/train/train_images',
            'train_anno': '/data/jianhonghan/遥感域适应数据集/DOTA1.0/crop_800_100_3c/annotations/train_3c.json',
            'val_img': '/data/jianhonghan/遥感域适应数据集/DOTA1.0/crop_800_100_3c/val/train_images',
            'val_anno': '/data/jianhonghan/遥感域适应数据集/DOTA1.0/crop_800_100_3c/annotations/val_3c.json',
        },

        'optical': {
                      'train_img': '/data/jianhonghan/遥感域适应数据集/天大无人机数据集_day_night/train/optical/day_img',
                      'train_anno': '/data/jianhonghan/遥感域适应数据集/天大无人机数据集_day_night/train/optical/train.json',
                      'val_img': '/data/jianhonghan/遥感域适应数据集/天大无人机数据集_day_night/test/optical/day_img',
                      'val_anno': '/data/jianhonghan/遥感域适应数据集/天大无人机数据集_day_night/test/optical/test.json',
                  },
        'infrared': {
            'train_img': '/data/jianhonghan/遥感域适应数据集/天大无人机数据集_day_night/train/infrared/imgr',
            'train_anno': '/data/jianhonghan/遥感域适应数据集/天大无人机数据集_day_night/train/infrared/train_r.json',
            'val_img': '/data/jianhonghan/遥感域适应数据集/天大无人机数据集_day_night/test/infrared/imgr',
            'val_anno': '/data/jianhonghan/遥感域适应数据集/天大无人机数据集_day_night/test/infrared/test_r.json',
        },

        'UCASAOD': {
            'train_img': '/data/jianhonghan/遥感域适应数据集/UCAS-AOD/CAR/train/images',
            'train_anno': '/data/jianhonghan/遥感域适应数据集/UCAS-AOD/CAR/annotations/train.json',
            'val_img': '/data/jianhonghan/遥感域适应数据集/UCAS-AOD/CAR/val/images',
            'val_anno': '/data/jianhonghan/遥感域适应数据集/UCAS-AOD/CAR/annotations/val.json',
        },
        'CARPK': {
            'train_img': '/data/jianhonghan/遥感域适应数据集/CARPK/coco/train/images',
            'train_anno': '/data/jianhonghan/遥感域适应数据集/CARPK/coco/train/train.json',
            'val_img': '/data/jianhonghan/遥感域适应数据集/CARPK/coco/test/images',
            'val_anno': '/data/jianhonghan/遥感域适应数据集/CARPK/coco/test/test.json',
        },
        'HRRSD': {
            'train_img': '/data/jianhonghan/遥感域适应数据集/HRRSD/imgs',
            'train_anno': '/data/jianhonghan/遥感域适应数据集/HRRSD/train.json',

        },
        'SSDD': {
            'train_img': '/data/jianhonghan/遥感域适应数据集/SSDD/images/train',
            'train_anno': '/data/jianhonghan/遥感域适应数据集/SSDD/annotations/train.json',
            'val_img': '/data/jianhonghan/遥感域适应数据集/SSDD/images/test',
            'val_anno': '/data/jianhonghan/遥感域适应数据集/SSDD/annotations/test.json',
        },
        'clear': {
            'train_img': '/data/jianhonghan/遥感域适应数据集/项目数据集/domain-transfer/clear/images',
            'train_anno': '/data/jianhonghan/遥感域适应数据集/项目数据集/domain-transfer/clear/annotations/clear_train.json',
            'val_img': '',
            'val_anno': '',
        },
        'cloudy': {
            'train_img': '/data/jianhonghan/遥感域适应数据集/项目数据集/domain-transfer/cloudy_split/train/images',
            'train_anno': '/data/jianhonghan/遥感域适应数据集/项目数据集/domain-transfer/cloudy_split/train/annotations/cloudy_train.json',
            'val_img': '/data/jianhonghan/遥感域适应数据集/项目数据集/domain-transfer/cloudy_split/test/images',
            'val_anno': '/data/jianhonghan/遥感域适应数据集/项目数据集/domain-transfer/cloudy_split/test/annotations/cloudy_test.json',
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
