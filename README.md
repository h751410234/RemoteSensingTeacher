# AQT: Adversarial Query Transformers for Domain Adaptive Object Detection
By Wei-Jie Huang, Yu-Lin Lu, Shih-Yao Lin, Yusheng Xie, and Yen-Yu Lin.

This repository contains the implementation accompanying our paper [AQT: Adversarial Query Transformers for Domain Adaptive Object Detection](http://vllab.cs.nctu.edu.tw/images/paper/ijcai-huang22.pdf). This work was accepted to [IJCAI-ECAI 2022](https://ijcai-22.org/).

If you find it helpful for your research, please consider citing:

```
@inproceedings{huang2022aqt,
  title     = {AQT: Adversarial Query Transformers for Domain Adaptive Object Detection},
  author    = {Huang, Wei-Jie and Lu, Yu-Lin and Lin, Shih-Yao and Xie, Yusheng and Lin, Yen-Yu},
  booktitle = {International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2022},
}
```

![](/figs/overview.png)

## Acknowledgment
This implementation is bulit upon [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR/).

## Installation
Please refer to the instructions [here](https://github.com/fundamentalvision/Deformable-DETR/#installation). We leave our system information for reference.

* OS: Ubuntu 16.04
* Python: 3.6
* CUDA: 9.2
* cuDNN: 7
* PyTorch: 1.5.1
* torchvision: 0.6.1

## Dataset Preparation
Please construct the datasets following these steps:

- Download the datasets from their sources:
  - **Cityscapes / Foggy Cityscapes**: download `gtFine_trainvaltest.zip` (labels), `leftImg8bit_trainvaltest.zip` (Cityscapes images), `leftImg8bit_trainvaltest_foggy.zip` (Foggy Cityscapes images) from the official [website](https://www.cityscapes-dataset.com/).
  - **Sim10k**: download `images` and `annotations` of 10k subset from the official [website](https://fcav.engin.umich.edu/projects/driving-in-the-matrix).
  - **BDD100k**: download `100k Images` and `Labels` from the official [website](https://bdd-data.berkeley.edu/).

- Convert the annotation files into COCO-format annotations (you can build them following the annotation conversion [script](util/anno_convert.py)).

- Prepare a directory `datasets` as follows (or you can modify the setting [here](/datasets/DAOD.py))
```
datasets/
├─ bdd_daytime/
│  ├─ annotations/
│  ├─ train/
│  ├─ val/
├─ cityscapes/
│  ├─ annotations/
│  ├─ leftImg8bit/
│  |  ├─ train/
│  |  ├─ val/
│  ├─ leftImg8bit_foggy/
│  |  ├─ train/
│  |  ├─ val/
├─ sim10k/
│  ├─ annotations/
│  ├─ VOC2012/
```

## Training / Evaluation
We provide training script on single node as follows, please refer to the instructions [here](https://github.com/fundamentalvision/Deformable-DETR/#training) for other settings.
```
GPUS_PER_NODE={NUM_GPUS} ./tools/run_dist_launch.sh {NUM_GPUS} python main.py --config_file {CONFIG_FILE}
```

We use [yacs](https://github.com/rbgirshick/yacs) for configuration. The configuration files are in `./configs`. If you want to override configuration using command line arguments, please consider the following script, which performs evaluation on a pre-trained model:
```
GPUS_PER_NODE={NUM_GPUS} ./tools/run_dist_launch.sh {NUM_GPUS} python main.py --config_file {CONFIG_FILE} --opts EVAL True RESUME {CHECKPOINT_FILE}
```

## Pre-trained models

- **Cityscapes to Foggy Cityscapes**: [cfg](./configs/r50_uda_c2fc.yaml), [model](https://github.com/weii41392/AQT/releases/download/v0.1/cityscapes_to_foggy_cityscapes.pth)
- **Sim10k to Cityscapes**: [cfg](./configs/r50_uda_s2c.yaml), [model](https://github.com/weii41392/AQT/releases/download/v0.1/sim10k_to_cityscapes.pth)
- **Cityscapes to BDD100k daytime**: [cfg](./configs/r50_uda_c2b.yaml), [model](https://github.com/weii41392/AQT/releases/download/v0.1/cityscapes_to_bdd100k_daytime.pth)
