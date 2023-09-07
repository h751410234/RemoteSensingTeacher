# Remote Sensing Teacher: Cross-Domain Detection Transformer with Frequency-Enhanced Feature Alignment in Remote Sensing Imagery
By Jianhong Han etc.

This repository contains the implementation accompanying our paper Remote Sensing Teacher: Cross-Domain Detection Transformer with Frequency-Enhanced Feature Alignment in Remote Sensing Imagery.

If you find it helpful for your research, please consider citing:

```
@inproceedings{XXX,
  title     = {Remote Sensing Teacher: Cross-Domain Detection Transformer with Frequency-Enhanced Feature Alignment in Remote Sensing Imagery},
  author    = {Jianhong Han etc.},
  booktitle = {XXX},
  year      = {2023},
}
```

![](/figs/overview.png)

## Acknowledgment
This implementation is bulit upon [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR/) and [AQT](https://github.com/weii41392/AQT).

## Installation
Please refer to the instructions [here](https://github.com/fundamentalvision/Deformable-DETR/#installation). We leave our system information for reference.

* OS: Ubuntu 16.04
* Python: 3.11
* CUDA: 11.8
* PyTorch: 2.0.1
* torchvision: 0.15.2

## Dataset Preparation
Please construct the datasets following these steps:

- Download the datasets from their sources.

- Convert the annotation files into COCO-format annotations.

- Modify the dataset path setting within the script [DAOD.py](./datasets/DAOD.py)

```
'dateset's name': {
    'train_img'  : '',  #train image dir
    'train_anno' : '',  #train coco format json file
    'val_img'    : '',  #val image dir
    'val_anno'   : '',  #val coco format json file
},
```
- Add domain adaptation direction within the script [__init__.py](./datasets/__init__.py). During training, the domain adaptation direction will be automatically parsed and corresponding data will be loaded. In our paper, we provide four adaptation directions for remote sensing scenarios.
```
DAOD_dataset = [
    'xView_to_DOTA',      #dateset's name1_to_dateset's name2
    'UCASAOD_to_CARPK',
    'CARPK_to_UCASAOD',
    'HRRSD_to_SSDD',
]
```

## Training / Evaluation
We provide training script on single node as follows.
- Training with single GPU
```
python main.py --config_file {CONFIG_FILE}
```
- Training with Multi-GPU
```
GPUS_PER_NODE={NUM_GPUS} ./tools/run_dist_launch.sh {NUM_GPUS} python main.py --config_file {CONFIG_FILE}
```

We provide evaluation script to evaluate pre-trained model:
- Evaluation Model.
```
python evaluation.py --config_file {CONFIG_FILE} --opts EVAL True RESUME {CHECKPOINT_FILE}
```
- Evaluation EMA Model.
```
python evaluation.py --config_file {CONFIG_FILE} --opts EVAL True SSOD.RESUME_EMA {CHECKPOINT_FILE}
```

## Pre-trained models
We provide specific experimental configurations and pre-trained models to facilitate the reproduction of our results. 
You can learn the details of Remote Sensing Teacher through the paper, and please cite our papers if the code is useful for your papers. Thank you!
- **xView to DOTA**: [cfg](./configs/r50_uda_xView2DOTA_b16.yaml), [model]()
- **UCAS-AOD to CARPK**: [cfg](./configs/r50_uda_UCASAOD2CARPK_b16.yaml), [model]()
- **CARPK to UCAS-AOD**: [cfg](./configs/r50_uda_CARPK2UCASAOD_b16.yaml), [model]()
- **HRRSD to SSDD**: [cfg](./configs/r50_uda_HRRSD2SSDD_b16.yaml), [model]()

## Result Visualization 

![](/figs/detect_result.png)

## Reference
https://github.com/fundamentalvision/Deformable-DETR  
https://github.com/weii41392/AQT