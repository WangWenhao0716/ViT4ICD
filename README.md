# ViT4ICD
[Neurocomputing 2024] The official implementation of "Vision Transformers are Active Learners for Image Copy Detection"

![image](https://github.com/WangWenhao0716/ViT4ICD/blob/main/demo.png)

The proposed method performs better than all the winning solutions in the [ISC21 descriptor track](https://www.drivendata.org/competitions/85/competition-image-similarity-2-final/page/407/).
![image](https://github.com/WangWenhao0716/ViT4ICD/blob/main/compare.png)

TL;DR: This paper develops a method for training Vision Transformer (ViT) for Image Copy Detection (ICD).

## Environment

Please install the packages according to the ``environment.yaml`` file in this directory. The minimum hardware requirement of our ViT4ICD is 4 V100 GPUs.

## Datasets
1. Download the training set (``train_v1_s3_all_bw.tar``) from [Google Drive](https://drive.google.com/file/d/1ztrvXFea8jHMRtNvpo22zdJ8sQB8iDGT/view?usp=sharing).

2. Also download all the files from [official website](https://sites.google.com/view/isc2021/dataset?authuser=0). We use the development query set.


## Train

0. Please download the pre-trained model from [here](https://dl.fbaipublicfiles.com/barlowtwins/ep1000_bs2048_lrw0.2_lrb0.0048_lambd0.0051/resnet50.pth), and save as ``logs/pretrained/resnet50_bar.pth`` in ``Stage 1``.

1. Please go to ``Stage 1``, and running
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_balance_cos_ema.py \
-ds train_v1_s3_all_bw -a resnet50_un --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 5 \
--lr 0.00035 --iters 8000 --epochs 25 \
--data-dir /path/to/train_v1_s3_all_bw/ \
--logs-dir logs/train_v1_s3_all_bw/50UN_two_losses_m0.6 \
--height 256 --width 256
```

By running:
```
import torch
mod = torch.load('logs/train_v1_s3_all_bw/50UN_two_losses_m0.6/checkpoint_24_ema.pth.tar',map_location='cpu')
torch.save(mod['state_dict'], 'logs/train_v1_s3_all_bw/50UN_two_losses_m0.6/stage_1.pth.tar')
```
We will get ```stage_1.pth.tar```. Please move it to ``Stage 23/``.

2. Please go to ``Stage 23``, and running
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_balance_cos_ema_com_bw_forimage.py \
-ds train_v1_s3_all_bw -a vit_base --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 5 \
--lr 0.00035 --iters 8000 --epochs 25 \
--data-dir /path/to/train_v1_s3_all_bw/ \
--logs-dir logs/train_v1_s3_all_bw/vit_two_losses_com_L2_norm_100_all_same_forimage \
--height 224 --width 224
```
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_balance_cos_ema_com_tune_bw_forimage.py \
-ds train_v1_s3_all_bw -a vit_base --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 10 \
--lr 0.00035 --iters 8000 --epochs 10 \
--data-dir /path/to/train_v1_s3_all_bw/ \
--logs-dir logs/train_v1_s3_all_bw/vit_two_losses_com_L2_norm_100_all_same_tune_forimage \
--height 224 --width 224 \
--resume logs/train_v1_s3_all_bw/vit_two_losses_com_L2_norm_100_all_same_forimage/checkpoint_24_ema.pth.tar
```

By running:
```
import torch
mod = torch.load('logs/train_v1_s3_all_bw/vit_two_losses_com_L2_norm_100_all_same_tune_forimage/checkpoint_9_ema.pth.tar',map_location='cpu')
torch.save(mod['state_dict'], 'logs/train_v1_s3_all_bw/vit_two_losses_com_L2_norm_100_all_same_tune_forimage/stage_2.pth.tar')
```
We will get ```stage_2.pth.tar```.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_single_source_gem_coslr_wb_balance_cos_ema_com_tune_bw_gt_ng_forimage.py \
-ds train_v1_s3_all -a vit_base --margin 0.0 \
--num-instances 4 -b 128 -j 8 --warmup-step 10 \
--lr 0.00035 --iters 8000 --epochs 10 \
--data-dir /path/to/train_v1_s3_all_bw/ \
--logs-dir logs/train_v1_s3_all_bw/vit_two_losses_com_L2_norm_100_all_tune_bw_gt_ng_1_forimage \
--height 224 --width 224 \
--resume logs/train_v1_s3_all_bw/vit_two_losses_com_L2_norm_100_all_same_tune_forimage/checkpoint_9_ema.pth.tar
```

By running:
```
import torch
mod = torch.load('logs/train_v1_s3_all_bw/vit_two_losses_com_L2_norm_100_all_tune_bw_gt_ng_1_forimage/checkpoint_9_ema.pth.tar',map_location='cpu')
torch.save(mod['state_dict'], 'logs/train_v1_s3_all_bw/vit_two_losses_com_L2_norm_100_all_tune_bw_gt_ng_1_forimage/stage_3.pth.tar')
```
We will get ```stage_3.pth.tar```.

## Test
Please go to the ``extract_features`` folder, and make the directory ``mkdir ./feature/vit_stage3``.
You can use the trained model ```stage_3.pth.tar```, or directly download it from [here](https://drive.google.com/file/d/1_CL28ZVvNOxOhUA3eJMLqw4shj6KvmXj/view?usp=sharing).

1. Extract reference feature:
```
CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
      --image_dir /path/to/reference_images/ \
      --o ./feature/vit_stage3/reference_v1.hdf5 \
      --model vit_base  --GeM_p 3 --bw \
      --checkpoint stage_3.pth.tar --imsize 224
```
2. Extract training features:
```
CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
      --image_dir /path/to/training_images/ \
      --o ./feature/vit_stage3/training_v1.hdf5 \
      --model vit_base  --GeM_p 3 --bw \
      --checkpoint stage_3.pth.tar --imsize 224 
```
3. Extract query featuress:
```
CUDA_VISIBLE_DEVICES=0 python extract_feature.py \
      --image_dir /path/to/queries/ \
      --o ./feature/vit_stage3/query_v1.hdf5 \
      --model vit_base  --GeM_p 3 --bw \
      --checkpoint stage_3.pth.tar --imsize 224
```
4. Score normalization:
```
CUDA_VISIBLE_DEVICES=0 python score_normalization.py \
    --query_descs ./feature/vit_stage3/query_0_v1.hdf5\
    --db_descs ./feature/vit_stage3/reference_{0..19}_v1.hdf5 \
    --train_descs ./feature/vit_stage3/training_{0..19}_v1.hdf5 \
    --factor 2 --n 10 \
    --o ./feature/vit_stage3/predictions_v1.csv \
    --reduction avg --max_results 500000
```
5. Get the final predictions:
```
import pandas as pd
df = pd.read_csv('./feature/vit_stage3/predictions_v1.csv')
df_1 = df[df['query_id']>'Q25000']
df_1.to_csv('./feature/vit_stage3/predictions_v1_after.csv', index=None)
```
and then
```
python compute_metrics.py \
--preds_filepath ./feature/vit_stage3/predictions_v1_after.csv \
--gt_filepath dev_ground_truth_after.csv
```
This should give
```
Track 1 results of 259570 predictions (5008 GT matches)
Average Precision: 0.78596
Recall at P90    : 0.73602
Threshold at P90 : -0.114972
Recall at rank 1:  0.82488
Recall at rank 10: 0.84265
```

## Citation
```
@article{tan2024vision,
title = {Vision transformers are active learners for image copy detection},
journal = {Neurocomputing},
volume = {587},
pages = {127687},
year = {2024},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2024.127687},
url = {https://www.sciencedirect.com/science/article/pii/S0925231224004582},
author = {Zhentao Tan and Wenhao Wang and Caifeng Shan},
}
```
