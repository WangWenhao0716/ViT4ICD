# ViT4ICD
[In Submission] The official implementation of "Vision Transformers are Active Learners for Image Copy Detection"

![image](https://github.com/WangWenhao0716/ViT4ICD/blob/main/demo.png)

TL;DR: This paper develops a method for training Vision Transformer (ViT) for Image Copy Detection (ICD).

## Environment

Please install the packages according to the ``environment.yaml`` file in this directory. The minimum hardware requirement of our ViT4ICD is 4 V100 GPUs.

## Datasets
1. Download the training set (``train_v1_s3_all_bw.tar``) from [Google Drive]().

2. Also download the test set, including the reference set and query set from [official website](https://sites.google.com/view/isc2021/dataset?authuser=0).


## Train

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


## Citation
```
@inproceedings{
    tan2024vision,
    title={Vision Transformers are Active Learners for Image Copy Detection},
    author={Tan Zhentao and Wenhao Wang and Caifeng Shan and Jungong Han},
    booktitle={In submission},
    year={2024},
}
```
