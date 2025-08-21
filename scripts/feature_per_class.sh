#!/usr/bin/env bash

# 数据集与样本数（SHOT=-1 代表全量）
DATASET='cub'
SHOT=5

# 已训练权重（用于解析 LoRA 与文本嵌入路径）
FINETUNED_CKPT="/data/st/Diff-Mix/ckpts/cub/shot5_lora_rank10_snr"

# 采样/特征策略
# 可选：['diff-mix', 'diff-aug', 'diff-gen', 'real-mix', 'real-aug', 'real-gen', 'ti-aug', 'ti-mix']
SAMPLE_STRATEGY='diff-aug'
RESOLUTION=512

# 特征提取说明：
# 保存四类特征：
# 1. 像素的CLIP特征（对像素图直接用CLIP编码）
# 2. 文本的CLIP特征（从checkpoint中预计算的文本嵌入特征）
# 3. 上采样倒数第二层特征（UNet上采样块倒数第二层的激活特征）
# 4. 上采样最后一层特征（UNet上采样块最后一层的激活特征）

# GPU 设置（将可见设备映射后，本脚本用 cuda:0）
export CUDA_VISIBLE_DEVICES=0
GPU_ID=0

python /data/st/Diff-Mix/featrue_per_class_txt.py \
  --dataset=$DATASET \
  --task=vanilla \
  --examples_per_class=$SHOT \
  --resolution=$RESOLUTION \
  --sample_strategy=$SAMPLE_STRATEGY \
  --model_path='runwayml/stable-diffusion-v1-5' \
  --finetuned_ckpt=$FINETUNED_CKPT \
  --guidance_scale=7.5 \
  --gpu_id=$GPU_ID \
  --seed=42 \
  --batch_size=4 \
  --dataloader_num_workers=0



