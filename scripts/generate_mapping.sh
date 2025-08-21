#!/usr/bin/env bash

# 数据集与样本数（SHOT=-1 代表全量）
DATASET='cub'
SHOT=5

# 映射参数
MODE='aug'  # 可选: 'aug' 或 'mix'
MULTIPLIER=5 # 扩增倍数

# 随机种子（确保可复现）
SEED=42

# 输出目录
OUTPUT_DIR="outputs/mappings"

# GPU 设置（将可见设备映射后，本脚本用 cuda:0）
export CUDA_VISIBLE_DEVICES=0

echo "=== 源目标映射生成器 ==="
echo "数据集: $DATASET"
echo "每类样本数: $SHOT"
echo "映射模式: $MODE"
echo "扩增倍数: $MULTIPLIER"
echo "随机种子: $SEED"
echo "输出目录: $OUTPUT_DIR"
echo "=========================="

python /data/st/Diff-Mix/source2target.py \
  --dataset=$DATASET \
  --task=vanilla \
  --examples_per_class=$SHOT \
  --resolution=512 \
  --seed=$SEED \
  --mode=$MODE \
  --multiplier=$MULTIPLIER \
  --output_dir=$OUTPUT_DIR
