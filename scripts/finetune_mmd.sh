#!/bin/bash

# Training script for train_lora_mmd.py with MMD distribution matching loss
MODEL_NAME="runwayml/stable-diffusion-v1-5"
DATASET='cub'
SHOT=5 # set -1 for full shot
OUTPUT_DIR="ckpts/${DATASET}/shot${SHOT}_lora_rank10_mmd_only"   #意思是秩为10的LoRA模型

accelerate launch --mixed_precision='fp16' --main_process_port 29508 \
    train_lora_mmd.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET \
    --resolution=224 \
    --random_flip \
    --max_train_steps=5000 \
    --num_train_epochs=10 \
    --checkpointing_steps=1000 \
    --learning_rate=5e-05 \
    --ti_learning_rate=5e-05 \
    --lr_scheduler='constant' \
     --local_files_only \
    --lr_warmup_steps=0 \
    --seed=42 \
    --rank=10 \
    --examples_per_class $SHOT \
    --train_batch_size=16 \
    --validation_prompt="a photo of a bird" \
    --num_validation_images=2 \
    --output_dir=$OUTPUT_DIR \
    --report_to='wandb' \
    