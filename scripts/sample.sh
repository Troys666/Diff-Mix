
DATASET='cub'
# set -1 for full shot
SHOT=5 
FINETUNED_CKPT="/data/st/Diff-Mix/ckpts/cub/shot5_lora_rank10_mmd_sim"
# ['diff-mix', 'diff-aug', 'diff-gen', 'real-mix', 'real-aug', 'real-gen', 'ti_mix', 'ti_aug']
SAMPLE_STRATEGY='diff-aug' 
STRENGTH=0.8
# ['fixed', 'uniform']. 'fixed': use fixed $STRENGTH, 'uniform': sample from [0.3, 0.5, 0.7, 0.9]
STRENGTH_STRATEGY='fixed' 
# expand the dataset by 5 times
MULTIPLIER=5 
# spwan 4 processes
GPU_IDS=(0 1)  # 相对索引：0对应实际GPU1，1对应实际GPU2

# 设置只有指定的GPU可见，这样cuda:0对应实际的GPU 1，cuda:1对应实际的GPU 2,真的巨聪明的改发
export CUDA_VISIBLE_DEVICES=1,2

python  scripts/sample_mp.py \
--model_path='runwayml/stable-diffusion-v1-5' \
--output_root='outputs/aug_samples_mmd_sim' \
--dataset=$DATASET \
--finetuned_ckpt=$FINETUNED_CKPT \
--syn_dataset_mulitiplier=$MULTIPLIER \
--strength_strategy=$STRENGTH_STRATEGY \
--sample_strategy=$SAMPLE_STRATEGY \
--examples_per_class=$SHOT \
--resolution=512 \
--batch_size=4 \
--aug_strength=0.8 \
--gpu_ids ${GPU_IDS[@]}


