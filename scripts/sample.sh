
DATASET='cub'
# set -1 for full shot
SHOT=5 
FINETUNED_CKPT="/data/st/Diff-Mix/ckpts/cub/shot5_lora_rank10_snr"

# 映射文件路径
MAPPING_FILE="/data/st/Diff-Mix/outputs/mappings/cub_shot5_mix_mult5_seed42.csv"

# ['diff-mix', 'diff-aug', 'diff-gen', 'real-mix', 'real-aug', 'real-gen', 'ti_mix', 'ti_aug']
# ['diff-mix', 'diff-aug', 'diff-gen', 'real-mix', 'real-aug', 'real-gen', 'ti_mix', 'ti_aug']
SAMPLE_STRATEGY='diff-gen' 
STRENGTH=0.8
# ['fixed', 'uniform']. 'fixed': use fixed $STRENGTH, 'uniform': sample from [0.3, 0.5, 0.7, 0.9]
STRENGTH_STRATEGY='fixed' 
# spawn 2 processes
GPU_IDS=(0 1 2 3) 
export CUDA_VISIBLE_DEVICES=1,2,3,4
python /data/st/Diff-Mix/scripts/sample_mp1.py \
--model_path='runwayml/stable-diffusion-v1-5' \
--output_root='outputs2/aug_samples1' \
--dataset=$DATASET \
--finetuned_ckpt=$FINETUNED_CKPT \
--mapping_file=$MAPPING_FILE \
--strength_strategy=$STRENGTH_STRATEGY \
--sample_strategy=$SAMPLE_STRATEGY \
--examples_per_class=$SHOT \
--resolution=512 \
--batch_size=10 \
--aug_strength=$STRENGTH \
--guidance_scale=7.5 \
--gpu_ids ${GPU_IDS[@]} \
--ddim_eta=0.0 \
--num_inference_steps=25

