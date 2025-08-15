#dataset    (dog         cub        car         aircraft      flower      pet      food     chest     caltech   pascal  )
#lr(resnet) (0.001       0.05       0.10        0.10          0.05        0.01     0.01     0.01      0.01      0.01    )
#lr(vit)    (0.00001     0.001      0.001       0.001         0.0005      0.0001   0.00005  0.00005   0.00005   0.00005 )

GPU=2
DATASET="cub"
SHOT=5
# "shot{args.examples_per_class}_{args.sample_strategy}_{args.strength_strategy}_{args.aug_strength}"
SYNDATA_DIR="/data/st/Diff-Mix/outputs/aug_samples_mmd_sim/cub/shot5_diff-aug_fixed_0.8" # shot-1 denotes full shot
SYNDATA_P=0.1
GAMMA=0.8

python downstream_tasks/train_hub.py \
    --dataset $DATASET \
    --syndata_dir $SYNDATA_DIR \
    --syndata_p $SYNDATA_P \
    --model "resnet50" \
    --gamma $GAMMA \
    --examples_per_class $SHOT \
    --gpu $GPU \
    --amp 2 \
    --note $(date +%m%d%H%M) \
    --group_note "5shotmmd_aug_mmd" \
    --nepoch 120 \
    --res_mode 224 \
    --lr 0.05 \
    --seed 0 \
    --weight_decay 0.0005 