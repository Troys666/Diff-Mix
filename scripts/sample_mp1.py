import argparse
import os
import random
import re
import sys
import time
import csv

import numpy as np
import pandas as pd
import torch
import yaml

os.environ["CURL_CA_BUNDLE"] = ""

from collections import defaultdict
from multiprocessing import Process, Queue
from queue import Empty

from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from augmentation import AUGMENT_METHODS
from dataset import DATASET_NAME_MAPPING, IMBALANCE_DATASET_NAME_MAPPING
from utils.misc import parse_finetuned_ckpt


def check_args_valid(args):

    if args.sample_strategy == "real-gen":
        args.lora_path = None
        args.embed_path = None
        args.aug_strength = 0.8
    elif args.sample_strategy == "diff-gen":
        lora_path, embed_path = parse_finetuned_ckpt(args.finetuned_ckpt)#这里去嵌入token
        args.lora_path = lora_path
        args.embed_path = embed_path
        args.aug_strength = 0.8
    elif args.sample_strategy in ["real-aug", "real-mix"]:
        args.lora_path = None
        args.embed_path = None
    elif args.sample_strategy in ["diff-aug", "diff-mix"]:
        lora_path, embed_path = parse_finetuned_ckpt(args.finetuned_ckpt)
        args.lora_path = lora_path
        args.embed_path = embed_path


def load_mapping_file(mapping_file: str) -> list:
    """加载映射文件"""
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"映射文件不存在: {mapping_file}")
    
    mappings = []
    with open(mapping_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mappings.append({
                'pair_id': int(row['pair_id']),
                'source_idx': int(row['source_idx']),
                'source_class': int(row['source_class']),
                'source_name': row['source_name'],
                'target_idx': int(row['target_idx']),
                'target_class': int(row['target_class']),
                'target_name': row['target_name']
            })
    
    print(f"加载了 {len(mappings)} 个映射对")
    return mappings


def sample_func(args, in_queue, out_queue, gpu_id, process_id):

    os.environ["CURL_CA_BUNDLE"] = ""

    random.seed(args.seed + process_id)
    np.random.seed(args.seed + process_id)
    torch.manual_seed(args.seed + process_id)

    if args.task == "imbalanced":
        train_dataset = IMBALANCE_DATASET_NAME_MAPPING[args.dataset](
            split="train",
            seed=args.seed,
            resolution=args.resolution,
            imbalance_factor=args.imbalance_factor,
        )
    else:
        train_dataset = DATASET_NAME_MAPPING[args.dataset](
            split="train",
            seed=args.seed,  # dataset seed is fixed for all processes
            examples_per_class=args.examples_per_class,
            resolution=args.resolution,
        )

    model = AUGMENT_METHODS[args.sample_strategy](
        model_path=args.model_path,
        embed_path=args.embed_path,
        lora_path=args.lora_path,
        prompt=args.prompt,
        guidance_scale=args.guidance_scale,
        device=f"cuda:{gpu_id}",
        ddim_eta=args.ddim_eta,
        num_inference_steps=args.num_inference_steps,
    )

    while True:
        try:
            # 从队列中获取任务包
            task_package = in_queue.get(timeout=1)
        except Empty:
            print("queue empty, exit")
            break
        
        # 解包任务包
        target_class = task_package['target_class']
        target_idx = task_package['target_idx']
        target_name = task_package['target_name']
        mappings = task_package['mappings']
        
        # 获取target元数据
        target_metadata = train_dataset.get_metadata_by_idx(target_idx)
        
        # 收集所有source图像和保存路径
        source_images = []
        save_paths = []
        
        for mapping in mappings:
            # 获取源图像
            source_idx = mapping['source_idx']
            source_images.append(train_dataset.get_image_by_idx(source_idx))
            
            # 构建保存路径
            source_name = mapping['source_name'].replace(" ", "_").replace("/", "_")
            save_name = os.path.join(
                source_name, f"{target_name}-{mapping['pair_id']:06d}-{args.aug_strength}.png"
            )
            save_paths.append(os.path.join(args.output_path, "data", save_name))

        # 检查是否需要跳过
        if os.path.exists(save_paths[0]):
            print(f"skip {save_paths[0]} (and {len(save_paths)-1} others)")
        else:
            # 批量调用模型处理所有source图像
            images, _ = model(
                image=source_images,
                label=target_class,
                strength=args.aug_strength,
                metadata=target_metadata,
                resolution=args.resolution,
            )
            
            # 保存所有生成的图像
            for image, save_path in zip(images, save_paths):
                image.save(save_path)
            
            print(f"batch processed {len(images)} images for target {target_name}")


def main(args):

    torch.multiprocessing.set_start_method("spawn")

    os.makedirs(os.path.join(args.output_root, args.dataset), exist_ok=True)

    check_args_valid(args)
    if args.task == "vanilla":
        output_name = f"shot{args.examples_per_class}_{args.sample_strategy}_{args.strength_strategy}_{args.aug_strength}"
    else:  # imbalanced
        output_name = f"imb{args.imbalance_factor}_{args.sample_strategy}_{args.strength_strategy}_{args.aug_strength}"
    args.output_path = os.path.join(args.output_root, args.dataset, output_name)

    os.makedirs(args.output_path, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    gpu_ids = args.gpu_ids
    in_queue = Queue()
    out_queue = Queue()

    if args.task == "imbalanced":
        train_dataset = IMBALANCE_DATASET_NAME_MAPPING[args.dataset](
            split="train",
            seed=args.seed,
            resolution=args.resolution,
            imbalance_factor=args.imbalance_factor,
        )
    else:
        train_dataset = DATASET_NAME_MAPPING[args.dataset](
            split="train",
            seed=args.seed,
            examples_per_class=args.examples_per_class,
            resolution=args.resolution,
        )

    num_classes = len(train_dataset.class_names)

    for name in train_dataset.class_names:
        name = name.replace(" ", "_").replace("/", "_")
        os.makedirs(os.path.join(args.output_path, "data", name), exist_ok=True)

    # 加载映射文件
    mappings = load_mapping_file(args.mapping_file)
    
    # 按target_name分组，每个任务包处理5个任务
    target_name_groups = defaultdict(list)
    for mapping in mappings:
        # 获取target元数据以获取target_name
        target_metadata = train_dataset.get_metadata_by_idx(mapping['target_idx'])
        target_name = target_metadata["name"].replace(" ", "_").replace("/", "_")
        target_name_groups[target_name].append(mapping)
    
    # 创建任务包，每个任务包最多包含5个任务
    task_packages = []
    for target_name, group_mappings in target_name_groups.items():
        # 将每个target_name的映射分成大小为5的批次
        for i in range(0, len(group_mappings), 5):
            batch_mappings = group_mappings[i:i+5]
            # 获取第一个映射的target信息作为任务包标识
            first_mapping = batch_mappings[0]
            task_packages.append({
                'target_class': first_mapping['target_class'],
                'target_idx': first_mapping['target_idx'],
                'target_name': target_name,
                'mappings': batch_mappings
            })
    
    # 将任务包放入队列
    for task_package in task_packages:
        in_queue.put(task_package)
    
    num_tasks = len(task_packages)
    print(f"创建了 {num_tasks} 个任务包（按target_name分组，每个任务包最多5个任务）")

    sample_config = vars(args)
    sample_config["num_classes"] = num_classes
    sample_config["total_tasks"] = num_tasks
    sample_config["sample_strategy"] = args.sample_strategy
    sample_config["mapping_file"] = args.mapping_file

    with open(
        os.path.join(args.output_path, "config.yaml"), "w", encoding="utf-8"
    ) as f:
        yaml.dump(sample_config, f)

    processes = []
    total_tasks = in_queue.qsize()
    print("Number of total tasks", total_tasks)

    with tqdm(total=total_tasks, desc="Processing") as pbar:
        for process_id, gpu_id in enumerate(gpu_ids):
            process = Process(
                target=sample_func,
                args=(args, in_queue, out_queue, gpu_id, process_id),
            )
            process.start()
            processes.append(process)

        while any(process.is_alive() for process in processes):
            current_queue_size = in_queue.qsize()
            pbar.n = total_tasks - current_queue_size
            pbar.refresh()
            time.sleep(1)

        for process in processes:
            process.join()

    # Generate meta.csv for indexing images
    rootdir = os.path.join(args.output_path, "data")
    pattern_level_1 = r"(.+)"
    pattern_level_2 = r"(.+)-(\d+)-(.+).png"
    data_dict = defaultdict(list)
    for dir in os.listdir(rootdir):
        if not os.path.isdir(os.path.join(rootdir, dir)):
            continue
        match_1 = re.match(pattern_level_1, dir)
        first_dir = match_1.group(1).replace("_", " ")
        for file in os.listdir(os.path.join(rootdir, dir)):
            match_2 = re.match(pattern_level_2, file)
            second_dir = match_2.group(1).replace("_", " ")
            num = int(match_2.group(2))
            floating_num = float(match_2.group(3))
            data_dict["First Directory"].append(first_dir)
            data_dict["Second Directory"].append(second_dir)
            data_dict["Number"].append(num)
            data_dict["Strength"].append(floating_num)
            data_dict["Path"].append(os.path.join(dir, file))

    df = pd.DataFrame(data_dict)

    # Validate generated images
    valid_rows = []
    for index, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(args.output_path, "data", row["Path"])
        try:
            img = Image.open(image_path)
            img.close()
            valid_rows.append(row)
        except Exception as e:
            os.remove(image_path)
            print(f"Deleted {image_path} due to error: {str(e)}")

    valid_df = pd.DataFrame(valid_rows)
    csv_path = os.path.join(args.output_path, "meta.csv")
    valid_df.to_csv(csv_path, index=False)

    print("DataFrame:")
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference script")
    parser.add_argument(
        "--finetuned_ckpt",
        type=str,
        required=True,
        help="key for indexing finetuned model",
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        required=True,
        help="path to the source-target mapping CSV file",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/aug_samples",
        help="output root directory",
    )
    parser.add_argument(
        "--model_path", type=str, default="CompVis/stable-diffusion-v1-4"
    )
    parser.add_argument("--dataset", type=str, default="pascal", help="dataset name")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--examples_per_class",
        type=int,
        default=-1,
        help="synthetic examples per class",
    )
    parser.add_argument("--resolution", type=int, default=512, help="image resolution")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--prompt", type=str, default="a photo of a {name}", help="prompt for synthesis"
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="ti-mix",
        choices=[
            "real-gen",
            "real-aug",  # real guidance
            "real-mix",
            "ti-aug",
            "ti-mix",
            "diff-aug",
            "diff-mix",
            "diff-gen",
        ],
        help="sampling strategy for synthetic data",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="classifier free guidance scale",
    )
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[0], help="gpu ids")
    parser.add_argument(
        "--task",
        type=str,
        default="vanilla",
        choices=["vanilla", "imbalanced"],
        help="task",
    )
    parser.add_argument(
        "--imbalance_factor",
        type=float,
        default=0.01,
        choices=[0.01, 0.02, 0.1],
        help="imbalanced factor, only for imbalanced task",
    )
    parser.add_argument(
        "--strength_strategy",
        type=str,
        default="fixed",
        choices=["fixed", "uniform"],
    )
    parser.add_argument(
        "--aug_strength", type=float, default=0.5, help="augmentation strength"
    )
    parser.add_argument(
        "--ddim_eta", type=float, default=0.0, help="DDIM eta parameter (0.0 for DDIM, 1.0 for DDPM)"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=25, help="number of DDIM inference steps"
    )
    args = parser.parse_args()

    main(args)