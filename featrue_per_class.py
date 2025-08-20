import argparse
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Lambda, Compose
import torch

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


def dataset_feature_per_class(dataset, num_classes):
    
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
    )

    # 采用 model.pipe 的 UNet/VAE/Scheduler 提取中间特征
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import CLIPModel, AutoProcessor

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    vae = model.pipe.vae
    unet = model.pipe.unet
    scheduler = getattr(model.pipe, "scheduler", None)
    unet.eval(); vae.eval()

    weight_dtype = torch.float32
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)

    # CLIP作为嵌入器
    clip_model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = AutoProcessor.from_pretrained(clip_model_name)
    clip_model.eval()

    # 预处理到VAE输入（-1~1）
    vae_preprocess = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    # 为数据集设置正确的transform（VAE需要的[-1,1]范围）
    vae_transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    train_dataset.set_transform(vae_transform)

    # 注册UNet中间层hook - 提取上采样最后两层（最终层）
    features = {"up_block_penultimate": None, "up_block_final": None}

    def hook_up_block_penultimate(module, inputs, output):
        features["up_block_penultimate"] = output.detach()

    def hook_up_block_final(module, inputs, output):
        features["up_block_final"] = output.detach()

    # 注册上采样块的最后两层（最终层）
    # 通常UNet有4个上采样块，最后两层是索引2和3
    num_up_blocks = len(unet.up_blocks)
    penultimate_idx = num_up_blocks - 2  # 倒数第二层
    final_idx = num_up_blocks - 1        # 最后一层
    
    print(f"UNet上采样块数量: {num_up_blocks}")
    print(f"提取倒数第二层 (索引 {penultimate_idx}) 和最后一层 (索引 {final_idx})")
    
    up_block_penultimate_handle = unet.up_blocks[penultimate_idx].register_forward_hook(hook_up_block_penultimate)
    up_block_final_handle = unet.up_blocks[final_idx].register_forward_hook(hook_up_block_final)

    # 聚合容器 - 两个特征（像素的CLIP、上采样最后两层）
    class_names = [n.replace("/", " ") for n in train_dataset.class_names]
    num_classes = len(class_names)
    per_class_count = {i: 0 for i in range(num_classes)}
    
    # 两个特征：像素的CLIP、上采样最后两层
    sums_clip = {i: None for i in range(num_classes)}
    sums_up_block_penultimate = {i: None for i in range(num_classes)}
    sums_up_block_final = {i: None for i in range(num_classes)}

    # 时间步设为0，因为我们要提取特征，不需要加噪声
    timestep_int = 0

    def extract_clip_features(image: torch.Tensor) -> torch.Tensor:
        """提取像素的CLIP特征"""
        # 将图像从[-1,1]转换到[0,1]范围
        image = (image + 1) / 2
        image = torch.clamp(image, 0, 1)
        
        # 转换为PIL图像
        image = (image * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
        pil_images = [Image.fromarray(img) for img in image]
        
        # 使用CLIP提取特征
        inputs = clip_processor(images=pil_images, return_tensors="pt").to(device)
        with torch.no_grad():
            clip_features = clip_model.get_image_features(**inputs)
        
        return clip_features.float().detach().cpu()

    def global_avg_pool(feat_map: torch.Tensor) -> torch.Tensor:
        """全局平均池化特征图"""
        return feat_map.mean(dim=[0, 2, 3]).float().detach().cpu()

    # 定义collate_fn函数，类似train_lora.py中的方式
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # 创建DataLoader
    batch_size = getattr(args, 'batch_size', 4)  # 默认batch_size为4
    if args.task == "imbalanced":
        train_sampler = train_dataset.get_weighted_sampler()
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=getattr(args, 'dataloader_num_workers', 0),
            sampler=train_sampler,
        )
    else:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=getattr(args, 'dataloader_num_workers', 0),
        )

    # 批量处理数据集
    for batch in tqdm(train_dataloader, desc="Processing batches"):
        pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
        labels = batch["labels"]
        
        # 1. 提取像素的CLIP特征
        clip_features = extract_clip_features(pixel_values)
        
        # 2. 通过UNet提取上采样最后两层特征
        # 批量编码到潜在空间
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        bsz = latents.shape[0]
        
        # 时间步设为0，不加噪声
        timesteps = torch.tensor([timestep_int] * bsz, device=device, dtype=torch.long)
        noisy_latents = latents  # 直接使用原始latents，不加噪声

        # 获取文本嵌入
        encoder_hidden_states_list = []
        for i in range(bsz):
            label = int(labels[i].item())
            # 获取该标签对应的元数据
            label_indices = train_dataset.label_to_indices[label]
            sample_idx = label_indices[0]  # 使用第一个样本的元数据
            metadata = train_dataset.get_metadata_by_idx(sample_idx)
            
            # 构建prompt，类似sample_mp.py中的方式
            name = metadata.get("name", "")
            if hasattr(model, 'name2placeholder') and model.name2placeholder is not None:
                name = model.name2placeholder.get(name, name)
            if metadata.get("super_class", None) is not None:
                name = name + " " + metadata.get("super_class", "")
            prompt = args.prompt.format(name=name)
            
            # 使用text_encoder获取文本嵌入
            text_inputs = model.pipe.tokenizer(
                prompt,
                padding="max_length",
                max_length=model.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            
            with torch.no_grad():
                encoder_hidden_states = model.pipe.text_encoder(text_input_ids)[0].to(dtype=weight_dtype)
            encoder_hidden_states_list.append(encoder_hidden_states)
        
        # 将文本嵌入堆叠成batch
        encoder_hidden_states = torch.cat(encoder_hidden_states_list, dim=0)

        # 通过UNet获取上采样最后两层特征
        _ = unet(noisy_latents, timesteps, encoder_hidden_states)

        # 处理每个样本的特征
        for i in range(bsz):
            label = int(labels[i].item())
            
            # 提取单个样本的特征
            clip_feat = clip_features[i:i+1]    # [1, D]
            up_block_2_feat = global_avg_pool(features["up_block_penultimate"][i:i+1])  # [C_up2]
            up_block_3_feat = global_avg_pool(features["up_block_final"][i:i+1])  # [C_up3]

            per_class_count[label] += 1
            
            # 累积特征
            sums_clip[label] = (
                clip_feat.clone() if sums_clip[label] is None else sums_clip[label] + clip_feat
            )
            sums_up_block_penultimate[label] = (
                up_block_2_feat.clone() if sums_up_block_penultimate[label] is None else sums_up_block_penultimate[label] + up_block_2_feat
            )
            sums_up_block_final[label] = (
                up_block_3_feat.clone() if sums_up_block_final[label] is None else sums_up_block_final[label] + up_block_3_feat
            )

        # 清理特征缓存
        features["up_block_penultimate"] = None
        features["up_block_final"] = None

    # 保存均值：将本次运行包裹在一个大的文件夹中（不保存像素特征）
    def _run_folder_name(a):
        shot_str = (
            f"shot{a.examples_per_class}" if a.examples_per_class is not None and a.examples_per_class > 0 else "full"
        )
        ckpt_name = os.path.basename(a.finetuned_ckpt) if getattr(a, "finetuned_ckpt", None) else "base"
        return f"{a.dataset}_{shot_str}_{a.sample_strategy}_res{a.resolution}_seed{a.seed}_{ckpt_name}"

    run_folder = _run_folder_name(args)
    base_out_dir = os.path.join("outputs", "features_per_class", run_folder)
    os.makedirs(base_out_dir, exist_ok=True)
    
    for cls_id in range(num_classes):
        cnt = max(per_class_count[cls_id], 1)
        name = str(class_names[cls_id]).replace("/", " ").replace(" ", "_")
        class_dir = os.path.join(base_out_dir, f"{name}_{cls_id}")
        os.makedirs(class_dir, exist_ok=True)

        # 计算平均特征（不保存像素特征）
        avg_clip = None if sums_clip[cls_id] is None else sums_clip[cls_id] / cnt
        avg_up_block_2 = None if sums_up_block_penultimate[cls_id] is None else sums_up_block_penultimate[cls_id] / cnt
        avg_up_block_3 = None if sums_up_block_final[cls_id] is None else sums_up_block_final[cls_id] / cnt

        save_obj = {
            "class_id": cls_id,
            "class_name": class_names[cls_id],
            "clip_features": avg_clip,        # 像素的CLIP特征
            "up_block_penultimate_features": avg_up_block_2,  # 上采样倒数第二层特征
            "up_block_final_features": avg_up_block_3,        # 上采样最后一层特征
            "count": per_class_count[cls_id],
        }
        torch.save(save_obj, os.path.join(class_dir, f"{name}_avg_features.pt"))

    up_block_penultimate_handle.remove()
    up_block_final_handle.remove()


def main(_args):
    # 沿用现有的数据集与模型定义
    random.seed(args.seed + process_id)
    np.random.seed(args.seed + process_id)
    torch.manual_seed(args.seed + process_id)

    # 在此仅调用实现（实际所需参数在全局 args/gpu_id/process_id 中）
    dataset_feature_per_class(None, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Feature extraction per class via UNet intermediates")
    parser.add_argument("--dataset", type=str, default="pascal")
    parser.add_argument("--task", type=str, default="vanilla", choices=["vanilla", "imbalanced"]) 
    parser.add_argument("--examples_per_class", type=int, default=-1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--sample_strategy", type=str, default="diff-aug",
                        choices=[
                            "real-gen","real-aug","real-mix",
                            "ti-aug","ti-mix",
                            "diff-aug","diff-mix","diff-gen"
                        ])
    parser.add_argument("--model_path", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--finetuned_ckpt", type=str, default=None)
    parser.add_argument("--embed_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="a photo of a {name}")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--imbalance_factor", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of workers for dataloader")

    args = parser.parse_args()

    # 从 finetuned_ckpt 自动解析 lora/embed（若提供）
    if args.finetuned_ckpt is not None:
        try:
            lora_path, embed_path = parse_finetuned_ckpt(args.finetuned_ckpt)
            args.lora_path = lora_path
            args.embed_path = embed_path
        except Exception as e:
            print(f"Warning: parse_finetuned_ckpt failed: {e}")

    # 赋值到全局，供 dataset_feature_per_class 使用
    globals()["args"] = args
    globals()["process_id"] = 0
    globals()["gpu_id"] = args.gpu_id

    main(args)

def _check_args_and_fill_paths(args):
    # 对不同采样策略补全 lora/embed 路径
    if args.sample_strategy in ["diff-gen", "diff-aug", "diff-mix", "ti-aug", "ti-mix"]:
        if args.finetuned_ckpt is not None and (args.lora_path is None or args.embed_path is None):
            lora_path, embed_path = parse_finetuned_ckpt(args.finetuned_ckpt)
            args.lora_path = lora_path
            args.embed_path = embed_path
    else:
        args.lora_path = None if args.lora_path is None else args.lora_path
        args.embed_path = None if args.embed_path is None else args.embed_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Feature extraction per class using UNet intermediates")
    parser.add_argument("--dataset", type=str, default="pascal", help="dataset name")
    parser.add_argument("--task", type=str, default="vanilla", choices=["vanilla", "imbalanced"]) 
    parser.add_argument("--examples_per_class", type=int, default=-1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--imbalance_factor", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample_strategy", type=str, default="diff-mix",
                        choices=["real-gen", "real-aug", "real-mix", "ti-aug", "ti-mix", "diff-aug", "diff-mix", "diff-gen"]) 
    parser.add_argument("--model_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--finetuned_ckpt", type=str, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--embed_path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="a photo of a {name}")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of workers for dataloader")
    args = parser.parse_args()

    # 全局变量供函数使用（与现有风格保持一致）
    process_id = 0
    gpu_id = args.gpu_id
    _check_args_and_fill_paths(args)

    # 构建一次数据集与类数供日志与调用
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

    print(f"Dataset: {args.dataset}")
    print(f"Num classes: {len(train_dataset.class_names)}")
    print(f"Resolution: {args.resolution}")
    print(f"Sample strategy: {args.sample_strategy}")
    print(f"GPU: {gpu_id}")

    # 执行特征提取
    dataset_feature_per_class(train_dataset, len(train_dataset.class_names))
