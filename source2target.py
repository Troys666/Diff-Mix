import argparse
import os
import random
import re
import sys
import time
import csv
from typing import List, Tuple, Dict, Optional

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


class SourceTargetMapper:
    """源目标映射生成器"""
    
    def __init__(self, dataset, seed: int = 42):
        self.dataset = dataset
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # 扫描数据集获取items
        self.items = self._scan_dataset()
        print(f"扫描到 {len(self.items)} 个样本，{len(set(item[1] for item in self.items))} 个类别")
    
    def _scan_dataset(self) -> List[Tuple[int, int]]:
        """扫描数据集，返回 (idx, class) 列表"""
        items = []
        for idx in range(len(self.dataset)):
            label = self.dataset.get_label_by_idx(idx)
            items.append((idx, label))
        return items
    
    def _expand_items(self, multiplier: int) -> List[Tuple[int, int]]:
        """将每个item重复multiplier次"""
        expanded = []
        for item in self.items:
            expanded.extend([item] * multiplier)
        return expanded
    
    def _shuffle_with_seed(self, items: List, seed: int) -> List:
        """使用指定seed打乱列表"""
        random.seed(seed)
        shuffled = items.copy()
        random.shuffle(shuffled)
        return shuffled
    
    def _check_aug_constraints(self, src_list: List[Tuple[int, int]], 
                              tgt_list: List[Tuple[int, int]]) -> Tuple[bool, int]:
        """检查aug模式的约束：同类但尽量不同图片"""
        self_pairs = 0
        for i, (src_idx, src_class) in enumerate(src_list):
            tgt_idx, tgt_class = tgt_list[i]
            if src_class != tgt_class:
                return False, 0  # 不同类，违反约束
            if src_idx == tgt_idx:
                self_pairs += 1  # 自配对
        return True, self_pairs
    
    def _check_mix_constraints(self, src_list: List[Tuple[int, int]], 
                              tgt_list: List[Tuple[int, int]]) -> Tuple[bool, int]:
        """检查mix模式的约束：不同类"""
        same_class_pairs = 0
        for i, (src_idx, src_class) in enumerate(src_list):
            tgt_idx, tgt_class = tgt_list[i]
            if src_class == tgt_class:
                same_class_pairs += 1
        return same_class_pairs == 0, same_class_pairs
    
    def _fix_aug_constraints(self, src_list: List[Tuple[int, int]], 
                           tgt_list: List[Tuple[int, int]], 
                           max_attempts: int = 100) -> Tuple[List[Tuple[int, int]], int]:
        """修复aug模式约束：同类配对但避免自配对"""
        # 首先确保target列表中的每个item都与对应的source同类
        fixed_tgt_list = []
        total_self_pairs = 0
        
        # 按类分组处理
        src_by_class = defaultdict(list)
        for i, (src_idx, src_class) in enumerate(src_list):
            src_by_class[src_class].append((i, src_idx, src_class))
        
        # 为每个类别内的source找到同类的target
        for class_id in sorted(src_by_class.keys()):
            class_sources = src_by_class[class_id]
            
            # 获取该类别的所有样本作为候选target
            class_candidates = [(idx, label) for idx, label in self.items if label == class_id]
            
            # 为每个source分配一个同类的target
            for source_pos, source_idx, source_class in class_sources:
                # 从同类候选中随机选择一个target（避免自配对）
                available_targets = [(idx, label) for idx, label in class_candidates if idx != source_idx]
                
                if available_targets:
                    # 随机选择一个target
                    random.seed(self.seed + source_pos)
                    target_idx, target_class = random.choice(available_targets)
                else:
                    # 如果没有其他同类样本，只能自配对
                    target_idx, target_class = source_idx, source_class
                    total_self_pairs += 1
                
                # 将target插入到正确的位置
                while len(fixed_tgt_list) <= source_pos:
                    fixed_tgt_list.append(None)
                fixed_tgt_list[source_pos] = (target_idx, target_class)
        
        if total_self_pairs > 0:
            print(f"警告：仍有 {total_self_pairs} 个自配对（某些类别样本数过少）")
        
        return fixed_tgt_list, total_self_pairs
    
    def _fix_mix_constraints(self, src_list: List[Tuple[int, int]], 
                           tgt_list: List[Tuple[int, int]], 
                           max_attempts: int = 200) -> Tuple[List[Tuple[int, int]], int]:
        """修复mix模式约束：避免同类配对"""
        best_tgt_list = tgt_list.copy()
        min_same_class_pairs = len(tgt_list)
        
        for attempt in range(max_attempts):
            # 重新打乱target列表
            test_tgt_list = self._shuffle_with_seed(tgt_list, self.seed + attempt)
            
            # 尝试交换修复冲突
            fixed_tgt_list = self._swap_fix_mix_constraints(src_list, test_tgt_list)
            
            # 检查结果
            is_valid, same_class_pairs = self._check_mix_constraints(src_list, fixed_tgt_list)
            
            if same_class_pairs < min_same_class_pairs:
                min_same_class_pairs = same_class_pairs
                best_tgt_list = fixed_tgt_list
            
            if is_valid:
                break  # 找到完美解
        
        if min_same_class_pairs > 0:
            print(f"警告：仍有 {min_same_class_pairs} 个同类配对（类别数过少）")
        
        return best_tgt_list, min_same_class_pairs
    
    def _swap_fix_mix_constraints(self, src_list: List[Tuple[int, int]], 
                                tgt_list: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """通过交换修复mix模式约束"""
        fixed_tgt_list = tgt_list.copy()
        
        for i in range(len(src_list)):
            src_class = src_list[i][1]
            tgt_class = fixed_tgt_list[i][1]
            
            if src_class == tgt_class:  # 发现冲突
                # 尝试在后面的位置找到可交换的项
                swapped = False
                for j in range(i + 1, len(fixed_tgt_list)):
                    if (fixed_tgt_list[j][1] != src_class and 
                        fixed_tgt_list[j][1] != src_list[j][1]):
                        # 交换i和j位置的target
                        fixed_tgt_list[i], fixed_tgt_list[j] = fixed_tgt_list[j], fixed_tgt_list[i]
                        swapped = True
                        break
                
                if not swapped:
                    # 尝试在前面的位置找到可交换的项
                    for j in range(i):
                        if (fixed_tgt_list[j][1] != src_class and 
                            fixed_tgt_list[j][1] != src_list[j][1]):
                            # 交换i和j位置的target
                            fixed_tgt_list[i], fixed_tgt_list[j] = fixed_tgt_list[j], fixed_tgt_list[i]
                            swapped = True
                            break
        
        return fixed_tgt_list
    
    def generate_mapping(self, mode: str, multiplier: int, output_file: str) -> Dict:
        """生成源目标映射"""
        print(f"生成 {mode} 模式的映射，扩增倍数: {multiplier}")
        
        # 1. 准备与重复
        expanded = self._expand_items(multiplier)
        print(f"扩展后总样本数: {len(expanded)}")
        
        # 2. 初始化source/target列表
        src_list = self._shuffle_with_seed(expanded, self.seed)
        tgt_list = self._shuffle_with_seed(expanded, self.seed + 1000)  # 使用不同seed
        
        # 3. 根据模式修复约束
        if mode == "aug":
            tgt_list, self_pairs = self._fix_aug_constraints(src_list, tgt_list)
            print(f"aug模式：自配对数量: {self_pairs}")
        elif mode == "mix":
            tgt_list, same_class_pairs = self._fix_mix_constraints(src_list, tgt_list)
            print(f"mix模式：同类配对数量: {same_class_pairs}")
        else:
            raise ValueError(f"不支持的模式: {mode}")
        
        # 4. 生成配对并写入文件
        pairs = []
        for i, (src_idx, src_class) in enumerate(src_list):
            tgt_idx, tgt_class = tgt_list[i]
            
            # 获取元数据
            src_metadata = self.dataset.get_metadata_by_idx(src_idx)
            tgt_metadata = self.dataset.get_metadata_by_idx(tgt_idx)
            
            pair = {
                'pair_id': i,
                'source_idx': src_idx,
                'source_class': src_class,
                'source_name': src_metadata.get('name', ''),
                'target_idx': tgt_idx,
                'target_class': tgt_class,
                'target_name': tgt_metadata.get('name', ''),
            }
            pairs.append(pair)
        
        # 写入CSV文件
        self._write_pairs_to_csv(pairs, output_file)
        
        # 返回统计信息
        stats = {
            'mode': mode,
            'multiplier': multiplier,
            'total_pairs': len(pairs),
            'unique_sources': len(set(pair['source_idx'] for pair in pairs)),
            'unique_targets': len(set(pair['target_idx'] for pair in pairs)),
            'classes': len(set(pair['source_class'] for pair in pairs))
        }
        
        if mode == "aug":
            stats['self_pairs'] = self_pairs
        elif mode == "mix":
            stats['same_class_pairs'] = same_class_pairs
        
        return stats
    
    def _write_pairs_to_csv(self, pairs: List[Dict], output_file: str):
        """将配对写入CSV文件"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['pair_id', 'source_idx', 'source_class', 'source_name', 
                         'target_idx', 'target_class', 'target_name']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pairs)
        
        print(f"映射文件已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser("源目标映射生成器")
    parser.add_argument("--dataset", type=str, default="cub", help="数据集名称")
    parser.add_argument("--task", type=str, default="vanilla", choices=["vanilla", "imbalanced"])
    parser.add_argument("--examples_per_class", type=int, default=-1, help="每类样本数")
    parser.add_argument("--resolution", type=int, default=512, help="图像分辨率")
    parser.add_argument("--imbalance_factor", type=float, default=0.01, help="不平衡因子")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--mode", type=str, default="aug", choices=["aug", "mix"], help="映射模式")
    parser.add_argument("--multiplier", type=int, default=2, help="扩增倍数")
    parser.add_argument("--output_dir", type=str, default="outputs/mappings", help="输出目录")
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 加载数据集
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
    
    print(f"数据集: {args.dataset}")
    print(f"任务类型: {args.task}")
    print(f"类别数: {len(train_dataset.class_names)}")
    print(f"总样本数: {len(train_dataset)}")
    
    # 创建映射生成器
    mapper = SourceTargetMapper(train_dataset, seed=args.seed)
    
    # 生成输出文件名
    shot_str = f"shot{args.examples_per_class}" if args.examples_per_class > 0 else "full"
    output_file = os.path.join(
        args.output_dir, 
        f"{args.dataset}_{shot_str}_{args.mode}_mult{args.multiplier}_seed{args.seed}.csv"
    )
    
    # 生成映射
    stats = mapper.generate_mapping(args.mode, args.multiplier, output_file)
    
    # 打印统计信息
    print("\n=== 映射统计信息 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
