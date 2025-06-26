"""
数据集加载和划分模块

负责扫描VoxCeleb2数据集目录，根据配置进行合理的数据集划分。
实现训练集、注册集和测试集的文件路径管理功能。
"""

import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm

from .config import config


def scan_voxceleb_dataset(base_path: str) -> Dict[str, List[str]]:
    """
    扫描VoxCeleb2数据集目录，获取所有说话人的音频文件路径
    
    Args:
        base_path: VoxCeleb2数据集根目录路径
        
    Returns:
        字典，格式为 {speaker_id: [file_path1, file_path2, ...]}
    """
    speaker_files = defaultdict(list)
    
    # VoxCeleb2的目录结构: base_path/dev/aac/speaker_id/video_id/segment.m4a
    dev_path = os.path.join(base_path, "dev", "aac")
    
    if not os.path.exists(dev_path):
        raise FileNotFoundError(f"VoxCeleb2数据集路径不存在: {dev_path}")
    
    print(f"正在扫描数据集: {dev_path}")
    
    # 遍历所有说话人ID目录
    speaker_dirs = [d for d in os.listdir(dev_path) 
                   if os.path.isdir(os.path.join(dev_path, d)) and d.startswith('id')]
    
    for speaker_id in tqdm(speaker_dirs, desc="扫描说话人目录"):
        speaker_path = os.path.join(dev_path, speaker_id)
        
        # 遍历该说话人的所有视频目录
        for video_id in os.listdir(speaker_path):
            video_path = os.path.join(speaker_path, video_id)
            
            if os.path.isdir(video_path):
                # 遍历该视频的所有音频片段
                for audio_file in os.listdir(video_path):
                    if audio_file.endswith('.m4a'):
                        file_path = os.path.join(video_path, audio_file)
                        speaker_files[speaker_id].append(file_path)
    
    # 为每个说话人的文件列表排序，确保结果可重现
    for speaker_id in speaker_files:
        speaker_files[speaker_id].sort()
    
    print(f"扫描完成，共找到 {len(speaker_files)} 个说话人，"
          f"总计 {sum(len(files) for files in speaker_files.values())} 个音频文件")
    
    return dict(speaker_files)


def get_dataset_split(base_path: str, train_ratio: float = None, 
                     enroll_files_per_speaker: int = None,
                     random_seed: int = None) -> Tuple[Dict, Dict, Dict]:
    """
    将数据集划分为全局训练集、注册集和测试集
    
    Args:
        base_path: 数据集根目录路径
        train_ratio: 全局训练集比例
        enroll_files_per_speaker: 每个说话人用于注册的文件数量
        random_seed: 随机种子
        
    Returns:
        三元组 (global_train_files, enroll_files, test_files_closed_set)
        - global_train_files: {speaker_id: [file_paths]} 用于训练UBM或VQ码本
        - enroll_files: {speaker_id: [file_paths]} 用于说话人注册
        - test_files_closed_set: {speaker_id: [file_paths]} 用于闭集测试
    """
    # 使用配置文件中的默认值
    if train_ratio is None:
        train_ratio = config.TRAIN_RATIO
    if enroll_files_per_speaker is None:
        enroll_files_per_speaker = config.ENROLL_FILES_PER_SPEAKER
    if random_seed is None:
        random_seed = config.RANDOM_SEED
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 扫描数据集
    all_speaker_files = scan_voxceleb_dataset(base_path)
    
    # 过滤掉文件数量不足的说话人
    min_files_required = enroll_files_per_speaker + 1  # 至少需要注册文件+1个测试文件
    valid_speakers = {speaker_id: files for speaker_id, files in all_speaker_files.items()
                     if len(files) >= min_files_required}
    
    print(f"过滤后有效说话人数量: {len(valid_speakers)} "
          f"(每个说话人至少需要 {min_files_required} 个文件)")
    
    # 随机划分说话人
    speaker_ids = list(valid_speakers.keys())
    random.shuffle(speaker_ids)
    
    split_point = int(len(speaker_ids) * train_ratio)
    global_train_speakers = speaker_ids[:split_point]
    enroll_test_speakers = speaker_ids[split_point:]
    
    print(f"全局训练集说话人数量: {len(global_train_speakers)}")
    print(f"注册/测试集说话人数量: {len(enroll_test_speakers)}")
    
    # 构建全局训练集
    global_train_files = {}
    for speaker_id in global_train_speakers:
        global_train_files[speaker_id] = valid_speakers[speaker_id]
    
    # 构建注册集和测试集
    enroll_files = {}
    test_files_closed_set = {}
    
    for speaker_id in enroll_test_speakers:
        files = valid_speakers[speaker_id].copy()
        random.shuffle(files)  # 随机打乱文件顺序
        
        # 前N个文件用于注册
        enroll_files[speaker_id] = files[:enroll_files_per_speaker]
        
        # 剩余文件用于测试
        test_files_closed_set[speaker_id] = files[enroll_files_per_speaker:]
    
    # 统计信息
    total_global_train = sum(len(files) for files in global_train_files.values())
    total_enroll = sum(len(files) for files in enroll_files.values())
    total_test = sum(len(files) for files in test_files_closed_set.values())
    
    print(f"数据集划分完成:")
    print(f"  全局训练集: {len(global_train_files)} 个说话人, {total_global_train} 个文件")
    print(f"  注册集: {len(enroll_files)} 个说话人, {total_enroll} 个文件")
    print(f"  闭集测试集: {len(test_files_closed_set)} 个说话人, {total_test} 个文件")
    
    return global_train_files, enroll_files, test_files_closed_set


def load_test_trials(test_files: Dict[str, List[str]]) -> List[Tuple[str, str, bool]]:
    """
    从测试文件生成测试试验列表
    
    Args:
        test_files: 测试文件字典 {speaker_id: [file_paths]}
        
    Returns:
        试验列表，每个元素为 (file_path, speaker_id, is_target)
        is_target=True表示目标试验，False表示非目标试验
    """
    trials = []
    speaker_ids = list(test_files.keys())
    
    # 生成目标试验（正例）
    for speaker_id, files in test_files.items():
        for file_path in files:
            trials.append((file_path, speaker_id, True))
    
    # 生成非目标试验（负例）
    for speaker_id, files in test_files.items():
        for file_path in files:
            # 随机选择其他说话人作为冒名者
            other_speakers = [sid for sid in speaker_ids if sid != speaker_id]
            if other_speakers:
                imposter_id = random.choice(other_speakers)
                trials.append((file_path, imposter_id, False))
    
    # 打乱试验顺序
    random.shuffle(trials)
    
    print(f"生成测试试验: {len(trials)} 个试验 "
          f"(目标试验: {sum(1 for _, _, is_target in trials if is_target)}, "
          f"非目标试验: {sum(1 for _, _, is_target in trials if not is_target)})")
    
    return trials


if __name__ == "__main__":
    # 测试数据加载功能
    try:
        global_train, enroll, test = get_dataset_split(config.VOXCELEB_PATH)
        trials = load_test_trials(test)
        print("数据加载测试成功!")
    except Exception as e:
        print(f"数据加载测试失败: {e}")
