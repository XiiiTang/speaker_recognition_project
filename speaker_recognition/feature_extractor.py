"""
音频特征提取模块

负责将音频文件转换为MFCC特征矩阵。
支持m4a格式转换和多种音频预处理功能。
"""

import os
import tempfile
import numpy as np
import librosa
from pydub import AudioSegment
from typing import Optional, Tuple
import warnings

from .config import config

# 忽略librosa的一些警告
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')


def convert_m4a_to_wav(m4a_path: str, temp_dir: str = None) -> str:
    """
    将m4a格式音频文件转换为wav格式
    
    Args:
        m4a_path: m4a文件路径
        temp_dir: 临时文件目录，如果为None则使用系统临时目录
        
    Returns:
        转换后的wav文件路径
    """
    if temp_dir is None:
        temp_dir = config.TEMP_DIR
    
    # 确保临时目录存在
    os.makedirs(temp_dir, exist_ok=True)
    
    # 生成临时wav文件路径
    temp_wav_path = os.path.join(temp_dir, 
                                f"temp_{os.path.basename(m4a_path)}.wav")
    
    try:
        # 使用pydub加载m4a文件
        audio = AudioSegment.from_file(m4a_path, format="m4a")
        
        # 转换为单声道
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # 设置采样率
        audio = audio.set_frame_rate(config.SAMPLE_RATE)
        
        # 导出为wav格式
        audio.export(temp_wav_path, format="wav")
        
        return temp_wav_path
        
    except Exception as e:
        raise RuntimeError(f"转换音频文件失败 {m4a_path}: {e}")


def load_audio(audio_path: str, sr: int = None) -> Tuple[np.ndarray, int]:
    """
    加载音频文件
    
    Args:
        audio_path: 音频文件路径
        sr: 目标采样率，如果为None则使用配置中的采样率
        
    Returns:
        (音频信号, 采样率)
    """
    if sr is None:
        sr = config.SAMPLE_RATE
    
    temp_wav_path = None
    
    try:
        # 如果是m4a文件，先转换为wav
        if audio_path.lower().endswith('.m4a'):
            temp_wav_path = convert_m4a_to_wav(audio_path)
            load_path = temp_wav_path
        else:
            load_path = audio_path
        
        # 使用librosa加载音频
        y, sr_actual = librosa.load(load_path, sr=sr, mono=True)
        
        return y, sr_actual
        
    except Exception as e:
        raise RuntimeError(f"加载音频文件失败 {audio_path}: {e}")
        
    finally:
        # 清理临时文件
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except:
                pass  # 忽略删除临时文件的错误


def preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """
    音频预处理
    
    Args:
        y: 音频信号
        sr: 采样率
        
    Returns:
        预处理后的音频信号
    """
    # 预加重
    if config.PREEMPHASIS > 0:
        y = np.append(y[0], y[1:] - config.PREEMPHASIS * y[:-1])
    
    # 去除静音段（可选）
    # y, _ = librosa.effects.trim(y, top_db=20)
    
    return y


def extract_mfcc(audio_path: str, n_mfcc: int = None, 
                n_fft: int = None, hop_length: int = None,
                win_length: int = None) -> np.ndarray:
    """
    从音频文件提取MFCC特征
    
    Args:
        audio_path: 音频文件路径
        n_mfcc: MFCC特征维度
        n_fft: FFT窗口大小
        hop_length: 帧移长度
        win_length: 窗口长度
        
    Returns:
        MFCC特征矩阵，形状为 (n_mfcc, n_frames)
    """
    # 使用配置中的默认值
    if n_mfcc is None:
        n_mfcc = config.N_MFCC
    if n_fft is None:
        n_fft = config.N_FFT
    if hop_length is None:
        hop_length = config.HOP_LENGTH
    if win_length is None:
        win_length = config.WIN_LENGTH
    
    try:
        # 加载音频
        y, sr = load_audio(audio_path)
        
        # 预处理
        y = preprocess_audio(y, sr)
        
        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=True,
            pad_mode='reflect'
        )
        
        # 计算一阶和二阶差分（Delta和Delta-Delta）
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # 拼接MFCC及其差分特征
        features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        
        return features
        
    except Exception as e:
        raise RuntimeError(f"提取MFCC特征失败 {audio_path}: {e}")


def extract_mfcc_from_files(file_paths: list, verbose: bool = True) -> list:
    """
    从多个音频文件批量提取MFCC特征
    
    Args:
        file_paths: 音频文件路径列表
        verbose: 是否显示进度
        
    Returns:
        MFCC特征列表，每个元素为一个特征矩阵
    """
    features_list = []
    
    if verbose:
        from tqdm import tqdm
        file_paths = tqdm(file_paths, desc="提取MFCC特征")
    
    for file_path in file_paths:
        try:
            mfcc = extract_mfcc(file_path)
            features_list.append(mfcc)
        except Exception as e:
            if verbose:
                print(f"警告: 提取特征失败 {file_path}: {e}")
            continue
    
    return features_list


def normalize_features(features: np.ndarray, method: str = 'cmvn') -> np.ndarray:
    """
    特征归一化
    
    Args:
        features: 特征矩阵，形状为 (n_features, n_frames)
        method: 归一化方法，'cmvn'表示倒谱均值方差归一化
        
    Returns:
        归一化后的特征矩阵
    """
    if method == 'cmvn':
        # 倒谱均值方差归一化
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        # 避免除零
        std = np.where(std == 0, 1, std)
        features = (features - mean) / std
    elif method == 'minmax':
        # 最小-最大归一化
        min_val = np.min(features, axis=1, keepdims=True)
        max_val = np.max(features, axis=1, keepdims=True)
        # 避免除零
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        features = (features - min_val) / range_val
    
    return features


def extract_features_for_speaker(file_paths: list, normalize: bool = True) -> np.ndarray:
    """
    为单个说话人提取并合并所有文件的特征
    
    Args:
        file_paths: 该说话人的音频文件路径列表
        normalize: 是否进行特征归一化
        
    Returns:
        合并后的特征矩阵，形状为 (n_features, total_frames)
    """
    all_features = []
    
    for file_path in file_paths:
        try:
            mfcc = extract_mfcc(file_path)
            if normalize:
                mfcc = normalize_features(mfcc)
            all_features.append(mfcc)
        except Exception as e:
            print(f"警告: 提取特征失败 {file_path}: {e}")
            continue
    
    if not all_features:
        raise ValueError("没有成功提取到任何特征")
    
    # 沿时间轴拼接所有特征
    combined_features = np.hstack(all_features)
    
    return combined_features


if __name__ == "__main__":
    # 测试特征提取功能
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        try:
            mfcc = extract_mfcc(test_file)
            print(f"成功提取MFCC特征，形状: {mfcc.shape}")
        except Exception as e:
            print(f"特征提取测试失败: {e}")
    else:
        print("用法: python feature_extractor.py <音频文件路径>")
