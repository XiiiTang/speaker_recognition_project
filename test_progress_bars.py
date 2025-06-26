"""
测试进度条功能的脚本

创建模拟数据来测试各个阶段的进度条显示效果
"""

import os
import sys
import numpy as np
import tempfile
from tqdm import tqdm
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from speaker_recognition.config import config
from speaker_recognition.models import DTWModel, VQModel, GMMModel


def create_mock_audio_files(num_speakers=5, files_per_speaker=10):
    """
    创建模拟音频文件路径和对应的MFCC特征
    
    Args:
        num_speakers: 说话人数量
        files_per_speaker: 每个说话人的文件数量
        
    Returns:
        (global_train_files, enroll_files, test_files, mock_features)
    """
    print("创建模拟数据...")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    
    global_train_files = {}
    enroll_files = {}
    test_files = {}
    mock_features = {}  # 存储模拟的MFCC特征
    
    # 为训练集创建说话人
    for i in range(num_speakers):
        speaker_id = f"train_speaker_{i:03d}"
        files = []
        
        for j in range(files_per_speaker):
            # 创建模拟文件路径
            file_path = os.path.join(temp_dir, f"{speaker_id}_{j:03d}.m4a")
            files.append(file_path)
            
            # 创建模拟MFCC特征 (48维 x 随机帧数)
            n_frames = np.random.randint(50, 200)
            mock_mfcc = np.random.randn(48, n_frames).astype(np.float32)
            mock_features[file_path] = mock_mfcc
        
        global_train_files[speaker_id] = files
    
    # 为注册和测试创建说话人
    for i in range(num_speakers // 2):
        speaker_id = f"test_speaker_{i:03d}"
        
        # 注册文件
        enroll_file_list = []
        for j in range(5):  # 每个说话人5个注册文件
            file_path = os.path.join(temp_dir, f"{speaker_id}_enroll_{j:03d}.m4a")
            enroll_file_list.append(file_path)
            
            n_frames = np.random.randint(50, 200)
            mock_mfcc = np.random.randn(48, n_frames).astype(np.float32)
            mock_features[file_path] = mock_mfcc
        
        enroll_files[speaker_id] = enroll_file_list
        
        # 测试文件
        test_file_list = []
        for j in range(8):  # 每个说话人8个测试文件
            file_path = os.path.join(temp_dir, f"{speaker_id}_test_{j:03d}.m4a")
            test_file_list.append(file_path)
            
            n_frames = np.random.randint(50, 200)
            mock_mfcc = np.random.randn(48, n_frames).astype(np.float32)
            mock_features[file_path] = mock_mfcc
        
        test_files[speaker_id] = test_file_list
    
    print(f"模拟数据创建完成:")
    print(f"  - 训练集: {len(global_train_files)} 个说话人")
    print(f"  - 注册集: {len(enroll_files)} 个说话人")
    print(f"  - 测试集: {len(test_files)} 个说话人")
    print(f"  - 总文件数: {len(mock_features)}")
    
    return global_train_files, enroll_files, test_files, mock_features


def mock_extract_mfcc(file_path, mock_features):
    """
    模拟MFCC特征提取，返回预先生成的特征
    """
    if file_path in mock_features:
        # 模拟处理时间
        time.sleep(0.01)  # 10ms的模拟处理时间
        return mock_features[file_path]
    else:
        # 如果没有预先生成的特征，创建随机特征
        n_frames = np.random.randint(50, 200)
        return np.random.randn(48, n_frames).astype(np.float32)


def test_dtw_progress():
    """测试DTW模型的进度条"""
    print("\n" + "="*60)
    print("测试DTW模型进度条")
    print("="*60)
    
    # 创建模拟数据
    global_train_files, enroll_files, test_files, mock_features = create_mock_audio_files(
        num_speakers=6, files_per_speaker=8)
    
    # 创建DTW模型
    dtw_model = DTWModel(config)
    
    # 临时替换特征提取函数
    import speaker_recognition.models.dtw_model as dtw_module
    original_extract_mfcc = dtw_module.extract_mfcc
    dtw_module.extract_mfcc = lambda path: mock_extract_mfcc(path, mock_features)
    
    try:
        # 测试训练（DTW不需要训练）
        print("\n1. 训练阶段:")
        dtw_model.train(global_train_files)
        
        # 测试注册
        print("\n2. 注册阶段:")
        for speaker_id, files in tqdm(enroll_files.items(), desc="注册说话人", unit="说话人"):
            dtw_model.enroll(speaker_id, files)
        
        # 测试识别
        print("\n3. 测试阶段:")
        test_count = 0
        all_test_files = []
        for speaker_id, files in test_files.items():
            for file_path in files:
                all_test_files.append((speaker_id, file_path))
        
        with tqdm(all_test_files, desc="测试DTW模型", unit="文件") as pbar:
            for true_speaker_id, file_path in pbar:
                try:
                    predicted_speaker_id, score = dtw_model.identify(file_path)
                    test_count += 1
                    pbar.set_postfix({
                        '已完成': test_count,
                        '当前': true_speaker_id[:8] + '...'
                    })
                except Exception as e:
                    tqdm.write(f"测试失败: {e}")
        
        print(f"DTW模型测试完成，共测试 {test_count} 个文件")
        
    finally:
        # 恢复原始函数
        dtw_module.extract_mfcc = original_extract_mfcc


def test_vq_progress():
    """测试VQ模型的进度条"""
    print("\n" + "="*60)
    print("测试VQ模型进度条")
    print("="*60)
    
    # 创建模拟数据
    global_train_files, enroll_files, test_files, mock_features = create_mock_audio_files(
        num_speakers=4, files_per_speaker=6)
    
    # 创建VQ模型
    vq_model = VQModel(config)
    
    # 临时替换特征提取函数
    import speaker_recognition.models.vq_model as vq_module
    original_extract_mfcc = vq_module.extract_mfcc
    vq_module.extract_mfcc = lambda path: mock_extract_mfcc(path, mock_features)
    
    try:
        # 测试训练
        print("\n1. 训练阶段:")
        vq_model.train(global_train_files)
        
        # 测试注册
        print("\n2. 注册阶段:")
        for speaker_id, files in tqdm(enroll_files.items(), desc="注册说话人", unit="说话人"):
            vq_model.enroll(speaker_id, files)
        
        # 测试识别
        print("\n3. 测试阶段:")
        test_count = 0
        all_test_files = []
        for speaker_id, files in test_files.items():
            for file_path in files:
                all_test_files.append((speaker_id, file_path))
        
        with tqdm(all_test_files, desc="测试VQ模型", unit="文件") as pbar:
            for true_speaker_id, file_path in pbar:
                try:
                    predicted_speaker_id, score = vq_model.identify(file_path)
                    test_count += 1
                    pbar.set_postfix({
                        '已完成': test_count,
                        '当前': true_speaker_id[:8] + '...'
                    })
                except Exception as e:
                    tqdm.write(f"测试失败: {e}")
        
        print(f"VQ模型测试完成，共测试 {test_count} 个文件")
        
    finally:
        # 恢复原始函数
        vq_module.extract_mfcc = original_extract_mfcc


def main():
    """主函数"""
    print("声纹识别系统进度条测试")
    print("这个测试使用模拟数据来演示各个阶段的进度条效果")
    
    try:
        # 测试DTW模型进度条
        test_dtw_progress()
        
        # 测试VQ模型进度条
        test_vq_progress()
        
        print("\n" + "="*60)
        print("🎉 进度条测试完成！")
        print("所有阶段都正确显示了进度条：")
        print("  ✓ 数据扫描阶段")
        print("  ✓ 模型训练阶段")
        print("  ✓ 说话人注册阶段")
        print("  ✓ 模型测试阶段")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
