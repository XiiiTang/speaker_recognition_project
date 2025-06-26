"""
声纹识别系统测试脚本

用于验证系统的基本功能，包括模块导入、配置加载、模型创建等。
"""

import os
import sys
import traceback

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        # 测试配置模块
        from speaker_recognition.config import config
        print("✓ 配置模块导入成功")
        
        # 测试数据加载模块
        from speaker_recognition.data_loader import get_dataset_split
        print("✓ 数据加载模块导入成功")
        
        # 测试特征提取模块
        from speaker_recognition.feature_extractor import extract_mfcc
        print("✓ 特征提取模块导入成功")
        
        # 测试模型模块
        from speaker_recognition.models import DTWModel, VQModel, GMMModel
        print("✓ 模型模块导入成功")
        
        # 测试评估模块
        from speaker_recognition.evaluate import calculate_accuracy, calculate_eer
        print("✓ 评估模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        traceback.print_exc()
        return False


def test_config():
    """测试配置"""
    print("\n测试配置...")
    
    try:
        from speaker_recognition.config import config
        
        print(f"✓ VoxCeleb路径: {config.VOXCELEB_PATH}")
        print(f"✓ 模型保存路径: {config.MODEL_SAVE_PATH}")
        print(f"✓ 采样率: {config.SAMPLE_RATE}")
        print(f"✓ MFCC维度: {config.N_MFCC}")
        print(f"✓ VQ码本大小: {config.VQ_CODEBOOK_SIZE}")
        print(f"✓ GMM分量数: {config.GMM_N_COMPONENTS}")
        
        # 检查目录是否存在
        if os.path.exists(config.MODEL_SAVE_PATH):
            print("✓ 模型保存目录存在")
        else:
            print("! 模型保存目录不存在，但会自动创建")
        
        return True
        
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    try:
        from speaker_recognition.models import DTWModel, VQModel, GMMModel
        from speaker_recognition.config import config
        
        # 测试DTW模型
        dtw_model = DTWModel(config)
        print(f"✓ DTW模型创建成功: {dtw_model}")
        
        # 测试VQ模型
        vq_model = VQModel(config)
        print(f"✓ VQ模型创建成功: {vq_model}")
        
        # 测试GMM模型
        gmm_model = GMMModel(config)
        print(f"✓ GMM模型创建成功: {gmm_model}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        traceback.print_exc()
        return False


def test_feature_extraction():
    """测试特征提取（使用模拟数据）"""
    print("\n测试特征提取...")
    
    try:
        import numpy as np
        from speaker_recognition.feature_extractor import normalize_features
        
        # 创建模拟MFCC特征
        mock_mfcc = np.random.randn(16, 100)  # 16维MFCC，100帧
        
        # 测试特征归一化
        normalized_mfcc = normalize_features(mock_mfcc)
        print(f"✓ 特征归一化成功，输入形状: {mock_mfcc.shape}, 输出形状: {normalized_mfcc.shape}")
        
        # 检查归一化效果
        mean_vals = np.mean(normalized_mfcc, axis=1)
        std_vals = np.std(normalized_mfcc, axis=1)
        print(f"✓ 归一化后均值范围: [{np.min(mean_vals):.3f}, {np.max(mean_vals):.3f}]")
        print(f"✓ 归一化后标准差范围: [{np.min(std_vals):.3f}, {np.max(std_vals):.3f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ 特征提取测试失败: {e}")
        traceback.print_exc()
        return False


def test_evaluation_functions():
    """测试评估函数"""
    print("\n测试评估函数...")
    
    try:
        from speaker_recognition.evaluate import calculate_accuracy, calculate_eer
        import random
        
        # 生成模拟数据
        random.seed(42)
        
        # 测试准确率计算
        true_labels = ['A', 'B', 'C', 'A', 'B']
        pred_labels = ['A', 'B', 'A', 'A', 'B']
        accuracy = calculate_accuracy(true_labels, pred_labels)
        print(f"✓ 准确率计算成功: {accuracy:.2%}")
        
        # 测试EER计算
        target_scores = [random.gauss(2.0, 0.5) for _ in range(50)]
        imposter_scores = [random.gauss(0.0, 0.5) for _ in range(100)]
        eer, threshold = calculate_eer(target_scores, imposter_scores)
        print(f"✓ EER计算成功: {eer:.2%}, 阈值: {threshold:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 评估函数测试失败: {e}")
        traceback.print_exc()
        return False


def test_data_structure():
    """测试数据结构"""
    print("\n测试数据结构...")
    
    try:
        # 检查数据目录
        data_dir = "data"
        if os.path.exists(data_dir):
            print(f"✓ 数据目录存在: {data_dir}")
            
            # 列出数据目录内容
            contents = os.listdir(data_dir)
            print(f"✓ 数据目录内容: {contents}")
        else:
            print(f"! 数据目录不存在: {data_dir}")
        
        # 检查保存目录
        saved_models_dir = "saved_models"
        if os.path.exists(saved_models_dir):
            print(f"✓ 模型保存目录存在: {saved_models_dir}")
        else:
            print(f"! 模型保存目录不存在: {saved_models_dir}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据结构测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("声纹识别系统测试")
    print("=" * 60)
    
    tests = [
        ("模块导入", test_imports),
        ("配置测试", test_config),
        ("模型创建", test_model_creation),
        ("特征提取", test_feature_extraction),
        ("评估函数", test_evaluation_functions),
        ("数据结构", test_data_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"运行测试: {test_name}")
        print(f"{'-' * 40}")
        
        if test_func():
            passed += 1
            print(f"✓ {test_name} 测试通过")
        else:
            print(f"✗ {test_name} 测试失败")
    
    print(f"\n{'=' * 60}")
    print(f"测试总结: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统基本功能正常。")
        print("\n下一步:")
        print("1. 确保VoxCeleb2数据集路径正确配置")
        print("2. 运行 'python main.py --model dtw --mode full' 开始完整测试")
        print("3. 或运行 'python main.py --help' 查看所有选项")
    else:
        print("⚠️  部分测试失败，请检查错误信息并修复问题。")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
