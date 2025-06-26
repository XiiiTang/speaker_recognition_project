"""
快速测试脚本 - 验证基本导入和功能
"""

def test_basic_imports():
    """测试基本导入"""
    print("测试基本导入...")
    
    try:
        # 测试numpy
        import numpy as np
        print("✓ numpy导入成功")
        
        # 测试scipy
        import scipy
        print("✓ scipy导入成功")
        
        # 测试sklearn
        import sklearn
        print("✓ sklearn导入成功")
        
        # 测试配置
        from speaker_recognition.config import config
        print("✓ 配置模块导入成功")
        print(f"  - VoxCeleb路径: {config.VOXCELEB_PATH}")
        print(f"  - 采样率: {config.SAMPLE_RATE}")
        
        # 测试模型导入
        from speaker_recognition.models.base_model import BaseModel
        print("✓ 基础模型导入成功")
        
        from speaker_recognition.models.dtw_model import DTWModel
        print("✓ DTW模型导入成功")
        
        from speaker_recognition.models.vq_model import VQModel
        print("✓ VQ模型导入成功")
        
        from speaker_recognition.models.gmm_model import GMMModel
        print("✓ GMM模型导入成功")
        
        # 测试模型创建
        dtw = DTWModel(config)
        print(f"✓ DTW模型创建成功: {dtw}")
        
        vq = VQModel(config)
        print(f"✓ VQ模型创建成功: {vq}")
        
        gmm = GMMModel(config)
        print(f"✓ GMM模型创建成功: {gmm}")
        
        return True
        
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_functions():
    """测试特征处理函数"""
    print("\n测试特征处理函数...")
    
    try:
        import numpy as np
        from speaker_recognition.feature_extractor import normalize_features
        
        # 创建测试数据
        test_features = np.random.randn(16, 100)
        print(f"✓ 创建测试特征: {test_features.shape}")
        
        # 测试归一化
        normalized = normalize_features(test_features)
        print(f"✓ 特征归一化成功: {normalized.shape}")
        
        # 检查归一化效果
        mean_vals = np.mean(normalized, axis=1)
        std_vals = np.std(normalized, axis=1)
        print(f"✓ 归一化后均值接近0: {np.allclose(mean_vals, 0, atol=1e-10)}")
        print(f"✓ 归一化后标准差接近1: {np.allclose(std_vals, 1, atol=1e-10)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 特征处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation_functions():
    """测试评估函数"""
    print("\n测试评估函数...")
    
    try:
        from speaker_recognition.evaluate import calculate_accuracy, calculate_eer
        
        # 测试准确率
        true_labels = ['A', 'B', 'C', 'A', 'B']
        pred_labels = ['A', 'B', 'A', 'A', 'B']
        accuracy = calculate_accuracy(true_labels, pred_labels)
        print(f"✓ 准确率计算: {accuracy:.2%}")
        
        # 测试EER
        import random
        random.seed(42)
        target_scores = [random.gauss(1.0, 0.3) for _ in range(50)]
        imposter_scores = [random.gauss(-0.5, 0.3) for _ in range(100)]
        
        eer, threshold = calculate_eer(target_scores, imposter_scores)
        print(f"✓ EER计算: {eer:.2%}, 阈值: {threshold:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 评估函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 50)
    print("声纹识别系统快速测试")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_feature_functions,
        test_evaluation_functions
    ]
    
    passed = 0
    for test_func in tests:
        if test_func():
            passed += 1
        else:
            break  # 如果有测试失败，停止后续测试
    
    print(f"\n{'=' * 50}")
    if passed == len(tests):
        print("🎉 所有基本测试通过！")
        print("系统基本功能正常，可以进行进一步测试。")
    else:
        print(f"⚠️  测试失败 ({passed}/{len(tests)} 通过)")
        print("请检查错误信息并修复问题。")
    print("=" * 50)


if __name__ == "__main__":
    main()
