"""
声纹识别系统主程序

串联所有模块实现完整的训练、注册和测试流程。
支持DTW、VQ、GMM三种算法的训练和评估。
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from speaker_recognition.config import config
from speaker_recognition.data_loader import get_dataset_split, load_test_trials
from speaker_recognition.models import DTWModel, VQModel, GMMModel
from speaker_recognition.evaluate import (
    calculate_accuracy, calculate_eer, print_evaluation_report,
    plot_det_curve, plot_score_distribution
)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='声纹识别系统')
    
    parser.add_argument('--model', type=str, choices=['dtw', 'vq', 'gmm'], 
                       default='gmm', help='选择识别算法 (默认: gmm)')
    
    parser.add_argument('--mode', type=str, 
                       choices=['train', 'test', 'full'], default='full',
                       help='运行模式: train(仅训练), test(仅测试), full(完整流程)')
    
    parser.add_argument('--data-path', type=str, default=None,
                       help='数据集路径 (默认使用配置文件中的路径)')
    
    parser.add_argument('--save-model', type=str, default=None,
                       help='模型保存路径 (默认使用配置文件中的路径)')
    
    parser.add_argument('--load-model', type=str, default=None,
                       help='加载已训练模型的路径')
    
    parser.add_argument('--output-dir', type=str, default='results',
                       help='结果输出目录 (默认: results)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细日志')
    
    return parser.parse_args()


def create_model(model_type: str):
    """创建指定类型的模型"""
    if model_type == 'dtw':
        return DTWModel(config)
    elif model_type == 'vq':
        return VQModel(config)
    elif model_type == 'gmm':
        return GMMModel(config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def train_model(model, global_train_files: Dict[str, List[str]], 
               model_type: str, save_path: str = None):
    """训练模型"""
    print(f"\n{'='*50}")
    print(f"开始训练 {model_type.upper()} 模型")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # 训练模型
    model.train(global_train_files)
    
    training_time = time.time() - start_time
    print(f"\n{model_type.upper()} 模型训练完成，耗时: {training_time:.2f} 秒")
    
    # 保存模型
    if save_path:
        model.save(save_path)
        print(f"模型已保存到: {save_path}")
    
    return model


def enroll_speakers(model, enroll_files: Dict[str, List[str]], model_type: str):
    """注册说话人"""
    print(f"\n{'='*50}")
    print(f"开始注册说话人 ({model_type.upper()} 模型)")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    for speaker_id, files in enroll_files.items():
        print(f"正在注册说话人: {speaker_id}")
        model.enroll(speaker_id, files)
    
    enrollment_time = time.time() - start_time
    print(f"\n说话人注册完成，共注册 {len(enroll_files)} 个说话人")
    print(f"注册耗时: {enrollment_time:.2f} 秒")
    
    return model


def test_model(model, test_files: Dict[str, List[str]], 
              model_type: str) -> Tuple[List[str], List[str], List[float], List[float]]:
    """测试模型"""
    print(f"\n{'='*50}")
    print(f"开始测试 {model_type.upper()} 模型")
    print(f"{'='*50}")
    
    true_labels = []
    predicted_labels = []
    target_scores = []
    imposter_scores = []
    
    start_time = time.time()
    total_tests = sum(len(files) for files in test_files.values())
    current_test = 0
    
    # 对每个测试文件进行识别
    for true_speaker_id, files in test_files.items():
        for file_path in files:
            current_test += 1
            
            if current_test % 10 == 0:
                print(f"进度: {current_test}/{total_tests} "
                      f"({current_test/total_tests*100:.1f}%)")
            
            try:
                # 识别说话人
                predicted_speaker_id, score = model.identify(file_path)
                
                true_labels.append(true_speaker_id)
                predicted_labels.append(predicted_speaker_id)
                
                # 计算目标分数和冒名者分数
                target_score = model.get_speaker_score(file_path, true_speaker_id)
                target_scores.append(target_score)
                
                # 随机选择一个其他说话人作为冒名者
                other_speakers = [sid for sid in test_files.keys() if sid != true_speaker_id]
                if other_speakers:
                    import random
                    imposter_id = random.choice(other_speakers)
                    imposter_score = model.get_speaker_score(file_path, imposter_id)
                    imposter_scores.append(imposter_score)
                
            except Exception as e:
                print(f"警告: 测试文件 {file_path} 失败: {e}")
                continue
    
    test_time = time.time() - start_time
    print(f"\n模型测试完成，共测试 {len(true_labels)} 个文件")
    print(f"测试耗时: {test_time:.2f} 秒")
    
    return true_labels, predicted_labels, target_scores, imposter_scores


def save_results(results: Dict, output_dir: str, model_type: str):
    """保存结果到文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存评估结果
    result_file = os.path.join(output_dir, f"{model_type}_results.txt")
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"{model_type.upper()} 模型评估结果\n")
        f.write("=" * 50 + "\n")
        f.write(f"识别准确率: {results.get('accuracy', 0):.2%}\n")
        if 'eer' in results:
            f.write(f"等错误率(EER): {results['eer']:.2%}\n")
            f.write(f"EER对应阈值: {results['eer_threshold']:.4f}\n")
    
    print(f"结果已保存到: {result_file}")


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置详细日志
    if args.verbose:
        config.VERBOSE = True
    
    # 验证数据路径
    data_path = args.data_path or config.VOXCELEB_PATH
    if not config.validate_paths():
        print(f"警告: 数据集路径可能不存在: {data_path}")
        print("请检查配置文件中的VOXCELEB_PATH设置")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("声纹识别系统启动")
    print(f"模型类型: {args.model.upper()}")
    print(f"运行模式: {args.mode}")
    print(f"数据路径: {data_path}")
    
    # 创建模型
    model = create_model(args.model)
    print(f"模型创建成功: {model}")
    
    # 加载数据集
    if args.mode in ['train', 'full']:
        print("\n正在加载和划分数据集...")
        global_train_files, enroll_files, test_files = get_dataset_split(data_path)
    
    # 训练阶段
    if args.mode in ['train', 'full']:
        if args.load_model:
            print(f"加载预训练模型: {args.load_model}")
            model.load(args.load_model)
        else:
            model = train_model(model, global_train_files, args.model, args.save_model)
        
        # 注册说话人
        model = enroll_speakers(model, enroll_files, args.model)
    
    # 测试阶段
    if args.mode in ['test', 'full']:
        if args.mode == 'test' and args.load_model:
            print(f"加载训练好的模型: {args.load_model}")
            model.load(args.load_model)
            # 需要重新加载数据集用于测试
            _, enroll_files, test_files = get_dataset_split(data_path)
        
        # 执行测试
        true_labels, predicted_labels, target_scores, imposter_scores = test_model(
            model, test_files, args.model)
        
        # 评估结果
        print(f"\n{'='*50}")
        print("评估结果")
        print(f"{'='*50}")
        
        results = print_evaluation_report(
            true_labels, predicted_labels, target_scores, imposter_scores)
        
        # 保存结果
        save_results(results, args.output_dir, args.model)
        
        # 绘制图表
        if target_scores and imposter_scores:
            det_curve_path = os.path.join(args.output_dir, f"{args.model}_det_curve.png")
            plot_det_curve(target_scores, imposter_scores, 
                          f"{args.model.upper()} Model DET Curve", det_curve_path)
            
            score_dist_path = os.path.join(args.output_dir, f"{args.model}_score_distribution.png")
            plot_score_distribution(target_scores, imposter_scores,
                                  f"{args.model.upper()} Model Score Distribution", 
                                  score_dist_path)
    
    print(f"\n{'='*50}")
    print("程序执行完成")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
