"""
性能评估模块

实现声纹识别系统的各种性能指标计算，包括准确率、等错误率(EER)、
检测错误权衡(DET)曲线等评估指标。
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from .config import config


def calculate_accuracy(true_labels: List[str], predicted_labels: List[str]) -> float:
    """
    计算闭集识别的准确率
    
    Args:
        true_labels: 真实标签列表
        predicted_labels: 预测标签列表
        
    Returns:
        准确率 (0-1之间的浮点数)
    """
    if len(true_labels) != len(predicted_labels):
        raise ValueError("真实标签和预测标签的长度不匹配")
    
    return accuracy_score(true_labels, predicted_labels)


def calculate_top_k_accuracy(true_labels: List[str], 
                           predicted_scores: List[Dict[str, float]], 
                           k: int = 5) -> float:
    """
    计算Top-K准确率
    
    Args:
        true_labels: 真实标签列表
        predicted_scores: 预测分数字典列表，每个字典包含 {speaker_id: score}
        k: Top-K中的K值
        
    Returns:
        Top-K准确率
    """
    if len(true_labels) != len(predicted_scores):
        raise ValueError("真实标签和预测分数的长度不匹配")
    
    correct = 0
    total = len(true_labels)
    
    for true_label, scores_dict in zip(true_labels, predicted_scores):
        # 按分数排序，取前K个
        sorted_speakers = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        top_k_speakers = [speaker for speaker, _ in sorted_speakers[:k]]
        
        if true_label in top_k_speakers:
            correct += 1
    
    return correct / total


def calculate_eer(target_scores: List[float], imposter_scores: List[float]) -> Tuple[float, float]:
    """
    计算等错误率(Equal Error Rate, EER)
    
    Args:
        target_scores: 目标试验(正例)的分数列表
        imposter_scores: 冒名者试验(负例)的分数列表
        
    Returns:
        (EER值, 对应的阈值)
    """
    if not target_scores or not imposter_scores:
        raise ValueError("目标分数或冒名者分数列表为空")
    
    # 合并所有分数并排序
    all_scores = sorted(target_scores + imposter_scores)
    
    # 生成阈值范围
    min_score = min(all_scores)
    max_score = max(all_scores)
    
    # 使用配置中的阈值步长
    threshold_step = getattr(config, 'EER_THRESHOLD_STEP', 0.01)
    thresholds = np.arange(min_score, max_score + threshold_step, threshold_step)
    
    fars = []  # False Acceptance Rate
    frrs = []  # False Rejection Rate
    
    for threshold in thresholds:
        # 计算FAR: 冒名者分数 >= 阈值的比例
        fa = sum(1 for score in imposter_scores if score >= threshold)
        far = fa / len(imposter_scores)
        
        # 计算FRR: 目标分数 < 阈值的比例
        fr = sum(1 for score in target_scores if score < threshold)
        frr = fr / len(target_scores)
        
        fars.append(far)
        frrs.append(frr)
    
    # 找到FAR和FRR最接近的点
    fars = np.array(fars)
    frrs = np.array(frrs)
    
    # 使用插值找到精确的EER点
    try:
        # 创建插值函数
        interp_func = interp1d(thresholds, fars - frrs, kind='linear')
        
        # 找到FAR - FRR = 0的点
        eer_threshold = brentq(interp_func, min(thresholds), max(thresholds))
        
        # 计算对应的EER值
        eer_far = np.interp(eer_threshold, thresholds, fars)
        eer_frr = np.interp(eer_threshold, thresholds, frrs)
        eer = (eer_far + eer_frr) / 2
        
    except (ValueError, RuntimeError):
        # 如果插值失败，使用最接近的点
        diff = np.abs(fars - frrs)
        min_idx = np.argmin(diff)
        eer = (fars[min_idx] + frrs[min_idx]) / 2
        eer_threshold = thresholds[min_idx]
    
    return eer, eer_threshold


def calculate_det_curve(target_scores: List[float], 
                       imposter_scores: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算检测错误权衡(DET)曲线的FAR和FRR值
    
    Args:
        target_scores: 目标试验分数列表
        imposter_scores: 冒名者试验分数列表
        
    Returns:
        (FAR数组, FRR数组)
    """
    # 合并所有分数并排序
    all_scores = sorted(target_scores + imposter_scores)
    
    # 生成阈值
    min_score = min(all_scores)
    max_score = max(all_scores)
    thresholds = np.linspace(min_score, max_score, 1000)
    
    fars = []
    frrs = []
    
    for threshold in thresholds:
        # 计算FAR和FRR
        fa = sum(1 for score in imposter_scores if score >= threshold)
        far = fa / len(imposter_scores)
        
        fr = sum(1 for score in target_scores if score < threshold)
        frr = fr / len(target_scores)
        
        fars.append(far)
        frrs.append(frr)
    
    return np.array(fars), np.array(frrs)


def plot_det_curve(target_scores: List[float], 
                  imposter_scores: List[float],
                  title: str = "DET Curve",
                  save_path: Optional[str] = None) -> None:
    """
    绘制DET曲线
    
    Args:
        target_scores: 目标试验分数列表
        imposter_scores: 冒名者试验分数列表
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    fars, frrs = calculate_det_curve(target_scores, imposter_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fars * 100, frrs * 100, 'b-', linewidth=2)
    plt.xlabel('False Acceptance Rate (%)')
    plt.ylabel('False Rejection Rate (%)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    
    # 标记EER点
    eer, _ = calculate_eer(target_scores, imposter_scores)
    plt.plot(eer * 100, eer * 100, 'ro', markersize=8, label=f'EER = {eer:.2%}')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"DET曲线已保存到: {save_path}")
    else:
        plt.show()


def plot_score_distribution(target_scores: List[float], 
                          imposter_scores: List[float],
                          title: str = "Score Distribution",
                          save_path: Optional[str] = None) -> None:
    """
    绘制分数分布直方图
    
    Args:
        target_scores: 目标试验分数列表
        imposter_scores: 冒名者试验分数列表
        title: 图表标题
        save_path: 保存路径，如果为None则显示图表
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(imposter_scores, bins=50, alpha=0.7, label='Imposter Scores', 
             color='red', density=True)
    plt.hist(target_scores, bins=50, alpha=0.7, label='Target Scores', 
             color='blue', density=True)
    
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    target_mean = np.mean(target_scores)
    target_std = np.std(target_scores)
    imposter_mean = np.mean(imposter_scores)
    imposter_std = np.std(imposter_scores)
    
    plt.axvline(target_mean, color='blue', linestyle='--', alpha=0.8,
                label=f'Target Mean: {target_mean:.2f}')
    plt.axvline(imposter_mean, color='red', linestyle='--', alpha=0.8,
                label=f'Imposter Mean: {imposter_mean:.2f}')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分数分布图已保存到: {save_path}")
    else:
        plt.show()


def generate_confusion_matrix(true_labels: List[str], 
                            predicted_labels: List[str],
                            speaker_ids: Optional[List[str]] = None) -> np.ndarray:
    """
    生成混淆矩阵
    
    Args:
        true_labels: 真实标签列表
        predicted_labels: 预测标签列表
        speaker_ids: 说话人ID列表，如果为None则自动从标签中提取
        
    Returns:
        混淆矩阵
    """
    if speaker_ids is None:
        speaker_ids = sorted(list(set(true_labels + predicted_labels)))
    
    return confusion_matrix(true_labels, predicted_labels, labels=speaker_ids)


def print_evaluation_report(true_labels: List[str], 
                          predicted_labels: List[str],
                          target_scores: Optional[List[float]] = None,
                          imposter_scores: Optional[List[float]] = None) -> Dict:
    """
    打印完整的评估报告
    
    Args:
        true_labels: 真实标签列表
        predicted_labels: 预测标签列表
        target_scores: 目标试验分数列表（可选）
        imposter_scores: 冒名者试验分数列表（可选）
        
    Returns:
        评估结果字典
    """
    results = {}
    
    # 计算准确率
    if len(true_labels) == 0:
        print("⚠️  警告: 没有成功的测试样本！")
        results['accuracy'] = 0.0
        accuracy = 0.0
    else:
        accuracy = calculate_accuracy(true_labels, predicted_labels)
        results['accuracy'] = accuracy
    
    print("=" * 50)
    print("声纹识别系统评估报告")
    print("=" * 50)
    print(f"总测试样本数: {len(true_labels)}")
    
    if len(true_labels) == 0:
        print("❌ 识别准确率: 无法计算（没有成功的测试样本）")
        print("\n可能的原因:")
        print("1. 所有说话人注册失败")
        print("2. 测试文件处理失败")
        print("3. 数据集路径配置错误")
        return results
    else:
        print(f"识别准确率: {accuracy:.2%}")
    
    # 计算EER（如果提供了分数）
    if target_scores and imposter_scores:
        eer, eer_threshold = calculate_eer(target_scores, imposter_scores)
        results['eer'] = eer
        results['eer_threshold'] = eer_threshold
        print(f"等错误率(EER): {eer:.2%}")
        print(f"EER对应阈值: {eer_threshold:.4f}")
    
    # 打印分类报告（仅当有测试样本时）
    if len(true_labels) > 0 and len(set(true_labels)) > 0:
        print("\n详细分类报告:")
        print(classification_report(true_labels, predicted_labels))
        
        # 混淆矩阵统计
        cm = generate_confusion_matrix(true_labels, predicted_labels)
        results['confusion_matrix'] = cm
        print(f"\n混淆矩阵形状: {cm.shape}")
    else:
        print("\n无法生成详细分类报告：没有有效的测试样本")
    
    print("=" * 50)
    
    return results


if __name__ == "__main__":
    # 测试评估功能
    import random
    
    # 生成模拟数据
    random.seed(42)
    
    # 模拟真实标签和预测标签
    speakers = ['speaker1', 'speaker2', 'speaker3', 'speaker4', 'speaker5']
    true_labels = [random.choice(speakers) for _ in range(100)]
    predicted_labels = [random.choice(speakers) for _ in range(100)]
    
    # 模拟分数
    target_scores = [random.gauss(2.0, 0.5) for _ in range(50)]
    imposter_scores = [random.gauss(0.0, 0.5) for _ in range(200)]
    
    # 运行评估
    results = print_evaluation_report(true_labels, predicted_labels, 
                                    target_scores, imposter_scores)
    
    print("评估模块测试完成!")
