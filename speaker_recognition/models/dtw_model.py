"""
DTW声纹识别模型

基于动态时间规整(Dynamic Time Warping)的声纹识别算法实现。
DTW通过计算测试语音与注册模板之间的最小累积距离来进行说话人识别。
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import cdist
from tqdm import tqdm

from .base_model import BaseModel
from ..feature_extractor import extract_mfcc, extract_mfcc_from_files, normalize_features


class DTWModel(BaseModel):
    """基于DTW的声纹识别模型"""
    
    def __init__(self, config_obj=None):
        """
        初始化DTW模型
        
        Args:
            config_obj: 配置对象
        """
        super().__init__(config_obj)
        self.distance_metric = getattr(self.config, 'DTW_DISTANCE_METRIC', 'euclidean')
        
    def train(self, file_paths: Dict[str, List[str]]) -> None:
        """
        DTW模型不需要全局训练阶段
        
        Args:
            file_paths: 训练文件路径字典（DTW中不使用）
        """
        print("DTW模型不需要全局训练阶段")
        self.is_trained = True
    
    def enroll(self, speaker_id: str, file_paths: List[str]) -> None:
        """
        为说话人注册DTW模板
        
        Args:
            speaker_id: 说话人ID
            file_paths: 注册文件路径列表
        """
        print(f"正在为说话人 {speaker_id} 注册DTW模板...")
        
        # 提取所有注册文件的MFCC特征
        mfcc_templates = []
        
        for file_path in tqdm(file_paths, desc=f"提取 {speaker_id} 的特征"):
            try:
                mfcc = extract_mfcc(file_path)
                # 特征归一化
                mfcc = normalize_features(mfcc)
                # 转置为 (n_frames, n_features) 格式，便于DTW计算
                mfcc = mfcc.T
                mfcc_templates.append(mfcc)
            except Exception as e:
                print(f"警告: 提取特征失败 {file_path}: {e}")
                continue
        
        if not mfcc_templates:
            raise ValueError(f"说话人 {speaker_id} 没有成功提取到任何特征")
        
        # 存储该说话人的所有MFCC模板
        self.speaker_models[speaker_id] = mfcc_templates
        
        print(f"说话人 {speaker_id} 注册完成，共 {len(mfcc_templates)} 个模板")
    
    def _compute_dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """
        计算两个序列之间的DTW距离
        
        Args:
            seq1: 第一个序列，形状为 (n_frames1, n_features)
            seq2: 第二个序列，形状为 (n_frames2, n_features)
            
        Returns:
            DTW距离
        """
        # 计算帧间距离矩阵
        distance_matrix = cdist(seq1, seq2, metric=self.distance_metric)
        
        n1, n2 = distance_matrix.shape
        
        # 初始化DTW累积距离矩阵
        dtw_matrix = np.full((n1 + 1, n2 + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # 动态规划计算DTW
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                cost = distance_matrix[i-1, j-1]
                
                # 三种路径选择：对角线、垂直、水平
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j-1],  # 对角线
                    dtw_matrix[i-1, j],    # 垂直
                    dtw_matrix[i, j-1]     # 水平
                )
        
        # 返回归一化的DTW距离
        return dtw_matrix[n1, n2] / (n1 + n2)
    
    def _compute_dtw_distance_optimized(self, seq1: np.ndarray, seq2: np.ndarray, 
                                      window_size: int = None) -> float:
        """
        计算DTW距离的优化版本（带约束窗口）
        
        Args:
            seq1: 第一个序列
            seq2: 第二个序列
            window_size: 约束窗口大小，如果为None则不使用约束
            
        Returns:
            DTW距离
        """
        if window_size is None:
            return self._compute_dtw_distance(seq1, seq2)
        
        # 计算帧间距离矩阵
        distance_matrix = cdist(seq1, seq2, metric=self.distance_metric)
        
        n1, n2 = distance_matrix.shape
        
        # 初始化DTW累积距离矩阵
        dtw_matrix = np.full((n1 + 1, n2 + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # 带约束窗口的DTW计算
        for i in range(1, n1 + 1):
            # 计算约束窗口范围
            j_start = max(1, i - window_size)
            j_end = min(n2 + 1, i + window_size + 1)
            
            for j in range(j_start, j_end):
                cost = distance_matrix[i-1, j-1]
                
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j-1],  # 对角线
                    dtw_matrix[i-1, j],    # 垂直
                    dtw_matrix[i, j-1]     # 水平
                )
        
        return dtw_matrix[n1, n2] / (n1 + n2)
    
    def identify(self, file_path: str) -> Tuple[str, float]:
        """
        识别测试文件的说话人
        
        Args:
            file_path: 测试音频文件路径
            
        Returns:
            (最可能的说话人ID, 置信度分数)
        """
        if not self.speaker_models:
            raise ValueError("没有注册的说话人模板")
        
        # 提取测试文件的MFCC特征
        try:
            test_mfcc = extract_mfcc(file_path)
            test_mfcc = normalize_features(test_mfcc)
            test_mfcc = test_mfcc.T  # 转置为 (n_frames, n_features)
        except Exception as e:
            raise RuntimeError(f"提取测试文件特征失败: {e}")
        
        best_speaker = None
        min_distance = np.inf
        all_distances = {}
        
        # 与每个注册说话人的模板进行比较
        for speaker_id, templates in self.speaker_models.items():
            speaker_distances = []
            
            # 计算与该说话人所有模板的DTW距离
            for template in templates:
                try:
                    distance = self._compute_dtw_distance(test_mfcc, template)
                    speaker_distances.append(distance)
                except Exception as e:
                    print(f"警告: 计算DTW距离失败 {speaker_id}: {e}")
                    continue
            
            if speaker_distances:
                # 使用最小距离作为该说话人的分数
                min_speaker_distance = min(speaker_distances)
                all_distances[speaker_id] = min_speaker_distance
                
                if min_speaker_distance < min_distance:
                    min_distance = min_speaker_distance
                    best_speaker = speaker_id
        
        if best_speaker is None:
            raise RuntimeError("无法计算任何说话人的DTW距离")
        
        # 将距离转换为置信度分数（距离越小，置信度越高）
        confidence_score = -min_distance  # 负距离作为分数
        
        return best_speaker, confidence_score
    
    def get_speaker_score(self, file_path: str, speaker_id: str) -> float:
        """
        计算测试文件对特定说话人的匹配分数
        
        Args:
            file_path: 测试音频文件路径
            speaker_id: 目标说话人ID
            
        Returns:
            匹配分数（负DTW距离）
        """
        if speaker_id not in self.speaker_models:
            raise ValueError(f"说话人 {speaker_id} 未注册")
        
        # 提取测试文件的MFCC特征
        try:
            test_mfcc = extract_mfcc(file_path)
            test_mfcc = normalize_features(test_mfcc)
            test_mfcc = test_mfcc.T
        except Exception as e:
            raise RuntimeError(f"提取测试文件特征失败: {e}")
        
        # 计算与该说话人所有模板的DTW距离
        templates = self.speaker_models[speaker_id]
        distances = []
        
        for template in templates:
            try:
                distance = self._compute_dtw_distance(test_mfcc, template)
                distances.append(distance)
            except Exception as e:
                print(f"警告: 计算DTW距离失败: {e}")
                continue
        
        if not distances:
            raise RuntimeError(f"无法计算说话人 {speaker_id} 的DTW距离")
        
        # 返回最小距离的负值作为分数
        return -min(distances)
    
    def _get_model_specific_data(self) -> Dict:
        """获取DTW模型特定的数据用于保存"""
        return {
            'distance_metric': self.distance_metric
        }
    
    def _load_model_specific_data(self, save_data: Dict) -> None:
        """加载DTW模型特定的数据"""
        self.distance_metric = save_data.get('distance_metric', 'euclidean')


if __name__ == "__main__":
    # 测试DTW模型
    from ..config import config
    
    # 创建DTW模型实例
    dtw_model = DTWModel(config)
    
    print("DTW模型创建成功")
    print(f"模型信息: {dtw_model}")
    
    # 这里可以添加更多的测试代码
