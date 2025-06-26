"""
VQ声纹识别模型

基于矢量量化(Vector Quantization)的声纹识别算法实现。
VQ通过训练通用码本，然后为每个说话人生成特征直方图进行识别。
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm

from .base_model import BaseModel
from ..feature_extractor import extract_mfcc, extract_mfcc_from_files, normalize_features


class VQModel(BaseModel):
    """基于VQ的声纹识别模型"""
    
    def __init__(self, config_obj=None):
        """
        初始化VQ模型
        
        Args:
            config_obj: 配置对象
        """
        super().__init__(config_obj)
        self.codebook_size = getattr(self.config, 'VQ_CODEBOOK_SIZE', 256)
        self.max_iter = getattr(self.config, 'VQ_MAX_ITER', 300)
        self.n_init = getattr(self.config, 'VQ_N_INIT', 10)
        self.codebook = None
        self.kmeans_model = None
        
    def train(self, file_paths: Dict[str, List[str]]) -> None:
        """
        训练VQ码本
        
        Args:
            file_paths: 训练文件路径字典 {speaker_id: [file_path1, file_path2, ...]}
        """
        print("开始训练VQ码本...")
        
        # 收集所有训练文件路径
        all_train_files = []
        for speaker_id, files in file_paths.items():
            all_train_files.extend(files)
        
        print(f"总计 {len(all_train_files)} 个训练文件")
        
        # 提取所有训练文件的MFCC特征
        all_features = []
        
        for file_path in tqdm(all_train_files, desc="提取训练特征"):
            try:
                mfcc = extract_mfcc(file_path)
                mfcc = normalize_features(mfcc)
                # 转置为 (n_frames, n_features) 并添加到特征列表
                all_features.append(mfcc.T)
            except Exception as e:
                print(f"警告: 提取特征失败 {file_path}: {e}")
                continue
        
        if not all_features:
            raise ValueError("没有成功提取到任何训练特征")
        
        # 将所有特征合并为一个大矩阵
        combined_features = np.vstack(all_features)
        print(f"合并特征矩阵形状: {combined_features.shape}")
        
        # 使用KMeans进行聚类训练码本
        print(f"开始KMeans聚类，码本大小: {self.codebook_size}")
        
        self.kmeans_model = KMeans(
            n_clusters=self.codebook_size,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.config.RANDOM_SEED,
            verbose=1 if self.config.VERBOSE else 0
        )
        
        self.kmeans_model.fit(combined_features)
        
        # 保存码本（聚类中心）
        self.codebook = self.kmeans_model.cluster_centers_
        self.is_trained = True
        
        print(f"VQ码本训练完成，码本形状: {self.codebook.shape}")
    
    def _quantize_features(self, features: np.ndarray) -> np.ndarray:
        """
        将特征向量量化为码字索引
        
        Args:
            features: 特征矩阵，形状为 (n_frames, n_features)
            
        Returns:
            码字索引数组，形状为 (n_frames,)
        """
        if self.codebook is None:
            raise ValueError("码本未训练，请先调用train方法")
        
        # 使用KMeans模型预测最近的聚类中心
        indices = self.kmeans_model.predict(features)
        return indices
    
    def _compute_histogram(self, indices: np.ndarray) -> np.ndarray:
        """
        计算码字索引的直方图
        
        Args:
            indices: 码字索引数组
            
        Returns:
            归一化的直方图向量
        """
        # 计算直方图
        histogram, _ = np.histogram(indices, bins=self.codebook_size, 
                                  range=(0, self.codebook_size))
        
        # 归一化
        if np.sum(histogram) > 0:
            histogram = histogram.astype(np.float64) / np.sum(histogram)
        
        return histogram
    
    def enroll(self, speaker_id: str, file_paths: List[str]) -> None:
        """
        为说话人注册VQ模型
        
        Args:
            speaker_id: 说话人ID
            file_paths: 注册文件路径列表
        """
        if not self.is_trained:
            raise ValueError("VQ码本未训练，请先调用train方法")
        
        print(f"正在为说话人 {speaker_id} 注册VQ模型...")
        
        # 提取所有注册文件的特征并量化
        all_indices = []
        
        for file_path in tqdm(file_paths, desc=f"处理 {speaker_id} 的文件"):
            try:
                mfcc = extract_mfcc(file_path)
                mfcc = normalize_features(mfcc)
                mfcc = mfcc.T  # 转置为 (n_frames, n_features)
                
                # 量化特征
                indices = self._quantize_features(mfcc)
                all_indices.extend(indices)
                
            except Exception as e:
                print(f"警告: 处理文件失败 {file_path}: {e}")
                continue
        
        if not all_indices:
            raise ValueError(f"说话人 {speaker_id} 没有成功处理任何文件")
        
        # 计算该说话人的码字直方图
        all_indices = np.array(all_indices)
        histogram = self._compute_histogram(all_indices)
        
        # 存储该说话人的直方图模型
        self.speaker_models[speaker_id] = histogram
        
        print(f"说话人 {speaker_id} 注册完成，处理了 {len(all_indices)} 个特征向量")
    
    def identify(self, file_path: str) -> Tuple[str, float]:
        """
        识别测试文件的说话人
        
        Args:
            file_path: 测试音频文件路径
            
        Returns:
            (最可能的说话人ID, 置信度分数)
        """
        if not self.is_trained:
            raise ValueError("VQ码本未训练，请先调用train方法")
        
        if not self.speaker_models:
            raise ValueError("没有注册的说话人模型")
        
        # 提取测试文件的特征并生成直方图
        test_histogram = self._extract_test_histogram(file_path)
        
        best_speaker = None
        best_score = -np.inf
        all_scores = {}
        
        # 与每个注册说话人的直方图进行比较
        for speaker_id, speaker_histogram in self.speaker_models.items():
            # 计算余弦相似度
            similarity = cosine_similarity(
                test_histogram.reshape(1, -1),
                speaker_histogram.reshape(1, -1)
            )[0, 0]
            
            all_scores[speaker_id] = similarity
            
            if similarity > best_score:
                best_score = similarity
                best_speaker = speaker_id
        
        if best_speaker is None:
            raise RuntimeError("无法识别说话人")
        
        return best_speaker, best_score
    
    def _extract_test_histogram(self, file_path: str) -> np.ndarray:
        """
        提取测试文件的码字直方图
        
        Args:
            file_path: 测试音频文件路径
            
        Returns:
            归一化的直方图向量
        """
        try:
            # 提取MFCC特征
            mfcc = extract_mfcc(file_path)
            mfcc = normalize_features(mfcc)
            mfcc = mfcc.T  # 转置为 (n_frames, n_features)
            
            # 量化特征
            indices = self._quantize_features(mfcc)
            
            # 计算直方图
            histogram = self._compute_histogram(indices)
            
            return histogram
            
        except Exception as e:
            raise RuntimeError(f"提取测试文件直方图失败: {e}")
    
    def get_speaker_score(self, file_path: str, speaker_id: str) -> float:
        """
        计算测试文件对特定说话人的匹配分数
        
        Args:
            file_path: 测试音频文件路径
            speaker_id: 目标说话人ID
            
        Returns:
            匹配分数（余弦相似度）
        """
        if speaker_id not in self.speaker_models:
            raise ValueError(f"说话人 {speaker_id} 未注册")
        
        # 提取测试文件的直方图
        test_histogram = self._extract_test_histogram(file_path)
        
        # 计算与目标说话人的余弦相似度
        speaker_histogram = self.speaker_models[speaker_id]
        similarity = cosine_similarity(
            test_histogram.reshape(1, -1),
            speaker_histogram.reshape(1, -1)
        )[0, 0]
        
        return similarity
    
    def get_codebook_utilization(self) -> Dict[str, float]:
        """
        分析码本利用率
        
        Returns:
            码本利用统计信息
        """
        if not self.speaker_models:
            return {}
        
        # 统计所有说话人直方图中非零码字的数量
        total_codewords_used = set()
        speaker_utilization = {}
        
        for speaker_id, histogram in self.speaker_models.items():
            used_codewords = np.where(histogram > 0)[0]
            total_codewords_used.update(used_codewords)
            speaker_utilization[speaker_id] = len(used_codewords) / self.codebook_size
        
        overall_utilization = len(total_codewords_used) / self.codebook_size
        
        return {
            'overall_utilization': overall_utilization,
            'total_used_codewords': len(total_codewords_used),
            'speaker_utilization': speaker_utilization
        }
    
    def _get_model_specific_data(self) -> Dict:
        """获取VQ模型特定的数据用于保存"""
        return {
            'codebook_size': self.codebook_size,
            'max_iter': self.max_iter,
            'n_init': self.n_init,
            'codebook': self.codebook,
            'kmeans_model': self.kmeans_model
        }
    
    def _load_model_specific_data(self, save_data: Dict) -> None:
        """加载VQ模型特定的数据"""
        self.codebook_size = save_data.get('codebook_size', 256)
        self.max_iter = save_data.get('max_iter', 300)
        self.n_init = save_data.get('n_init', 10)
        self.codebook = save_data.get('codebook')
        self.kmeans_model = save_data.get('kmeans_model')


if __name__ == "__main__":
    # 测试VQ模型
    from ..config import config
    
    # 创建VQ模型实例
    vq_model = VQModel(config)
    
    print("VQ模型创建成功")
    print(f"模型信息: {vq_model}")
    print(f"码本大小: {vq_model.codebook_size}")
    
    # 这里可以添加更多的测试代码
