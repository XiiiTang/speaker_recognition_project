"""
GMM声纹识别模型

基于高斯混合模型-通用背景模型(GMM-UBM)的声纹识别算法实现。
通过训练通用背景模型(UBM)，然后使用MAP自适应为每个说话人生成个性化模型。
"""

import numpy as np
from typing import Dict, List, Tuple
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
from tqdm import tqdm
import warnings

from .base_model import BaseModel
from ..feature_extractor import extract_mfcc, extract_mfcc_from_files, normalize_features

# 忽略sklearn的一些警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class GMMModel(BaseModel):
    """基于GMM-UBM的声纹识别模型"""
    
    def __init__(self, config_obj=None):
        """
        初始化GMM模型
        
        Args:
            config_obj: 配置对象
        """
        super().__init__(config_obj)
        self.n_components = getattr(self.config, 'GMM_N_COMPONENTS', 256)
        self.covariance_type = getattr(self.config, 'GMM_COVARIANCE_TYPE', 'diag')
        self.max_iter = getattr(self.config, 'GMM_MAX_ITER', 100)
        self.tol = getattr(self.config, 'GMM_TOL', 1e-3)
        self.relevance_factor = getattr(self.config, 'MAP_RELEVANCE_FACTOR', 16.0)
        
        self.ubm = None  # 通用背景模型
        
    def train(self, file_paths: Dict[str, List[str]]) -> None:
        """
        训练通用背景模型(UBM)
        
        Args:
            file_paths: 训练文件路径字典 {speaker_id: [file_path1, file_path2, ...]}
        """
        print("开始训练GMM通用背景模型(UBM)...")
        
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
        
        # 训练GMM-UBM
        print(f"开始训练UBM，高斯分量数: {self.n_components}")
        
        self.ubm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.config.RANDOM_SEED,
            verbose=1 if self.config.VERBOSE else 0,
            verbose_interval=10
        )
        
        self.ubm.fit(combined_features)
        self.is_trained = True
        
        print(f"UBM训练完成，收敛状态: {self.ubm.converged_}")
        print(f"最终对数似然: {self.ubm.score(combined_features[:1000]):.2f}")  # 使用部分数据评估

    def _map_adaptation(self, speaker_features: np.ndarray) -> GaussianMixture:
        """
        使用MAP自适应为说话人创建个性化GMM模型

        Args:
            speaker_features: 说话人的特征矩阵，形状为 (n_frames, n_features)

        Returns:
            自适应后的GMM模型
        """
        if self.ubm is None:
            raise ValueError("UBM未训练，请先调用train方法")

        # 检查特征数量是否足够
        n_samples = speaker_features.shape[0]
        if n_samples < self.n_components:
            print(f"警告: 特征数量({n_samples})少于GMM组件数({self.n_components})，使用适当的组件数")
            # 动态调整组件数量
            effective_components = min(self.n_components, max(1, n_samples // 2))
        else:
            effective_components = self.n_components

        # 使用UBM预测后验概率
        responsibilities = self.ubm.predict_proba(speaker_features)
        
        # 如果需要，截取前effective_components个组件
        if effective_components < self.n_components:
            responsibilities = responsibilities[:, :effective_components]

        # 计算每个高斯分量的统计量
        n_k = np.sum(responsibilities, axis=0)  # 每个分量的软计数

        # 避免除零
        n_k = np.maximum(n_k, 1e-10)

        # 计算自适应因子
        alpha_k = n_k / (n_k + self.relevance_factor)

        # 创建新的GMM模型，使用UBM的参数初始化
        adapted_gmm = GaussianMixture(
            n_components=effective_components,
            covariance_type='diag',  # 固定使用对角协方差
            max_iter=1,
            random_state=self.config.RANDOM_SEED
        )

        # 使用少量数据拟合以初始化参数
        adapted_gmm.fit(speaker_features[:min(100, len(speaker_features))])

        # 复制UBM的参数（前effective_components个）
        if effective_components < self.n_components:
            adapted_gmm.weights_ = self.ubm.weights_[:effective_components].copy()
            adapted_gmm.means_ = self.ubm.means_[:effective_components].copy()  
            adapted_gmm.covariances_ = self.ubm.covariances_[:effective_components].copy()
        else:
            adapted_gmm.weights_ = self.ubm.weights_.copy()
            adapted_gmm.means_ = self.ubm.means_.copy()
            adapted_gmm.covariances_ = self.ubm.covariances_.copy()
        
        # 重新归一化权重
        adapted_gmm.weights_ = adapted_gmm.weights_ / np.sum(adapted_gmm.weights_)

        # MAP自适应均值
        for k in range(effective_components):
            if n_k[k] > 0:
                # 计算该分量的期望统计量
                weighted_sum = np.sum(responsibilities[:, k:k+1] * speaker_features, axis=0)
                empirical_mean = weighted_sum / n_k[k]

                # MAP自适应
                adapted_gmm.means_[k] = (
                    alpha_k[k] * empirical_mean +
                    (1 - alpha_k[k]) * self.ubm.means_[k]
                )

        adapted_gmm.converged_ = True

        return adapted_gmm

    def enroll(self, speaker_id: str, file_paths: List[str]) -> None:
        """
        为说话人注册GMM模型

        Args:
            speaker_id: 说话人ID
            file_paths: 注册文件路径列表
        """
        if not self.is_trained:
            raise ValueError("UBM未训练，请先调用train方法")

        print(f"正在为说话人 {speaker_id} 注册GMM模型...")

        # 提取所有注册文件的特征
        all_features = []

        for file_path in tqdm(file_paths, desc=f"处理 {speaker_id} 的文件"):
            try:
                mfcc = extract_mfcc(file_path)
                mfcc = normalize_features(mfcc)
                mfcc = mfcc.T  # 转置为 (n_frames, n_features)
                all_features.append(mfcc)
            except Exception as e:
                print(f"警告: 处理文件失败 {file_path}: {e}")
                continue

        if not all_features:
            raise ValueError(f"说话人 {speaker_id} 没有成功处理任何文件")

        # 合并该说话人的所有特征
        speaker_features = np.vstack(all_features)

        # 使用MAP自适应创建个性化模型
        adapted_gmm = self._map_adaptation(speaker_features)

        # 存储该说话人的模型
        self.speaker_models[speaker_id] = adapted_gmm

        print(f"说话人 {speaker_id} 注册完成，处理了 {speaker_features.shape[0]} 个特征向量")

    def identify(self, file_path: str) -> Tuple[str, float]:
        """
        识别测试文件的说话人

        Args:
            file_path: 测试音频文件路径

        Returns:
            (最可能的说话人ID, 置信度分数)
        """
        if not self.is_trained:
            raise ValueError("UBM未训练，请先调用train方法")

        if not self.speaker_models:
            raise ValueError("没有注册的说话人模型")

        # 提取测试文件的特征
        try:
            test_mfcc = extract_mfcc(file_path)
            test_mfcc = normalize_features(test_mfcc)
            test_mfcc = test_mfcc.T  # 转置为 (n_frames, n_features)
        except Exception as e:
            raise RuntimeError(f"提取测试文件特征失败: {e}")

        best_speaker = None
        best_score = -np.inf
        all_scores = {}

        # 与每个注册说话人的GMM模型进行比较
        for speaker_id, speaker_gmm in self.speaker_models.items():
            try:
                # 计算对数似然分数
                log_likelihood = speaker_gmm.score(test_mfcc)
                all_scores[speaker_id] = log_likelihood

                if log_likelihood > best_score:
                    best_score = log_likelihood
                    best_speaker = speaker_id

            except Exception as e:
                print(f"警告: 计算说话人 {speaker_id} 的分数失败: {e}")
                continue

        if best_speaker is None:
            raise RuntimeError("无法识别说话人")

        return best_speaker, best_score

    def get_speaker_score(self, file_path: str, speaker_id: str) -> float:
        """
        计算测试文件对特定说话人的匹配分数

        Args:
            file_path: 测试音频文件路径
            speaker_id: 目标说话人ID

        Returns:
            匹配分数（对数似然）
        """
        if speaker_id not in self.speaker_models:
            raise ValueError(f"说话人 {speaker_id} 未注册")

        # 提取测试文件的特征
        try:
            test_mfcc = extract_mfcc(file_path)
            test_mfcc = normalize_features(test_mfcc)
            test_mfcc = test_mfcc.T
        except Exception as e:
            raise RuntimeError(f"提取测试文件特征失败: {e}")

        # 计算对数似然分数
        speaker_gmm = self.speaker_models[speaker_id]
        log_likelihood = speaker_gmm.score(test_mfcc)

        return log_likelihood

    def get_likelihood_ratio_score(self, file_path: str, speaker_id: str) -> float:
        """
        计算似然比分数（说话人模型 vs UBM）

        Args:
            file_path: 测试音频文件路径
            speaker_id: 目标说话人ID

        Returns:
            似然比分数
        """
        if speaker_id not in self.speaker_models:
            raise ValueError(f"说话人 {speaker_id} 未注册")

        if self.ubm is None:
            raise ValueError("UBM未训练")

        # 提取测试文件的特征
        try:
            test_mfcc = extract_mfcc(file_path)
            test_mfcc = normalize_features(test_mfcc)
            test_mfcc = test_mfcc.T
        except Exception as e:
            raise RuntimeError(f"提取测试文件特征失败: {e}")

        # 计算说话人模型的对数似然
        speaker_gmm = self.speaker_models[speaker_id]
        speaker_ll = speaker_gmm.score(test_mfcc)

        # 计算UBM的对数似然
        ubm_ll = self.ubm.score(test_mfcc)

        # 返回似然比（对数域中的差值）
        return speaker_ll - ubm_ll

    def get_ubm_info(self) -> Dict:
        """
        获取UBM模型信息

        Returns:
            UBM信息字典
        """
        if self.ubm is None:
            return {"trained": False}

        return {
            "trained": True,
            "n_components": self.ubm.n_components,
            "covariance_type": self.ubm.covariance_type,
            "converged": self.ubm.converged_,
            "n_iter": self.ubm.n_iter_,
            "lower_bound": getattr(self.ubm, 'lower_bound_', None)
        }

    def _get_model_specific_data(self) -> Dict:
        """获取GMM模型特定的数据用于保存"""
        return {
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'max_iter': self.max_iter,
            'tol': self.tol,
            'relevance_factor': self.relevance_factor,
            'ubm': self.ubm
        }

    def _load_model_specific_data(self, save_data: Dict) -> None:
        """加载GMM模型特定的数据"""
        self.n_components = save_data.get('n_components', 256)
        self.covariance_type = save_data.get('covariance_type', 'diag')
        self.max_iter = save_data.get('max_iter', 100)
        self.tol = save_data.get('tol', 1e-3)
        self.relevance_factor = save_data.get('relevance_factor', 16.0)
        self.ubm = save_data.get('ubm')


if __name__ == "__main__":
    # 测试GMM模型
    from ..config import config

    # 创建GMM模型实例
    gmm_model = GMMModel(config)

    print("GMM模型创建成功")
    print(f"模型信息: {gmm_model}")
    print(f"高斯分量数: {gmm_model.n_components}")
    print(f"协方差类型: {gmm_model.covariance_type}")

    # 这里可以添加更多的测试代码
