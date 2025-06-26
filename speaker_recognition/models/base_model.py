"""
声纹识别模型基类

定义所有声纹识别模型的统一抽象接口，确保所有模型都遵循相同的生命周期。
包括训练、注册、识别、保存和加载等核心方法。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import pickle
import os

from ..config import config


class BaseModel(ABC):
    """声纹识别模型抽象基类"""
    
    def __init__(self, config_obj=None):
        """
        初始化模型
        
        Args:
            config_obj: 配置对象，如果为None则使用默认配置
        """
        self.config = config_obj if config_obj is not None else config
        self.is_trained = False
        self.speaker_models = {}  # 存储每个说话人的模型
        self.model_name = self.__class__.__name__.lower().replace('model', '')
        
    @abstractmethod
    def train(self, file_paths: Dict[str, List[str]]) -> None:
        """
        训练通用模型（如UBM或VQ码本）
        
        Args:
            file_paths: 训练文件路径字典 {speaker_id: [file_path1, file_path2, ...]}
        """
        pass
    
    @abstractmethod
    def enroll(self, speaker_id: str, file_paths: List[str]) -> None:
        """
        为单个说话人注册模型
        
        Args:
            speaker_id: 说话人ID
            file_paths: 该说话人的注册文件路径列表
        """
        pass
    
    @abstractmethod
    def identify(self, file_path: str) -> Tuple[str, float]:
        """
        识别单个测试文件的说话人
        
        Args:
            file_path: 测试音频文件路径
            
        Returns:
            (最可能的说话人ID, 置信度分数)
        """
        pass
    
    def verify(self, file_path: str, claimed_speaker_id: str) -> Tuple[bool, float]:
        """
        验证测试文件是否属于声称的说话人
        
        Args:
            file_path: 测试音频文件路径
            claimed_speaker_id: 声称的说话人ID
            
        Returns:
            (是否匹配, 匹配分数)
        """
        predicted_speaker, score = self.identify(file_path)
        is_match = (predicted_speaker == claimed_speaker_id)
        return is_match, score
    
    def get_speaker_score(self, file_path: str, speaker_id: str) -> float:
        """
        计算测试文件对特定说话人的匹配分数
        
        Args:
            file_path: 测试音频文件路径
            speaker_id: 目标说话人ID
            
        Returns:
            匹配分数
        """
        # 默认实现：通过identify方法获取分数
        predicted_speaker, score = self.identify(file_path)
        if predicted_speaker == speaker_id:
            return score
        else:
            # 如果不是最佳匹配，返回较低的分数
            return -score if score > 0 else score
    
    def get_enrolled_speakers(self) -> List[str]:
        """
        获取已注册的说话人ID列表
        
        Returns:
            说话人ID列表
        """
        return list(self.speaker_models.keys())
    
    def is_speaker_enrolled(self, speaker_id: str) -> bool:
        """
        检查说话人是否已注册
        
        Args:
            speaker_id: 说话人ID
            
        Returns:
            是否已注册
        """
        return speaker_id in self.speaker_models
    
    def remove_speaker(self, speaker_id: str) -> bool:
        """
        移除已注册的说话人
        
        Args:
            speaker_id: 说话人ID
            
        Returns:
            是否成功移除
        """
        if speaker_id in self.speaker_models:
            del self.speaker_models[speaker_id]
            return True
        return False
    
    def save(self, file_path: str = None) -> None:
        """
        保存训练好的模型
        
        Args:
            file_path: 保存路径，如果为None则使用默认路径
        """
        if file_path is None:
            file_path = self.config.get_model_path(self.model_name, "model.pkl")
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 准备保存的数据
        save_data = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'speaker_models': self.speaker_models,
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else None
        }
        
        # 添加模型特定的数据
        model_specific_data = self._get_model_specific_data()
        save_data.update(model_specific_data)
        
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"模型已保存到: {file_path}")
        except Exception as e:
            raise RuntimeError(f"保存模型失败: {e}")
    
    def load(self, file_path: str = None) -> None:
        """
        加载模型
        
        Args:
            file_path: 模型文件路径，如果为None则使用默认路径
        """
        if file_path is None:
            file_path = self.config.get_model_path(self.model_name, "model.pkl")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"模型文件不存在: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                save_data = pickle.load(f)
            
            # 恢复基本属性
            self.is_trained = save_data.get('is_trained', False)
            self.speaker_models = save_data.get('speaker_models', {})
            
            # 恢复模型特定的数据
            self._load_model_specific_data(save_data)
            
            print(f"模型已从 {file_path} 加载")
            print(f"已注册说话人数量: {len(self.speaker_models)}")
            
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def _get_model_specific_data(self) -> Dict[str, Any]:
        """
        获取模型特定的数据用于保存
        子类可以重写此方法来保存额外的数据
        
        Returns:
            模型特定数据字典
        """
        return {}
    
    def _load_model_specific_data(self, save_data: Dict[str, Any]) -> None:
        """
        加载模型特定的数据
        子类可以重写此方法来加载额外的数据
        
        Args:
            save_data: 保存的数据字典
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_name': self.model_name,
            'model_class': self.__class__.__name__,
            'is_trained': self.is_trained,
            'num_enrolled_speakers': len(self.speaker_models),
            'enrolled_speakers': list(self.speaker_models.keys())
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        info = self.get_model_info()
        return (f"{info['model_class']}("
                f"trained={info['is_trained']}, "
                f"speakers={info['num_enrolled_speakers']})")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()
