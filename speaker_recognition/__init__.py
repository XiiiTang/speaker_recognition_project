"""
声纹识别系统核心包

这个包包含了声纹识别系统的所有核心模块：
- config: 配置管理
- data_loader: 数据集加载和划分
- feature_extractor: 音频特征提取
- evaluate: 性能评估
- models: 声纹识别算法模型
"""

__version__ = "1.0.0"
__author__ = "Speaker Recognition Project"

# 导入主要模块
from . import config
from . import data_loader
from . import feature_extractor
from . import evaluate
from . import models

__all__ = [
    'config',
    'data_loader', 
    'feature_extractor',
    'evaluate',
    'models'
]
