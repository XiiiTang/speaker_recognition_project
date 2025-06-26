"""
声纹识别系统配置文件

集中管理所有可调参数和文件路径，便于修改和实验。
包含数据路径、模型保存路径、数据集划分参数、音频特征参数等。
"""

import os

class Config:
    """声纹识别系统配置类"""
    
    # ==================== 数据路径配置 ====================
    # VoxCeleb2数据集根目录路径
    VOXCELEB_PATH = r"H:\算法分析与设计\VOX2"
    
    # 项目根目录
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 模型保存路径
    MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "saved_models")
    
    # 数据目录路径
    DATA_PATH = os.path.join(PROJECT_ROOT, "data")
    
    # ==================== 数据集划分参数 ====================
    # 全局训练集比例（用于训练UBM或VQ码本）
    TRAIN_RATIO = 0.8
    
    # 每个说话人用于注册的文件数量
    ENROLL_FILES_PER_SPEAKER = 5
    
    # 随机种子，确保实验可重复
    RANDOM_SEED = 42
    
    # ==================== 音频特征参数 ====================
    # 音频采样率
    SAMPLE_RATE = 16000
    
    # MFCC特征维度
    N_MFCC = 16
    
    # FFT窗口大小
    N_FFT = 512
    
    # 帧移长度
    HOP_LENGTH = 256
    
    # 窗口长度
    WIN_LENGTH = 512
    
    # 预加重系数
    PREEMPHASIS = 0.97
    
    # ==================== VQ模型参数 ====================
    # VQ码本大小
    VQ_CODEBOOK_SIZE = 256
    
    # KMeans聚类最大迭代次数
    VQ_MAX_ITER = 300
    
    # KMeans聚类随机初始化次数
    VQ_N_INIT = 10
    
    # ==================== GMM模型参数 ====================
    # GMM高斯混合分量数
    GMM_N_COMPONENTS = 256
    
    # GMM协方差类型
    GMM_COVARIANCE_TYPE = 'diag'
    
    # GMM最大迭代次数
    GMM_MAX_ITER = 100
    
    # GMM收敛容忍度
    GMM_TOL = 1e-3
    
    # MAP自适应相关性因子
    MAP_RELEVANCE_FACTOR = 16.0
    
    # ==================== DTW模型参数 ====================
    # DTW距离度量方式
    DTW_DISTANCE_METRIC = 'euclidean'
    
    # DTW路径约束类型
    DTW_STEP_PATTERN = 'symmetric2'
    
    # ==================== 评估参数 ====================
    # EER计算时的阈值步长
    EER_THRESHOLD_STEP = 0.01
    
    # EER计算时的阈值范围
    EER_THRESHOLD_RANGE = (-10.0, 10.0)
    
    # ==================== 其他参数 ====================
    # 是否显示详细日志
    VERBOSE = True
    
    # 并行处理线程数
    N_JOBS = -1
    
    # 临时文件目录
    TEMP_DIR = os.path.join(PROJECT_ROOT, "temp")
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            cls.MODEL_SAVE_PATH,
            cls.TEMP_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name, file_name):
        """获取模型文件的完整路径"""
        return os.path.join(cls.MODEL_SAVE_PATH, f"{model_name}_{file_name}")
    
    @classmethod
    def validate_paths(cls):
        """验证关键路径是否存在"""
        if not os.path.exists(cls.VOXCELEB_PATH):
            print(f"警告: VoxCeleb2数据集路径不存在: {cls.VOXCELEB_PATH}")
            return False
        return True

# 创建全局配置实例
config = Config()

# 在导入时创建必要目录
config.create_directories()
