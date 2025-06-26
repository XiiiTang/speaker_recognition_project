# 声纹识别系统

基于Python的模块化声纹识别系统，实现了DTW、VQ、GMM三种经典算法，支持VoxCeleb2数据集。

## 项目特点

- **模块化设计**: 清晰的代码结构，易于扩展和维护
- **多算法支持**: 实现DTW、VQ、GMM三种声纹识别算法
- **完整流程**: 包含数据加载、特征提取、模型训练、说话人注册、识别测试和性能评估
- **灵活配置**: 集中的配置管理，便于参数调优
- **详细评估**: 支持准确率、EER、DET曲线等多种评估指标

## 项目结构

```
speaker_recognition_project/
├── speaker_recognition/          # 核心源代码包
│   ├── __init__.py
│   ├── config.py                 # 配置管理
│   ├── data_loader.py            # 数据集加载和划分
│   ├── feature_extractor.py      # 音频特征提取
│   ├── evaluate.py               # 性能评估
│   └── models/                   # 算法模型
│       ├── __init__.py
│       ├── base_model.py         # 模型基类
│       ├── dtw_model.py          # DTW算法
│       ├── vq_model.py           # VQ算法
│       └── gmm_model.py          # GMM算法
├── data/                         # 数据集目录
├── saved_models/                 # 模型保存目录
├── main.py                       # 主程序入口
├── test_system.py                # 系统测试脚本
├── requirements.txt              # 依赖库
└── README.md                     # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖库：
- numpy: 数值计算
- scipy: 科学计算
- scikit-learn: 机器学习
- librosa: 音频处理
- pydub: 音频格式转换
- tqdm: 进度条显示
- matplotlib: 图表绘制

## 配置设置

在使用前，请检查并修改 `speaker_recognition/config.py` 中的配置：

```python
# 重要配置项
VOXCELEB_PATH = r"H:\算法分析与设计\VOX2"  # VoxCeleb2数据集路径
TRAIN_RATIO = 0.8                          # 训练集比例
ENROLL_FILES_PER_SPEAKER = 5               # 每个说话人的注册文件数
SAMPLE_RATE = 16000                        # 音频采样率
N_MFCC = 16                               # MFCC特征维度
```

## 使用方法

### 1. 系统测试

首先运行系统测试确保环境配置正确：

```bash
python test_system.py
```

### 2. 完整流程运行

运行完整的训练、注册和测试流程：

```bash
# 使用GMM算法（推荐）
python main.py --model gmm --mode full

# 使用DTW算法
python main.py --model dtw --mode full

# 使用VQ算法
python main.py --model vq --mode full
```

### 3. 分步骤运行

```bash
# 仅训练模型
python main.py --model gmm --mode train --save-model saved_models/gmm_model.pkl

# 仅测试模型
python main.py --model gmm --mode test --load-model saved_models/gmm_model.pkl
```

### 4. 自定义参数

```bash
# 指定数据路径和输出目录
python main.py --model gmm --data-path /path/to/voxceleb2 --output-dir results/gmm

# 显示详细日志
python main.py --model gmm --verbose
```

## 算法说明

### DTW (Dynamic Time Warping)
- **原理**: 通过动态时间规整计算测试语音与注册模板的最小累积距离
- **特点**: 无需训练阶段，直接使用注册语音作为模板
- **适用**: 小规模数据集，计算简单

### VQ (Vector Quantization)
- **原理**: 训练通用码本，为每个说话人生成特征直方图
- **特点**: 需要全局训练阶段，使用K-means聚类
- **适用**: 中等规模数据集，计算效率高

### GMM (Gaussian Mixture Model)
- **原理**: 训练通用背景模型(UBM)，使用MAP自适应生成说话人模型
- **特点**: 最先进的传统方法，性能最佳
- **适用**: 大规模数据集，识别精度高

## 评估指标

系统支持多种评估指标：

- **识别准确率**: 闭集识别的正确率
- **等错误率(EER)**: 误接受率等于误拒绝率时的错误率
- **DET曲线**: 检测错误权衡曲线
- **混淆矩阵**: 详细的分类结果矩阵

## 输出结果

运行后会在指定输出目录生成：

- `{model}_results.txt`: 评估结果文本
- `{model}_det_curve.png`: DET曲线图
- `{model}_score_distribution.png`: 分数分布图

## 数据集要求

支持VoxCeleb2数据集，目录结构应为：
```
VoxCeleb2/
└── dev/
    └── aac/
        ├── id00012/
        │   ├── 21Uxsk56VDQ/
        │   │   ├── 00001.m4a
        │   │   └── ...
        │   └── ...
        └── ...
```

## 性能优化建议

1. **数据预处理**: 确保音频质量，去除噪声
2. **特征优化**: 调整MFCC参数，如维度、窗口大小
3. **模型参数**: 根据数据集大小调整码本大小或GMM分量数
4. **硬件加速**: 使用多核CPU并行处理

## 常见问题

### Q: 提示"VoxCeleb2数据集路径不存在"
A: 请检查 `config.py` 中的 `VOXCELEB_PATH` 设置是否正确

### Q: 内存不足错误
A: 减少 `GMM_N_COMPONENTS` 或 `VQ_CODEBOOK_SIZE` 参数

### Q: 音频格式不支持
A: 系统支持m4a格式，会自动转换为wav处理

### Q: 识别准确率较低
A: 尝试增加注册文件数量或调整特征提取参数

## 扩展开发

系统采用模块化设计，便于扩展：

1. **添加新算法**: 继承 `BaseModel` 类实现新的识别算法
2. **自定义特征**: 修改 `feature_extractor.py` 添加新的特征提取方法
3. **评估指标**: 在 `evaluate.py` 中添加新的评估函数

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目仓库: [GitHub链接]
- 邮箱: [联系邮箱]
