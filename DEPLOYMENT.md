# 🎭 GFMamba 部署指南

## 📋 项目概述

GFMamba是一个基于Mamba架构的多模态情感分析模型，能够整合文本、音频和视觉特征进行情感预测。本项目支持CMU-MOSI等数据集，可用于回归式情感分析任务。

## 🧠 模型架构

**GFMamba** 模型包含以下核心组件：

- **模态投影器** (ModalityProjector): 将不同模态特征投影到统一维度
- **上下文提取器** (ContextExtractor): 提取各模态的上下文信息
- **TGMamba**: 基于Mamba的跨模态融合模块
- **模态内增强器** (IntraModalEnhancer): 增强各模态内部特征
- **图融合** (graph_fusion): 最终的多模态特征融合

## 🛠️ 安装要求

### 系统要求
- **Python**: >= 3.9
- **PyTorch**: >= 2.1.0
- **CUDA**: 11.8+ (可选，用于GPU加速)

### 依赖安装
```bash
# 克隆项目
git clone <repository-url>
cd GFMamba-main

# 安装依赖
pip install -r requirements.txt
```

## 📁 项目结构

```
GFMamba-main/
├── configs/
│   └── mosi_train.yaml          # 模型配置文件
├── core/                        # 核心功能模块
│   ├── dataset.py              # 数据集处理
│   ├── losses.py               # 损失函数
│   ├── metric.py               # 评估指标
│   ├── optimizer.py            # 优化器
│   ├── scheduler.py            # 学习率调度器
│   └── utils.py                # 工具函数
├── models/                      # 模型定义
│   ├── GFMamba.py              # 主模型
│   ├── enhance.py              # 增强模块
│   ├── gl_feature.py           # 全局特征
│   ├── graph_fusion.py         # 图融合
│   ├── Intramodel.py           # 模态内增强
│   └── mamba/                  # Mamba相关模块
├── ckpt/                       # 模型权重
│   └── mosi/
│       └── best_valid_model_seed_42.pth
├── inference.py                # 推理脚本
├── data_preprocessing.py       # 数据预处理
├── app.py                      # Web API
├── train.py                    # 训练脚本
└── requirements.txt            # 依赖列表
```

## 🚀 快速开始

### 1. 命令行推理

```bash
# 基本推理
python inference.py

# 使用自定义文件
python inference.py --text "Your text here" --audio "audio.wav" --video "video.mp4"
```

### 2. Web API服务

```bash
# 启动Web服务
python app.py

# 访问Web界面
# http://localhost:5000
```

### 3. 数据预处理

```bash
# 运行数据预处理测试
python data_preprocessing.py
```

## 📊 输入格式

### 文本输入
- **格式**: 字符串
- **处理**: BERT编码 (768维)
- **序列长度**: 最大512 tokens

### 音频输入
- **格式**: WAV, MP3, M4A
- **采样率**: 16kHz
- **特征**: MFCC (20个系数)
- **序列长度**: 50帧

### 视频输入
- **格式**: MP4, AVI, MOV
- **分辨率**: 自动调整到64x64
- **特征**: 5个视觉统计特征
- **序列长度**: 50帧

## 🔧 配置说明

### 模型配置 (configs/mosi_train.yaml)

```yaml
model:
  input_dim: [768, 20, 5]     # [文本, 音频, 视频]维度
  dim: 64                     # 模型内部特征维度
  ContextExtractor:
    conv_kernel_size: 5
    hidden_ratio: 2
    dropout: 0.5
  TGMamba:
    num_layers: 1
    dropout: 0.5
    causal: false
    mamba_config:
      d_state: 12
      expand: 2
      d_conv: 4
      bidirectional: true
  IntraModalEnhancer:
    dropout: 0.5
  graph_fusion:
    num_classes: 1
    hidden: 16
    dropout: 0.5
    fusion_hidden: 24
```

## 📈 训练模型

```bash
# 使用默认配置训练
python train.py

# 使用自定义配置
python train.py --config_file your_config.yaml --seed 42
```

## 🌐 API使用

### Web界面
访问 `http://localhost:5000` 使用图形界面进行情感分析。

### REST API

#### 情感分析
```bash
curl -X POST http://localhost:5000/analyze \
  -F "text=This is a great movie!" \
  -F "audio=@audio.wav" \
  -F "video=@video.mp4"
```

#### 健康检查
```bash
curl http://localhost:5000/health
```

#### API信息
```bash
curl http://localhost:5000/api/info
```

### 响应格式
```json
{
  "success": true,
  "data": {
    "sentiment_score": 0.5,
    "sentiment_label": "正面",
    "modalities_used": {
      "text": true,
      "audio": true,
      "video": false
    },
    "timestamp": "2024-01-01T12:00:00"
  }
}
```

## 🎯 使用示例

### Python代码示例

```python
from inference import GFMambaInference

# 初始化推理器
inference = GFMambaInference(
    config_path='configs/mosi_train.yaml',
    model_path='ckpt/mosi/best_valid_model_seed_42.pth'
)

# 情感分析
result = inference.predict_sentiment(
    text="This is a wonderful day!",
    audio_path="audio.wav",
    video_path="video.mp4"
)

print(f"情感分数: {result['sentiment_score']}")
print(f"情感标签: {result['sentiment_label']}")
```

### 数据预处理示例

```python
from data_preprocessing import DataPreprocessor

# 初始化预处理器
preprocessor = DataPreprocessor()

# 创建样本数据
sample = preprocessor.create_sample_data(
    text="Sample text",
    audio_path="audio.wav",
    video_path="video.mp4",
    label=0.5
)
```

## 🔍 故障排除

### 常见问题

1. **模型加载失败**
   ```
   错误: 无法加载模型权重
   解决: 检查ckpt/mosi/目录下是否有预训练模型文件
   ```

2. **CUDA内存不足**
   ```
   错误: CUDA out of memory
   解决: 减少batch_size或使用CPU模式
   ```

3. **依赖包冲突**
   ```
   错误: ImportError
   解决: 重新创建虚拟环境并安装依赖
   ```

### 调试模式
```bash
# 启用Flask调试模式
export FLASK_DEBUG=1
python app.py
```

## 📊 性能优化

### GPU加速
```python
# 检查GPU可用性
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
```

### 批处理
对于大量数据，建议使用批处理：
```python
# 批量推理
def batch_inference(inference, data_list):
    results = []
    for data in data_list:
        result = inference.predict_sentiment(**data)
        results.append(result)
    return results
```

## 🔒 安全考虑

- 文件上传限制：限制文件类型和大小
- 临时文件清理：自动删除上传的临时文件
- 数据隐私：不保存用户数据
- 网络安全：生产环境建议使用HTTPS

## 📞 技术支持

如有问题，请联系：
- 邮箱: zzhe232@aucklanduni.ac.nz
- 项目地址: [GitHub Repository]

## 📄 许可证

本项目遵循相应的开源许可证，请查看LICENSE文件了解详情。

---

**注意**: 本系统专为多模态情感分析设计，支持文本、音频、视频三种模态的输入和融合分析。
