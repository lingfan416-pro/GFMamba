# 🎭 GFMamba Deployment Guides

## 📋 Project Overview

GFMamba is a multimodal sentiment analysis model based on the Mamba architecture that integrates text, audio, and visual features for sentiment prediction. This project supports datasets such as CMU-MOSI and can be used for regression-based sentiment analysis tasks.

## 🧠 Model architecture

**GFMamba** The model contains the following core components:
- **ModalityProjector**: Projects features from different modalities to a unified dimension
- **ContextExtractor**: Extracts contextual information from each modality
- **TGMamba**: A cross-modal fusion module based on Mamba
- **IntraModalEnhance**: Enhances internal features of each modality
- **Graph Fusion**: Final multimodal feature fusion

## 🛠️ Installation Requirements

### System Requirements
- **Python**: >= 3.9
- **PyTorch**: >= 2.1.0
- **CUDA**: 11.8+ (Optional，using GPU to speed up)

### Installation Dependency
```bash
# Clone project
git clone <repository-url>
cd GFMamba-main

# Install dependency
pip install -r requirements.txt
```

## 📁 project structure

```
GFMamba-main/
├── configs/
│   └── mosi_train.yaml          # Model config file
├── core/                        # Model main function
│   ├── dataset.py              # dataset process
│   ├── losses.py               # loss functions
│   ├── metric.py               # evalutation metrics
│   ├── optimizer.py            # optimizer
│   ├── scheduler.py            # learning scheduler
│   └── utils.py                # utils functions
├── models/                      # model definitions
│   ├── GFMamba.py              # Main entry
│   ├── enhance.py              # enhance module
│   ├── gl_feature.py           # global feature
│   ├── graph_fusion.py         # graph fusion
│   ├── Intramodel.py           # Intra model enhancement
│   └── mamba/                  # Mamba related models
├── ckpt/                       # the weight of models
│   └── mosi/
│       └── best_valid_model_seed_42.pth
├── inference.py                # inference scripts
├── data_preprocessing.py       # data preprocessing 
├── app.py                      # Web API
├── train.py                    # train script
└── requirements.txt            # depenency list
```

## 🚀 Quick start

### 1. command line inference

```bash
# basic inference
python inference.py

# use self defined files 
python inference.py --text "Your text here" --audio "audio.wav" --video "video.mp4"
```

### 2. Web API Service

```bash
# Startup Web Service
python app.py

# Visit Web UI
# http://localhost:5000
```

### 3. Data Preprocessing

```bash
# running data pre-processing test
python data_preprocessing.py
```

## 📊 Inputs

### Text Inputs
- **format**: String
- **processing**: BERT encoding (768)
- **length of sequence**: Maximum 512 tokens

### Audio Input
- **Format**: WAV, MP3, M4A
- **Sampling rate**: 16kHz
- **Features**: MFCC (20个系数)
- **Length of Sequence**: 50 frames

### Video Input
- **Format**: MP4, AVI, MOV
- **Resolution**: Automatically adjust to 64x64
- **Features**: 5 visual statistical features
- **Length of Sequence**: 50 frames

## 🔧 Configration

### Model Configuration (configs/mosi_train.yaml)

```yaml
model:
  input_dim: [768, 20, 5]     # [Script, audio, video]Dimension
  dim: 64                     # Model internal feature dimensions
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

## 📈 Training model

```bash
# Train with default configuration
python train.py

# Use custom configuration
python train.py --config_file your_config.yaml --seed 42
```

## 🌐 API Usage

### Web UI
visit `http://localhost:5000` Use a graphical interface to perform sentiment analysis.

### REST API

#### sentiment analysis
```bash
curl -X POST http://localhost:5000/analyze \
  -F "text=This is a great movie!" \
  -F "audio=@audio.wav" \
  -F "video=@video.mp4"
```

#### health check
```bash
curl http://localhost:5000/health
```

#### API info
```bash
curl http://localhost:5000/api/info
```

### response format
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

## 🎯 Usage example

### Python code example

```python
from inference import GFMambaInference

# Initialize the reasoner
inference = GFMambaInference(
    config_path='configs/mosi_train.yaml',
    model_path='ckpt/mosi/best_valid_model_seed_42.pth'
)

# sentiment analysis
result = inference.predict_sentiment(
    text="This is a wonderful day!",
    audio_path="audio.wav",
    video_path="video.mp4"
)

print(f"Emotional score: {result['sentiment_score']}")
print(f"Emotional label: {result['sentiment_label']}")
```

### Data Preprocessing Example

```python
from data_preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Creat samplong data
sample = preprocessor.create_sample_data(
    text="Sample text",
    audio_path="audio.wav",
    video_path="video.mp4",
    label=0.5
)
```

## 🔍 Debugging

### FAQ

1. **Model loading failed**
   ```
   Error: Unable to load model weights
  Solution: Check if there is a pre-trained model file in the ckpt/mosi/ directory
   ```

2. **CUDA out of memory**
   ```
 Error: CUDA out of memory
 Solution: Reduce batch_size or use CPU mode
   ```

3. **Dependency package conflict**
   ```
  Error: ImportError
  Solution: Recreate the virtual environment and install dependencies
   ```

### debug mode
```bash
# Enable Flask debug mode
export FLASK_DEBUG=1
python app.py
```

## 📊 Performance optimization

### GPU Speed up
```python
# Check GPU availability
import torch
print(f"CUDA availability: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
```

### Batch processing
For large amounts of data, batch processing is recommended:
```python
# Batch inference
def batch_inference(inference, data_list):
    results = []
    for data in data_list:
        result = inference.predict_sentiment(**data)
        results.append(result)
    return results
```

## 🔒 security considerations

- File upload restrictions: Limit file types and sizes
- Temporary file cleaning: Automatically delete uploaded temporary files
- Data privacy: Do not store user data
- Network security: HTTPS is recommended for production environments

## 📞 tech support

The origin author:
If you have any questions, please contact:
- email: zzhe232@aucklanduni.ac.nz
- project repo: [GitHub Repository]

## 📄 License

This project follows the corresponding open source license, please see the LICENSE file for details.


---
result:

Table A1. Per-scenario configuration and outcomes.

| Scen. | Lighting | Background | Noise | Interviewer | Anxiety | Text | Audio | Video | Composite | Label | File |
|------:|----------|------------|------:|------------:|--------:|-----:|------:|------:|----------:|:-----:|:----:|
| 1 | nature | clean | off | live | 6.33 | -0.0010 | -0.6852 | 0.9990 | 0.1043 | neutral | 1.mov |
| 2 | soft | clean | off | live | 4.00 | -0.0009 | -0.7111 | 0.9986 | 0.0955 | neutral | 2.mov |
| 3 | soft | mess  | off | live | 4.33 | -0.0010 | -0.7985 | 0.9947 | 0.0651 | neutral | 3.mov |
| 4 | soft | mess  | on  | live | 7.00 | -0.0011 | -0.6079 | 0.9906 | 0.1272 | neutral | 4.mov |
| 5 | nature | mess | on  | live | 4.83 | -0.0010 | -0.8522 | 0.9913 | 0.0460 | neutral | 5.mov |
| 6 | nature | mess | off | live | 2.50 | -0.0011 | -0.8392 | 0.9878 | 0.0492 | neutral | 6.mov |
| 7 | nature | clean | on  | live | 5.17 | -0.0011 | -0.8095 | 0.9984 | 0.0626 | neutral | 7.mov |
| 8 | off | clean  | on  | live | 6.67 | -0.0011 | -0.7213 | 0.9979 | 0.0918 | neutral | 8.mov |
| 9 | off | clean  | off | avatar | 3.00 | -0.0012 | -0.6535 | 0.9979 | 0.1144 | neutral | 9.mov |
