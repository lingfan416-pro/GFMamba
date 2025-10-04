#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GFMamba 推理脚本
基于README中的模型架构进行多模态情感分析推理
"""

import os
import sys
import torch
import yaml
import numpy as np
import librosa
import cv2
from PIL import Image
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.GFMamba import GFMamba

class GFMambaInference:
    def __init__(self, config_path='configs/mosi_train.yaml', model_path=None):
        """
        初始化GFMamba推理器
        
        Args:
            config_path: 模型配置文件路径
            model_path: 预训练模型权重路径
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        # 初始化模型
        self.model = GFMamba(self.config).to(self.device)
        
        # 加载预训练权重
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"成功加载模型权重: {model_path}")
        else:
            print("警告: 未找到预训练模型权重，使用随机初始化的模型")
        
        self.model.eval()
        
        # 初始化文本编码器
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.text_encoder = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
            print("成功加载BERT文本编码器")
        except Exception as e:
            print(f"警告: 无法加载BERT编码器: {e}")
            self.tokenizer = None
            self.text_encoder = None
    
    def preprocess_text(self, text):
        """
        预处理文本输入
        
        Args:
            text: 输入文本字符串
            
        Returns:
            torch.Tensor: 文本特征 [1, seq_len, 768]
        """
        if self.tokenizer is None or self.text_encoder is None:
            # 如果没有BERT，使用简单的embedding
            # 这里简化为随机特征，实际应用中需要更好的文本编码
            seq_len = min(len(text.split()), 50)
            text_features = torch.randn(1, seq_len, 768).to(self.device)
            return text_features
        
        # 使用BERT编码
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            text_features = outputs.last_hidden_state  # [1, seq_len, 768]
        
        return text_features
    
    def preprocess_audio(self, audio_path, target_length=50):
        """
        预处理音频输入
        
        Args:
            audio_path: 音频文件路径
            target_length: 目标序列长度
            
        Returns:
            torch.Tensor: 音频特征 [1, target_length, 20]
        """
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            
            # 调整到目标长度
            if mfcc.shape[1] > target_length:
                # 截断
                mfcc = mfcc[:, :target_length]
            else:
                # 填充
                pad_width = target_length - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            
            # 转换为tensor
            audio_features = torch.FloatTensor(mfcc.T).unsqueeze(0).to(self.device)  # [1, target_length, 20]
            
            return audio_features
        except Exception as e:
            print(f"音频处理失败: {e}")
            # 返回零填充
            return torch.zeros(1, target_length, 20).to(self.device)
    
    def preprocess_video(self, video_path, target_length=50):
        """
        预处理视频输入
        
        Args:
            video_path: 视频文件路径
            target_length: 目标序列长度
            
        Returns:
            torch.Tensor: 视频特征 [1, target_length, 5]
        """
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            # 提取帧
            while len(frames) < target_length:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 转换为灰度图并调整大小
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                
                # 提取简单的视觉特征
                features = [
                    np.mean(resized),  # 平均亮度
                    np.std(resized),   # 亮度标准差
                    np.mean(cv2.Laplacian(resized, cv2.CV_64F)),  # 边缘强度
                    np.mean(cv2.Sobel(resized, cv2.CV_64F, 1, 0)),  # 水平梯度
                    np.mean(cv2.Sobel(resized, cv2.CV_64F, 0, 1))   # 垂直梯度
                ]
                
                frames.append(features)
            
            cap.release()
            
            # 填充到目标长度
            while len(frames) < target_length:
                frames.append([0.0] * 5)
            
            # 转换为tensor
            video_features = torch.FloatTensor(frames).unsqueeze(0).to(self.device)  # [1, target_length, 5]
            
            return video_features
        except Exception as e:
            print(f"视频处理失败: {e}")
            # 返回零填充
            return torch.zeros(1, target_length, 5).to(self.device)
    
    def predict_sentiment(self, text=None, audio_path=None, video_path=None):
        """
        预测情感分数
        
        Args:
            text: 文本内容
            audio_path: 音频文件路径
            video_path: 视频文件路径
            
        Returns:
            dict: 包含预测结果的字典
        """
        features = []
        
        # 处理文本模态
        if text:
            text_features = self.preprocess_text(text)
            features.append(text_features)
        else:
            # 如果没有文本，创建零填充
            text_features = torch.zeros(1, 50, 768).to(self.device)
            features.append(text_features)
        
        # 处理音频模态
        if audio_path and os.path.exists(audio_path):
            audio_features = self.preprocess_audio(audio_path)
            features.append(audio_features)
        else:
            # 如果没有音频，创建零填充
            audio_features = torch.zeros(1, 50, 20).to(self.device)
            features.append(audio_features)
        
        # 处理视频模态
        if video_path and os.path.exists(video_path):
            video_features = self.preprocess_video(video_path)
            features.append(video_features)
        else:
            # 如果没有视频，创建零填充
            video_features = torch.zeros(1, 50, 5).to(self.device)
            features.append(video_features)
        
        # 模型推理
        with torch.no_grad():
            output = self.model(*features)
            sentiment_score = output['sentiment_preds'].item()
        
        # 情感解释
        if sentiment_score < -1.0:
            sentiment_label = "非常负面"
        elif sentiment_score < -0.3:
            sentiment_label = "负面"
        elif sentiment_score < 0.3:
            sentiment_label = "中性"
        elif sentiment_score < 1.0:
            sentiment_label = "正面"
        else:
            sentiment_label = "非常正面"
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'modalities_used': {
                'text': text is not None,
                'audio': audio_path is not None and os.path.exists(audio_path),
                'video': video_path is not None and os.path.exists(video_path)
            },
            'attention_weights': output.get('att_weights', None)
        }

def main():
    """主函数，用于测试"""
    print("🎭 GFMamba 多模态情感分析推理")
    print("=" * 50)
    
    # 初始化推理器
    try:
        inference = GFMambaInference(
            config_path='configs/mosi_train.yaml',
            model_path='ckpt/mosi/best_valid_model_seed_42.pth'
        )
        print("✅ 推理器初始化成功")
    except Exception as e:
        print(f"❌ 推理器初始化失败: {e}")
        return
    
    # 示例测试
    print("\n📝 开始测试...")
    
    # 测试文本输入
    test_text = "This is a great movie, I really enjoyed it!"
    result = inference.predict_sentiment(text=test_text)
    
    print(f"输入文本: {test_text}")
    print(f"情感分数: {result['sentiment_score']:.3f}")
    print(f"情感标签: {result['sentiment_label']}")
    print(f"使用的模态: {result['modalities_used']}")
    
    # 如果有用户的文件，也可以测试
    user_video = "/Users/liyunfeng/Desktop/IMG_0721.MOV"
    user_text = "/Users/liyunfeng/Desktop/test.txt"
    
    if os.path.exists(user_video) and os.path.exists(user_text):
        print(f"\n🎬 测试用户文件...")
        
        # 读取文本内容
        try:
            with open(user_text, 'r', encoding='utf-8') as f:
                user_text_content = f.read().strip()
            
            result = inference.predict_sentiment(
                text=user_text_content,
                video_path=user_video
            )
            
            print(f"用户文本: {user_text_content[:100]}...")
            print(f"情感分数: {result['sentiment_score']:.3f}")
            print(f"情感标签: {result['sentiment_label']}")
            print(f"使用的模态: {result['modalities_used']}")
            
        except Exception as e:
            print(f"处理用户文件时出错: {e}")

if __name__ == "__main__":
    main()
