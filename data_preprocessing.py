#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本
用于处理多模态数据，支持CMU-MOSI等数据集格式
"""

import os
import pickle
import numpy as np
import librosa
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import warnings
warnings.filterwarnings("ignore")

class MultimodalDataset(Dataset):
    """多模态数据集类"""
    
    def __init__(self, data_path, config):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            config: 配置字典
        """
        self.config = config
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path):
        """加载数据"""
        if data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.data[idx]
        
        # 提取各模态数据
        text = sample.get('text', np.zeros((50, 768)))
        audio = sample.get('audio', np.zeros((50, 20)))
        video = sample.get('vision', np.zeros((50, 5)))
        labels = sample.get('labels', {'M': 0.0})
        
        return {
            'text': torch.FloatTensor(text),
            'audio': torch.FloatTensor(audio),
            'vision': torch.FloatTensor(video),
            'labels': {'M': torch.FloatTensor([labels['M']])}
        }

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, config_path='configs/mosi_train.yaml'):
        """初始化预处理器"""
        with open(config_path, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
    
    def preprocess_audio(self, audio_path, target_length=50):
        """
        预处理音频数据
        
        Args:
            audio_path: 音频文件路径
            target_length: 目标序列长度
            
        Returns:
            np.ndarray: 预处理后的音频特征
        """
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=sr, 
                n_mfcc=20,
                n_fft=2048,
                hop_length=512
            )
            
            # 调整到目标长度
            if mfcc.shape[1] > target_length:
                mfcc = mfcc[:, :target_length]
            else:
                pad_width = target_length - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            
            return mfcc.T  # [target_length, 20]
            
        except Exception as e:
            print(f"音频预处理失败: {e}")
            return np.zeros((target_length, 20))
    
    def preprocess_video(self, video_path, target_length=50):
        """
        预处理视频数据
        
        Args:
            video_path: 视频文件路径
            target_length: 目标序列长度
            
        Returns:
            np.ndarray: 预处理后的视频特征
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
                
                # 提取视觉特征
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
            
            return np.array(frames)  # [target_length, 5]
            
        except Exception as e:
            print(f"视频预处理失败: {e}")
            return np.zeros((target_length, 5))
    
    def preprocess_text(self, text, max_length=50):
        """
        预处理文本数据
        
        Args:
            text: 文本字符串
            max_length: 最大序列长度
            
        Returns:
            np.ndarray: 预处理后的文本特征
        """
        # 这里简化处理，实际应用中需要使用BERT等预训练模型
        # 将文本转换为固定长度的特征向量
        words = text.split()
        features = []
        
        for word in words[:max_length]:
            # 简单的词嵌入（实际应用中应使用预训练模型）
            word_features = np.random.randn(768)  # 768维特征
            features.append(word_features)
        
        # 填充到目标长度
        while len(features) < max_length:
            features.append(np.zeros(768))
        
        return np.array(features)  # [max_length, 768]
    
    def create_sample_data(self, text, audio_path=None, video_path=None, label=0.0):
        """
        创建样本数据
        
        Args:
            text: 文本内容
            audio_path: 音频文件路径
            video_path: 视频文件路径
            label: 标签
            
        Returns:
            dict: 样本数据字典
        """
        sample = {
            'text': self.preprocess_text(text),
            'labels': {'M': label}
        }
        
        if audio_path and os.path.exists(audio_path):
            sample['audio'] = self.preprocess_audio(audio_path)
        else:
            sample['audio'] = np.zeros((50, 20))
        
        if video_path and os.path.exists(video_path):
            sample['vision'] = self.preprocess_video(video_path)
        else:
            sample['vision'] = np.zeros((50, 5))
        
        return sample
    
    def save_processed_data(self, data, output_path):
        """
        保存预处理后的数据
        
        Args:
            data: 数据列表
            output_path: 输出路径
        """
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"数据已保存到: {output_path}")

def create_data_loader(data_path, config, batch_size=16, shuffle=True):
    """
    创建数据加载器
    
    Args:
        data_path: 数据文件路径
        config: 配置字典
        batch_size: 批次大小
        shuffle: 是否打乱数据
        
    Returns:
        DataLoader: 数据加载器
    """
    dataset = MultimodalDataset(data_path, config)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    return dataloader

def main():
    """主函数，用于测试数据预处理"""
    print("🔧 数据预处理测试")
    print("=" * 50)
    
    # 初始化预处理器
    preprocessor = DataPreprocessor()
    
    # 测试文本预处理
    test_text = "This is a test sentence for preprocessing."
    text_features = preprocessor.preprocess_text(test_text)
    print(f"文本特征形状: {text_features.shape}")
    
    # 测试音频预处理（如果有音频文件）
    test_audio = "/Users/liyunfeng/Desktop/IMG_0721.MOV"  # 从视频中提取音频
    if os.path.exists(test_audio):
        try:
            audio_features = preprocessor.preprocess_audio(test_audio)
            print(f"音频特征形状: {audio_features.shape}")
        except Exception as e:
            print(f"音频预处理失败: {e}")
    
    # 测试视频预处理
    test_video = "/Users/liyunfeng/Desktop/IMG_0721.MOV"
    if os.path.exists(test_video):
        try:
            video_features = preprocessor.preprocess_video(test_video)
            print(f"视频特征形状: {video_features.shape}")
        except Exception as e:
            print(f"视频预处理失败: {e}")
    
    # 创建样本数据
    sample = preprocessor.create_sample_data(
        text=test_text,
        audio_path=test_audio if os.path.exists(test_audio) else None,
        video_path=test_video if os.path.exists(test_video) else None,
        label=0.5
    )
    
    print(f"样本数据键: {sample.keys()}")
    print(f"文本特征形状: {sample['text'].shape}")
    print(f"音频特征形状: {sample['audio'].shape}")
    print(f"视频特征形状: {sample['vision'].shape}")
    print(f"标签: {sample['labels']['M']}")

if __name__ == "__main__":
    main()
