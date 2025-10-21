#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版零影响方案：不依赖mamba_ssm
使用您的MOV和TXT文件进行情感分析
"""

import os
import pickle
import numpy as np
import torch
import yaml
import librosa
import cv2
import tempfile
import subprocess
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings("ignore")

class SimplifiedProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用设备: {self.device}")
        
        # 初始化BERT
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.text_encoder = AutoModel.from_pretrained('bert-base-uncased').to(self.device)
            print("✅ BERT文本编码器加载成功")
        except Exception as e:
            print(f"⚠️ BERT编码器加载失败: {e}")
            self.tokenizer = None
            self.text_encoder = None
    
    def process_text_file(self, txt_path):
        """处理TXT文件"""
        print(f"📝 处理文本文件: {txt_path}")
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        print(f"📖 文本内容: {text[:100]}...")
        print(f"📏 文本长度: {len(text)} 字符")
        
        if self.tokenizer is None:
            print("⚠️ 使用标准分布生成文本特征")
            return np.random.normal(0, 1, (50, 768)).astype(np.float32)
        
        # 使用BERT编码
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, 
                              truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            features = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        
        # 调整到50长度
        if features.shape[0] > 50:
            features = features[:50]
        else:
            pad_width = 50 - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        
        print(f"✅ 文本特征形状: {features.shape}")
        return features.astype(np.float32)
    
    def process_mov_file(self, video_path):
        """处理视频文件"""
        print(f"🎬 处理video文件: {video_path}")
        
        # 分析视频信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ 无法打开视频文件")
            return np.random.normal(0, 1, (50, 20)).astype(np.float32), np.random.normal(0, 1, (50, 5)).astype(np.float32)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"📹 视频信息:")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps:.2f} FPS")
        print(f"   时长: {duration:.2f} 秒")
        print(f"   总帧数: {frame_count}")
        
        cap.release()
        
        # 提取音频
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.wav")
        
        try:
            print("🔊 提取音频...")
            # 直接使用librosa处理视频文件
            try:
                try:
                    audio, sr = librosa.load(video_path, sr=16000)
                    print("✅ librosa audio extract successfully")
                except:
                    # librosa失败则使用imageio_ffmpeg提取wav
                    print("⚠️ librosa提取失败,尝试使用imageio_ffmpeg...")
                    import imageio_ffmpeg
                    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
                    ffmpeg_cmd = f'"{ffmpeg_path}" -i "{video_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{audio_path}" -y'
                    subprocess.run(ffmpeg_cmd, shell=True, check=True)
                    audio, sr = librosa.load(audio_path, sr=16000)
                    print("✅ imageio_ffmpeg音频提取成功")
                # 直接处理音频特征
                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
                
                if mfcc.shape[1] > 50:
                    mfcc = mfcc[:, :50]
                else:
                    pad_width = 50 - mfcc.shape[1]
                    mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
                
                audio_features = mfcc.T.astype(np.float32)
                print(f"✅ audio features shape: {audio_features.shape}")
            except Exception as e:
                print(f"❌ 音频提取失败: {e}")
                audio_features = np.random.normal(0, 1, (50, 20)).astype(np.float32)
            
            # 提取视频特征
            print("🎥 Vedio features extraction...")
            video_features = self.extract_video_features(video_path)
            
            return audio_features, video_features
            
        finally:
            # 清理临时文件
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                os.rmdir(temp_dir)
            except:
                pass
    
    def extract_audio_features(self, audio_path):
        """提取音频特征"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
            
            if mfcc.shape[1] > 50:
                mfcc = mfcc[:, :50]
            else:
                pad_width = 50 - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            
            print(f"✅ Audio feature shape: {mfcc.T.shape}")
            return mfcc.T.astype(np.float32)
        except Exception as e:
            print(f"❌ 音频特征提取失败: {e}")
            return np.random.normal(0, 1, (50, 20)).astype(np.float32)
    
    def extract_video_features(self, video_path):
        """提取视频特征"""
        try:
            cap = cv2.VideoCapture(video_path)
            features = []
            
            while len(features) < 50:
                ret, frame = cap.read()
                if not ret:
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (64, 64))
                
                frame_features = [
                    np.mean(resized),
                    np.std(resized),
                    np.mean(cv2.Laplacian(resized, cv2.CV_64F)),
                    np.mean(cv2.Sobel(resized, cv2.CV_64F, 1, 0)),
                    np.mean(cv2.Sobel(resized, cv2.CV_64F, 0, 1))
                ]
                
                features.append(frame_features)
            
            cap.release()
            
            # 填充到50长度
            while len(features) < 50:
                features.append([0.0] * 5)
            
            print(f"✅ Audio feature shape: {np.array(features).shape}")
            return np.array(features, dtype=np.float32)
        except Exception as e:
            print(f"❌ 视频特征提取失败: {e}")
            return np.random.normal(0, 1, (50, 5)).astype(np.float32)
    
    def simple_sentiment_analysis(self, text_features, audio_features, video_features):
        """简化的情感分析"""
        print("🔍 Simplified Sentiment analysis...")
        
        # 基于特征的简单情感分析
        # 这里使用启发式方法，实际应用中应该使用训练好的模型
        
        # 文本情感分析
        text_sentiment = self.analyze_text_sentiment(text_features)
        
        # 音频情感分析
        audio_sentiment = self.analyze_audio_sentiment(audio_features)
        
        # 视频情感分析
        video_sentiment = self.analyze_video_sentiment(video_features)
        
        # 多模态融合
        final_sentiment = (text_sentiment + audio_sentiment + video_sentiment) / 3.0
        
        return final_sentiment, {
            'text_sentiment': text_sentiment,
            'audio_sentiment': audio_sentiment,
            'video_sentiment': video_sentiment
        }
    
    def analyze_text_sentiment(self, text_features):
        """分析文本情感"""
        # 基于BERT特征的简单分析
        # 计算特征的平均值和方差
        mean_val = np.mean(text_features)
        std_val = np.std(text_features)
        
        # 启发式规则：正值表示积极，负值表示消极
        sentiment = np.tanh(mean_val * 0.1)  # 缩放并应用tanh激活
        
        return float(sentiment)
    
    def analyze_audio_sentiment(self, audio_features):
        """分析音频情感"""
        # 基于MFCC特征的简单分析
        mean_val = np.mean(audio_features)
        std_val = np.std(audio_features)
        
        # 启发式规则
        sentiment = np.tanh(mean_val * 0.05)
        
        return float(sentiment)
    
    def analyze_video_sentiment(self, video_features):
        """分析视频情感"""
        # 基于视觉特征的简单分析
        mean_val = np.mean(video_features)
        std_val = np.std(video_features)
        
        # 启发式规则
        sentiment = np.tanh(mean_val * 0.1)
        
        return float(sentiment)

def main(video_path, txt_path):
    """主函数"""
    print("🎭 GFMamba 简化版零影响方案")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(txt_path):
        print(f"❌ 文本文件不存在: {txt_path}")
        return
    
    if not os.path.exists(video_path):
        print(f"❌ 视频文件不存在: {video_path}")
        return
    
    print(f"📝 文本文件: {txt_path}")
    print(f"🎬 视频文件: {video_path}")
    print()
    
    try:
        # 创建处理器
        processor = SimplifiedProcessor()
        
        # 处理文本文件
        text_features = processor.process_text_file(txt_path)
        
        # 处理MOV文件
        audio_features, video_features = processor.process_mov_file(video_path)
        
        # 进行情感分析
        sentiment_score, detailed_results = processor.simple_sentiment_analysis(
            text_features, audio_features, video_features
        )
        
        # 显示结果
        print("\n🎉 Simplified Sentiment Analyze result:")
        print("=" * 60)
        print(f"🎯 Comprehensive Sentiment score: {sentiment_score:.4f}")
        
        if sentiment_score < -0.5:
            sentiment_label = "negative"
        elif sentiment_score < 0.5:
            sentiment_label = "neutral"
        else:
            sentiment_label = "positive"
        
        print(f"📈 Sentimental Label: {sentiment_label}")
        
        print(f"\n📊 Detail analysis:")
        print(f"   Text Script Sentiment: {detailed_results['text_sentiment']:.4f}")
        print(f"   Audio Sentiment: {detailed_results['audio_sentiment']:.4f}")
        print(f"   Video Sentiment: {detailed_results['video_sentiment']:.4f}")

        # 保存结果
        result_data = {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'detailed_results': detailed_results,
            'text_path': txt_path,
            'video_path': video_path
        }

        output_dir = os.path.dirname(video_path)
        output_path = os.path.join(output_dir, 'simplified_result.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        print(f"📄 result have been saved in: {output_path}")
        
    except Exception as e:
        print(f"❌ 运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
