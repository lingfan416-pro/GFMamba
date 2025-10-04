#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GFMamba Web API
基于Flask的多模态情感分析API服务
"""

import os
import sys
import json
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
import warnings
warnings.filterwarnings("ignore")

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference import GFMambaInference

app = Flask(__name__)
CORS(app)

# 配置
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'm4a', 'mp4', 'avi', 'mov', 'txt'}

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 全局推理器实例
inference = None

def init_inference():
    """初始化推理器"""
    global inference
    try:
        inference = GFMambaInference(
            config_path='configs/mosi_train.yaml',
            model_path='ckpt/mosi/best_valid_model_seed_42.pth'
        )
        print("✅ 推理器初始化成功")
        return True
    except Exception as e:
        print(f"❌ 推理器初始化失败: {e}")
        return False

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_uploaded_file(file, prefix=''):
    """保存上传的文件"""
    if file and allowed_file(file.filename):
        filename = f"{prefix}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return filepath
    return None

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GFMamba 多模态情感分析</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        input[type="text"], textarea, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        textarea {
            height: 100px;
            resize: vertical;
        }
        .file-upload {
            border: 2px dashed #ddd;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            background-color: #fafafa;
        }
        .file-upload:hover {
            border-color: #007bff;
            background-color: #f0f8ff;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 5px;
            display: none;
        }
        .result.success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .result.error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .sentiment-score {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .sentiment-very-negative { color: #dc3545; }
        .sentiment-negative { color: #fd7e14; }
        .sentiment-neutral { color: #6c757d; }
        .sentiment-positive { color: #28a745; }
        .sentiment-very-positive { color: #20c997; }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 GFMamba 多模态情感分析</h1>
        <p style="text-align: center; color: #666; margin-bottom: 30px;">
            基于Mamba架构的多模态情感分析系统
        </p>
        
        <form id="analysisForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="text">文本内容:</label>
                <textarea id="text" name="text" placeholder="请输入要分析的文本内容..."></textarea>
            </div>
            
            <div class="form-group">
                <label>音频文件 (可选):</label>
                <div class="file-upload">
                    <input type="file" id="audio" name="audio" accept=".wav,.mp3,.m4a">
                    <p>支持格式: WAV, MP3, M4A</p>
                </div>
            </div>
            
            <div class="form-group">
                <label>视频文件 (可选):</label>
                <div class="file-upload">
                    <input type="file" id="video" name="video" accept=".mp4,.avi,.mov">
                    <p>支持格式: MP4, AVI, MOV</p>
                </div>
            </div>
            
            <button type="submit">开始分析</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>正在分析中，请稍候...</p>
        </div>
        
        <div class="result" id="result">
            <h3>分析结果</h3>
            <div id="resultContent"></div>
        </div>
    </div>

    <script>
        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const text = document.getElementById('text').value;
            const audioFile = document.getElementById('audio').files[0];
            const videoFile = document.getElementById('video').files[0];
            
            if (!text.trim()) {
                alert('请输入文本内容');
                return;
            }
            
            formData.append('text', text);
            
            if (audioFile) {
                formData.append('audio', audioFile);
            }
            if (videoFile) {
                formData.append('video', videoFile);
            }
            
            // 显示加载状态
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                // 隐藏加载状态
                document.getElementById('loading').style.display = 'none';
                
                if (result.success) {
                    displayResult(result.data);
                } else {
                    displayError(result.message);
                }
            } catch (error) {
                document.getElementById('loading').style.display = 'none';
                displayError('网络错误: ' + error.message);
            }
        });
        
        function displayResult(data) {
            const resultDiv = document.getElementById('result');
            const contentDiv = document.getElementById('resultContent');
            
            const sentimentScore = data.sentiment_score;
            const sentimentLabel = data.sentiment_label;
            const modalities = data.modalities_used;
            
            contentDiv.innerHTML = `
                <div class="sentiment-score sentiment-${sentimentLabel.replace('非常', 'very-').replace('负面', 'negative').replace('正面', 'positive').replace('中性', 'neutral')}">
                    情感分数: ${sentimentScore.toFixed(3)} (${sentimentLabel})
                </div>
                <p><strong>使用的模态:</strong> ${getModalitiesText(modalities)}</p>
                <p><strong>分析时间:</strong> ${new Date().toLocaleString()}</p>
            `;
            
            resultDiv.className = 'result success';
            resultDiv.style.display = 'block';
        }
        
        function displayError(message) {
            const resultDiv = document.getElementById('result');
            const contentDiv = document.getElementById('resultContent');
            
            contentDiv.innerHTML = `<p><strong>错误:</strong> ${message}</p>`;
            resultDiv.className = 'result error';
            resultDiv.style.display = 'block';
        }
        
        function getModalitiesText(modalities) {
            const used = [];
            if (modalities.text) used.push('文本');
            if (modalities.audio) used.push('音频');
            if (modalities.video) used.push('视频');
            return used.join(' + ') || '无';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """情感分析API"""
    try:
        # 检查推理器是否初始化
        if inference is None:
            return jsonify({
                'success': False,
                'message': '推理器未初始化，请检查模型文件'
            }), 500
        
        # 获取表单数据
        text = request.form.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'message': '请输入文本内容'
            }), 400
        
        # 处理文件上传
        audio_path = None
        video_path = None
        
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file and audio_file.filename:
                audio_path = save_uploaded_file(audio_file, 'audio')
        
        if 'video' in request.files:
            video_file = request.files['video']
            if video_file and video_file.filename:
                video_path = save_uploaded_file(video_file, 'video')
        
        # 执行分析
        result = inference.predict_sentiment(
            text=text,
            audio_path=audio_path,
            video_path=video_path
        )
        
        # 添加时间戳
        result['timestamp'] = datetime.now().isoformat()
        
        # 清理临时文件
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'分析过程中发生错误: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查API"""
    return jsonify({
        'status': 'healthy',
        'inference_initialized': inference is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/info', methods=['GET'])
def api_info():
    """API信息"""
    return jsonify({
        'name': 'GFMamba 多模态情感分析API',
        'version': '1.0.0',
        'description': '基于Mamba架构的多模态情感分析系统',
        'endpoints': {
            '/': 'Web界面',
            '/analyze': 'POST - 情感分析',
            '/health': 'GET - 健康检查',
            '/api/info': 'GET - API信息'
        },
        'supported_formats': {
            'audio': ['wav', 'mp3', 'm4a'],
            'video': ['mp4', 'avi', 'mov']
        }
    })

if __name__ == '__main__':
    print("🎭 GFMamba 多模态情感分析系统")
    print("=" * 50)
    
    # 初始化推理器
    if init_inference():
        print("✅ 系统初始化成功，启动Web服务...")
        print("🌐 访问地址: http://localhost:5000")
        print("📚 API文档: http://localhost:5000/api/info")
        print("🔍 健康检查: http://localhost:5000/health")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ 系统初始化失败，请检查模型文件和配置")
        sys.exit(1)
