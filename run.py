#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GFMamba 启动脚本
用于快速启动推理或Web服务
"""

import os
import sys
import argparse
import subprocess
import warnings
warnings.filterwarnings("ignore")

def check_dependencies():
    """检查依赖是否安装"""
    try:
        import torch
        import flask
        import librosa
        import cv2
        import transformers
        print("✅ 所有依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False

def check_model_files():
    """检查模型文件是否存在"""
    config_path = "configs/mosi_train.yaml"
    model_path = "ckpt/mosi/best_valid_model_seed_42.pth"
    
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        print("   将使用随机初始化的模型（仅用于测试）")
    
    print("✅ 模型文件检查完成")
    return True

def run_inference():
    """运行推理脚本"""
    print("🔍 启动推理模式...")
    try:
        from inference import main
        main()
    except Exception as e:
        print(f"❌ 推理失败: {e}")

def run_web_app():
    """运行Web应用"""
    print("🌐 启动Web服务...")
    try:
        from app import app, init_inference
        if init_inference():
            app.run(host='0.0.0.0', port=5000, debug=False)
        else:
            print("❌ 推理器初始化失败")
    except Exception as e:
        print(f"❌ Web服务启动失败: {e}")

def run_data_preprocessing():
    """运行数据预处理"""
    print("🔧 启动数据预处理...")
    try:
        from data_preprocessing import main
        main()
    except Exception as e:
        print(f"❌ 数据预处理失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='GFMamba 启动脚本')
    parser.add_argument('--mode', choices=['inference', 'web', 'preprocess', 'check'], 
                       default='web', help='运行模式')
    parser.add_argument('--check-deps', action='store_true', help='检查依赖')
    parser.add_argument('--check-model', action='store_true', help='检查模型文件')
    
    args = parser.parse_args()
    
    print("🎭 GFMamba 多模态情感分析系统")
    print("=" * 50)
    
    # 检查依赖
    if args.check_deps or args.mode != 'check':
        if not check_dependencies():
            sys.exit(1)
    
    # 检查模型文件
    if args.check_model or args.mode != 'check':
        if not check_model_files():
            if args.mode != 'check':
                print("⚠️  继续运行，但可能影响性能")
    
    # 根据模式运行
    if args.mode == 'inference':
        run_inference()
    elif args.mode == 'web':
        run_web_app()
    elif args.mode == 'preprocess':
        run_data_preprocessing()
    elif args.mode == 'check':
        print("✅ 系统检查完成")

if __name__ == "__main__":
    main()
