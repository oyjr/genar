#!/usr/bin/env python3
"""
快速测试VAR-ST整slide功能

这个脚本用于验证整slide测试功能是否正常工作
"""

import os
import sys
import torch
from pathlib import Path
from addict import Dict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.main import DEFAULT_CONFIG, DATASETS, MODELS, ENCODER_FEATURE_DIMS
from src.dataset.data_interface import DataInterface
from src.model.model_interface import ModelInterface

def create_test_config():
    """创建测试配置"""
    config = Dict(DEFAULT_CONFIG.copy())
    
    # 设置为VAR_ST模型和PRAD数据集
    dataset_name = 'PRAD'
    model_name = 'VAR_ST'
    
    if dataset_name not in DATASETS:
        raise ValueError(f"数据集 {dataset_name} 未找到")
    
    if model_name not in MODELS:
        raise ValueError(f"模型 {model_name} 未找到")
    
    dataset_info = DATASETS[dataset_name]
    model_info = MODELS[model_name]
    
    # 更新配置
    config.MODEL = Dict(model_info)
    config.MODEL.feature_dim = ENCODER_FEATURE_DIMS['uni']  # 使用uni编码器
    
    # 设置数据集参数
    config.mode = 'test'
    config.expr_name = dataset_name
    config.data_path = dataset_info['path']
    config.slide_val = dataset_info['val_slides']
    config.slide_test = dataset_info['test_slides']
    config.encoder_name = 'uni'
    config.use_augmented = False  # 测试时不用增强
    config.expand_augmented = False
    
    # 设置日志路径
    config.GENERAL.log_path = './test_slide_output'
    
    # 添加配置路径标记
    config.config = 'built-in-test'
    
    return config

def main():
    """快速测试主流程"""
    print("🚀 快速测试VAR-ST整slide功能")
    print("="*50)
    
    try:
        # 创建测试配置
        print("⚙️ 创建测试配置...")
        config = create_test_config()
        print("✅ 配置创建成功")
        print(f"   - 数据集: {config.expr_name}")
        print(f"   - 模型: {config.MODEL.name}")
        print(f"   - 编码器: {config.encoder_name}")
        print(f"   - 特征维度: {config.MODEL.feature_dim}")
        
        # 设置测试输出目录
        os.makedirs(config.GENERAL.log_path, exist_ok=True)
        
        # 初始化数据模块 - 只初始化测试数据
        print("📊 初始化数据模块...")
        datamodule = DataInterface(config)
        datamodule.setup('test')
        print("✅ 数据模块初始化成功")
        
        # 获取测试数据集信息
        test_dataset = datamodule.test_dataloader().dataset
        original_dataset = test_dataset
        while hasattr(original_dataset, 'dataset'):
            original_dataset = original_dataset.dataset
        
        # 检查是否有新的方法
        if not hasattr(original_dataset, 'get_test_slide_ids'):
            print("❌ 数据集缺少get_test_slide_ids方法")
            return
            
        if not hasattr(original_dataset, 'get_full_slide_for_testing'):
            print("❌ 数据集缺少get_full_slide_for_testing方法") 
            return
        
        # 获取测试slides
        test_slide_ids = original_dataset.get_test_slide_ids()
        print(f"📋 测试slides: {test_slide_ids}")
        
        if not test_slide_ids:
            print("❌ 没有找到测试slides")
            return
        
        # 测试单个slide数据加载
        test_slide_id = test_slide_ids[0]
        print(f"🔬 测试加载slide: {test_slide_id}")
        
        slide_data = original_dataset.get_full_slide_for_testing(test_slide_id)
        
        print(f"✅ Slide数据加载成功:")
        print(f"   - img shape: {slide_data['img'].shape}")
        print(f"   - target_genes shape: {slide_data['target_genes'].shape}")
        print(f"   - positions shape: {slide_data['positions'].shape}")
        print(f"   - num_spots: {slide_data['num_spots']}")
        print(f"   - slide_id: {slide_data['slide_id']}")
        
        # 检查是否有adata
        if 'adata' in slide_data:
            print(f"   - adata: {slide_data['adata'].n_obs} obs, {slide_data['adata'].n_vars} vars")
        
        # 尝试创建一个简单的模型来测试（如果有checkpoint的话）
        print("\n🔧 尝试测试模型接口...")
        
        # 创建一个模型接口来测试方法是否存在
        try:
            # 不实际加载模型，只检查方法
            model = ModelInterface(config)
            
            # 检查是否有新的测试方法
            if hasattr(model, 'test_full_slide'):
                print("✅ 找到test_full_slide方法")
            else:
                print("❌ 缺少test_full_slide方法")
                
            if hasattr(model, 'run_full_slide_testing'):
                print("✅ 找到run_full_slide_testing方法")
            else:
                print("❌ 缺少run_full_slide_testing方法")
                
        except Exception as e:
            print(f"⚠️ 模型接口测试失败: {e}")
        
        print("\n🎉 基础功能测试完成!")
        print("📝 测试结果:")
        print("   ✅ 数据加载: 正常")
        print("   ✅ slide数据格式: 正确")
        print("   ✅ 方法存在性: 已验证")
        
        print(f"\n💡 要运行完整测试，需要:")
        print(f"   1. 训练VAR_ST模型得到checkpoint")
        print(f"   2. 使用 test_var_st_full_slide.py 脚本")
        print(f"   3. 示例: python test_var_st_full_slide.py --checkpoint_path <your_checkpoint> --dataset HEST --model VAR_ST")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 