#!/usr/bin/env python3
"""
VAR-ST 整slide测试脚本

功能：
1. 加载训练好的VAR-ST模型
2. 对测试集中的每个slide进行完整测试
3. 逐spot预测，最后整合成完整的slide结果
4. 计算详细的评价指标和可视化

使用方法：
    python test_var_st_full_slide.py --checkpoint_path path/to/checkpoint.ckpt --dataset HEST --model VAR_ST
"""

import os
import sys
import torch
import argparse
import pytorch_lightning as pl
from pathlib import Path
from addict import Dict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.main import DEFAULT_CONFIG, DATASETS, MODELS, ENCODER_FEATURE_DIMS
from src.dataset.data_interface import DataInterface
from src.model.model_interface import ModelInterface

def create_test_config(dataset_name='PRAD', model_name='VAR_ST', encoder_name='uni'):
    """
    创建测试配置
    
    Args:
        dataset_name: 数据集名称
        model_name: 模型名称
        encoder_name: 编码器名称
    
    Returns:
        配置对象
    """
    config = Dict(DEFAULT_CONFIG.copy())
    
    if dataset_name not in DATASETS:
        raise ValueError(f"数据集 {dataset_name} 未找到，支持的数据集: {list(DATASETS.keys())}")
    
    if model_name not in MODELS:
        raise ValueError(f"模型 {model_name} 未找到，支持的模型: {list(MODELS.keys())}")
    
    if encoder_name not in ENCODER_FEATURE_DIMS:
        raise ValueError(f"编码器 {encoder_name} 未找到，支持的编码器: {list(ENCODER_FEATURE_DIMS.keys())}")
    
    dataset_info = DATASETS[dataset_name]
    model_info = MODELS[model_name]
    
    # 更新配置
    config.MODEL = Dict(model_info)
    config.MODEL.feature_dim = ENCODER_FEATURE_DIMS[encoder_name]
    
    # 设置数据集参数
    config.mode = 'test'
    config.expr_name = dataset_name
    config.data_path = dataset_info['path']
    config.slide_val = dataset_info['val_slides']
    config.slide_test = dataset_info['test_slides']
    config.encoder_name = encoder_name
    config.use_augmented = False  # 测试时不用增强
    config.expand_augmented = False
    
    # 添加配置路径标记
    config.config = 'built-in-test'
    
    return config

def main():
    """主测试流程"""
    parser = argparse.ArgumentParser(description="VAR-ST整slide测试")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default='HEST',
                       choices=list(DATASETS.keys()),
                       help='数据集名称')
    parser.add_argument('--model', type=str, default='VAR_ST',
                       choices=list(MODELS.keys()),
                       help='模型名称')
    parser.add_argument('--encoder', type=str, default='uni',
                       choices=list(ENCODER_FEATURE_DIMS.keys()),
                       help='编码器名称')
    parser.add_argument('--gpu', type=int, default=0,
                       help='使用的GPU ID')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 检查文件存在性
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint文件不存在: {args.checkpoint_path}")
        return
    
    print("🚀 VAR-ST 整slide测试启动")
    print(f"🔧 Checkpoint: {args.checkpoint_path}")
    print(f"📊 数据集: {args.dataset}")
    print(f"🤖 模型: {args.model}")
    print(f"🔍 编码器: {args.encoder}")
    print(f"💾 输出目录: {args.output_dir}")
    print(f"🖥️  GPU: {args.gpu}")
    print("="*60)
    
    # 创建配置
    try:
        config = create_test_config(args.dataset, args.model, args.encoder)
        print("✅ 配置创建成功")
    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
        return
    
    # 设置输出目录
    config.GENERAL.log_path = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    if torch.cuda.is_available() and args.gpu >= 0:
        device = f"cuda:{args.gpu}"
        print(f"✅ 使用GPU: {device}")
    else:
        device = "cpu"
        print("⚠️ 使用CPU运行")
    
    # 初始化数据模块
    try:
        print("📊 初始化数据模块...")
        datamodule = DataInterface(config)
        datamodule.setup('test')
        print("✅ 数据模块初始化成功")
        
        # 显示数据集信息
        test_dataset = datamodule.test_dataloader().dataset
        original_dataset = test_dataset
        while hasattr(original_dataset, 'dataset'):
            original_dataset = original_dataset.dataset
        
        test_slide_ids = original_dataset.get_test_slide_ids()
        print(f"📋 测试slides: {test_slide_ids}")
        
    except Exception as e:
        print(f"❌ 数据模块初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载模型
    try:
        print("🔧 加载模型...")
        model = ModelInterface.load_from_checkpoint(
            args.checkpoint_path,
            config=config,
            map_location=device
        )
        model = model.to(device)
        model.eval()
        print("✅ 模型加载成功")
        
        # 验证模型类型
        if not (hasattr(model, 'model_name') and model.model_name == args.model):
            print(f"⚠️ 警告: checkpoint中的模型可能不是{args.model}类型")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 设置trainer（用于访问datamodule）
    trainer = pl.Trainer(
        devices=[args.gpu] if torch.cuda.is_available() and args.gpu >= 0 else 'auto',
        accelerator='gpu' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu',
        logger=False,  # 禁用日志
        enable_checkpointing=False,  # 禁用checkpoint
        enable_progress_bar=False  # 禁用进度条
    )
    
    # 将datamodule绑定到trainer
    trainer.datamodule = datamodule
    model.trainer = trainer
    
    # 运行整slide测试
    try:
        print("\n🎯 开始整slide测试...")
        results = model.run_full_slide_testing()
        
        # 输出总结
        print("\n" + "="*60)
        print("🎉 测试完成总结")
        print("="*60)
        print(f"✅ 成功测试 {len(results['test_slide_ids'])} 个slides")
        print(f"📊 总spots数量: {results['overall_predictions'].shape[0]}")
        print(f"🧬 基因数量: {results['overall_predictions'].shape[1]}")
        print(f"🎯 整体性能:")
        
        overall_metrics = results['overall_metrics']
        print(f"   - PCC-10:  {overall_metrics['PCC-10']:.4f}")
        print(f"   - PCC-50:  {overall_metrics['PCC-50']:.4f}")
        print(f"   - PCC-200: {overall_metrics['PCC-200']:.4f}")
        print(f"   - MSE:     {overall_metrics['MSE']:.4f}")
        print(f"   - MAE:     {overall_metrics['MAE']:.4f}")
        print(f"   - RVD:     {overall_metrics['RVD']:.4f}")
        
        print(f"\n💾 结果已保存到: {args.output_dir}")
        print("📁 文件结构:")
        print(f"   - test_results/: 各slide详细结果")
        print(f"   - vis/: 可视化图表")
        print(f"   - overall_results.txt: 整体评估报告")
        
        # 显示各slide结果摘要
        print(f"\n📋 各Slide结果摘要:")
        for slide_id, slide_result in results['slide_results'].items():
            metrics = slide_result['metrics']
            num_spots = slide_result['num_spots']
            print(f"   {slide_id}: {num_spots} spots, PCC-10={metrics['PCC-10']:.4f}, MSE={metrics['MSE']:.4f}")
        
    except Exception as e:
        print(f"❌ 整slide测试失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎊 VAR-ST整slide测试成功完成!")

if __name__ == "__main__":
    main() 