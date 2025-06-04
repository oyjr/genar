#!/usr/bin/env python3
"""
验证数据加载器性能脚本

此脚本验证：
1. 验证数据加载器是否使用了正确的批次大小
2. 测试数据加载器的性能
3. 提供性能优化建议

使用方法：
python verify_dataloader_performance.py
"""

import sys
import os
sys.path.insert(0, 'src')
import time
import torch
from dataset.data_interface import DataInterface
from main import DATASETS, build_config_from_args, DEFAULT_CONFIG
from addict import Dict as AddictDict
import argparse

def test_dataloader_performance():
    """测试数据加载器性能"""
    print("🧪 验证数据加载器性能...")
    
    # 构建测试配置
    class MockArgs:
        def __init__(self):
            self.dataset = 'PRAD'
            self.model = 'TWO_STAGE_VAR_ST'
            self.training_stage = 1
            self.encoder = None
            self.gpus = 1
            self.epochs = None
            self.batch_size = None
            self.lr = None
            self.weight_decay = None
            self.patience = None
            self.strategy = 'auto'
            self.sync_batchnorm = False
            self.use_augmented = True
            self.expand_augmented = True
            self.mode = 'train'
            self.seed = None
            self.stage1_ckpt = None
            self.config = None
    
    args = MockArgs()
    config = build_config_from_args(args)
    
    print(f"✅ 配置构建完成")
    print(f"   - 训练批次大小: {config.DATA.train_dataloader.batch_size}")
    print(f"   - 验证批次大小: {config.DATA.val_dataloader.batch_size}")
    print(f"   - 测试批次大小: {config.DATA.test_dataloader.batch_size}")
    
    # 创建数据接口
    data_interface = DataInterface(config)
    data_interface.setup(stage='fit')
    
    # 获取数据加载器
    train_loader = data_interface.train_dataloader()
    val_loader = data_interface.val_dataloader()
    
    print(f"\n📊 数据加载器信息:")
    print(f"   - 训练数据集大小: {len(data_interface.train_dataset)}")
    print(f"   - 验证数据集大小: {len(data_interface.val_dataset)}")
    print(f"   - 训练批次数: {len(train_loader)}")
    print(f"   - 验证批次数: {len(val_loader)}")
    
    # 测试训练数据加载器速度
    print(f"\n⏱️  测试训练数据加载器性能...")
    start_time = time.time()
    num_batches = 10
    
    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break
        # 简单处理确保数据加载完成
        _ = batch['target_genes'].shape
    
    train_time = time.time() - start_time
    train_speed = num_batches / train_time
    print(f"   - 训练数据加载器: {train_speed:.2f} batches/sec")
    
    # 测试验证数据加载器速度
    print(f"\n⏱️  测试验证数据加载器性能...")
    start_time = time.time()
    
    for i, batch in enumerate(val_loader):
        if i >= num_batches:
            break
        # 简单处理确保数据加载完成
        _ = batch['target_genes'].shape
    
    val_time = time.time() - start_time
    val_speed = num_batches / val_time
    print(f"   - 验证数据加载器: {val_speed:.2f} batches/sec")
    
    # 计算性能比较
    speed_ratio = val_speed / train_speed if train_speed > 0 else 0
    print(f"\n📈 性能分析:")
    print(f"   - 验证/训练速度比: {speed_ratio:.3f}")
    
    if speed_ratio > 0.5:
        print("✅ 验证数据加载器性能正常")
    else:
        print("⚠️  验证数据加载器性能仍然较慢")
    
    return {
        'train_speed': train_speed,
        'val_speed': val_speed,
        'speed_ratio': speed_ratio,
        'train_batch_size': config.DATA.train_dataloader.batch_size,
        'val_batch_size': config.DATA.val_dataloader.batch_size
    }

def print_performance_tips():
    """打印性能优化建议"""
    print(f"\n🚀 数据加载器性能优化建议:")
    print(f"")
    print(f"1. **批次大小优化**:")
    print(f"   - 验证/测试可以使用更大批次：--batch_size 64 或 128")
    print(f"   - GPU内存允许的情况下，增大批次能显著提升速度")
    print(f"")
    print(f"2. **工作进程优化**:")
    print(f"   - 增加num_workers：在配置中设置为CPU核心数")
    print(f"   - 建议值：4-8个工作进程")
    print(f"")
    print(f"3. **内存优化**:")
    print(f"   - 启用pin_memory=True (已启用)")
    print(f"   - 启用persistent_workers=True (已启用)")
    print(f"")
    print(f"4. **硬件优化**:")
    print(f"   - 使用SSD存储数据")
    print(f"   - 确保足够的内存避免swap")
    print(f"   - 使用高速GPU (V100/A100/RTX系列)")
    print(f"")
    print(f"5. **训练特定优化**:")
    print(f"   - TWO_STAGE_VAR_ST Stage 2可以使用较小批次 (32-64)")
    print(f"   - Stage 1可以使用较大批次 (128-256)")

def main():
    """主函数"""
    print("🔍 数据加载器性能验证工具")
    print("=" * 60)
    
    try:
        # 测试性能
        results = test_dataloader_performance()
        
        # 打印优化建议
        print_performance_tips()
        
        # 总结
        print(f"\n📋 性能验证总结:")
        print(f"   - 训练批次大小: {results['train_batch_size']}")
        print(f"   - 验证批次大小: {results['val_batch_size']}")
        print(f"   - 训练速度: {results['train_speed']:.2f} batches/sec")
        print(f"   - 验证速度: {results['val_speed']:.2f} batches/sec")
        print(f"   - 速度比: {results['speed_ratio']:.3f}")
        
        if results['speed_ratio'] > 0.5:
            print(f"✅ 数据加载器性能正常，修复成功！")
        else:
            print(f"⚠️  仍有性能问题，请检查硬件或进一步优化")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 