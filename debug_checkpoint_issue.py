#!/usr/bin/env python3
"""
调试检查点更新问题的脚本
检查PyTorch Lightning如何监控和保存检查点
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os
from pathlib import Path

def analyze_checkpoint_issue():
    print("🔍 分析检查点更新问题")
    print("=" * 60)
    
    # 1. 检查现有检查点文件
    ckpt_dir = "logs/PRAD/TWO_STAGE_VAR_ST/"
    if os.path.exists(ckpt_dir):
        print(f"📁 检查点目录: {ckpt_dir}")
        ckpt_files = list(Path(ckpt_dir).glob("*.ckpt"))
        
        for ckpt_file in sorted(ckpt_files, key=lambda x: x.stat().st_mtime, reverse=True):
            stat = ckpt_file.stat()
            import time
            time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime))
            size_mb = stat.st_size / (1024 * 1024)
            print(f"  📄 {ckpt_file.name}")
            print(f"      时间: {time_str}")
            print(f"      大小: {size_mb:.1f} MB")
            
            # 尝试加载检查点并检查指标
            try:
                ckpt = torch.load(ckpt_file, map_location='cpu')
                
                # 检查保存的指标
                if 'epoch' in ckpt:
                    print(f"      Epoch: {ckpt['epoch']}")
                if 'global_step' in ckpt:
                    print(f"      Global Step: {ckpt['global_step']}")
                    
                # 寻找 val_mse 相关信息
                for key in ckpt.keys():
                    if 'val_mse' in str(key).lower():
                        print(f"      {key}: {ckpt[key]}")
                        
                # 检查保存的指标历史
                if 'callbacks' in ckpt:
                    callbacks = ckpt['callbacks']
                    for cb_name, cb_state in callbacks.items():
                        if 'ModelCheckpoint' in cb_name:
                            print(f"      ModelCheckpoint状态:")
                            if hasattr(cb_state, 'monitor') or 'monitor' in cb_state:
                                monitor = cb_state.get('monitor', 'unknown')
                                print(f"        监控指标: {monitor}")
                            if hasattr(cb_state, 'best_model_score') or 'best_model_score' in cb_state:
                                best_score = cb_state.get('best_model_score', 'unknown')
                                print(f"        最佳分数: {best_score}")
                                
            except Exception as e:
                print(f"      ❌ 加载检查点失败: {e}")
            
            print()
    else:
        print(f"❌ 检查点目录不存在: {ckpt_dir}")
    
    print("\n🔧 检查点文件名分析:")
    print("根据文件名分析:")
    print("- stage1-best-epoch=epoch=00-val_mse=val_mse=3.2939.ckpt")
    print("  问题: 出现了重复的字段名 'val_mse=val_mse='")
    print("  这可能是因为指标名称和格式字符串不匹配")
    
    print("\n🔧 可能的原因:")
    print("1. PyTorch Lightning没有接收到正确的val_mse指标")
    print("2. 指标记录的名称与ModelCheckpoint监控的名称不匹配")
    print("3. 检查点文件名格式字符串有问题")
    
    print("\n🔧 验证当前训练状态:")
    print("从进度条看到: val_mse=2.250")
    print("从文件名看到: val_mse=3.2939")
    print("2.250 < 3.2939，应该触发检查点更新")
    print("但是文件没有更新，说明检查点机制有问题")

def test_checkpoint_format():
    print("\n🧪 测试检查点格式字符串")
    print("=" * 60)
    
    # 模拟检查点文件名格式
    formats = [
        'stage1-best-epoch={epoch:02d}-val_mse={val_mse:.4f}',  # 当前格式
        'stage1-best-epoch={epoch:02d}-mse={val_mse:.4f}',     # 修复格式
        'stage1-best-{epoch:02d}-{val_mse:.4f}',               # 简化格式
    ]
    
    # 模拟指标值
    epoch = 1
    val_mse = 2.250
    
    for fmt in formats:
        try:
            # 测试格式字符串
            filename = fmt.format(epoch=epoch, val_mse=val_mse)
            print(f"✅ 格式: {fmt}")
            print(f"   结果: {filename}")
        except Exception as e:
            print(f"❌ 格式: {fmt}")
            print(f"   错误: {e}")
        print()

def check_metric_logging():
    print("\n🔍 检查指标记录机制")
    print("=" * 60)
    
    print("在 model_interface.py 中检查 val_mse 记录:")
    print("1. _update_metrics 函数:")
    print("   - 通过 metrics.update() 更新指标")
    print("   - 通过 self.log() 记录指标")
    print("   - 指标名称: f'{stage}_{name}' -> 'val_mse'")
    
    print("\n2. validation_step 函数:")
    print("   - 调用 _update_metrics('val', logits, target_genes)")
    print("   - 应该记录 val_mse 指标")
    
    print("\n3. ModelCheckpoint 配置:")
    print("   - monitor='val_mse'")
    print("   - mode='min'")
    print("   - filename='stage1-best-epoch={epoch:02d}-val_mse={val_mse:.4f}'")
    
    print("\n🔧 可能的问题:")
    print("1. 指标名称不匹配")
    print("2. 指标没有正确记录到 PyTorch Lightning")
    print("3. 检查点回调没有正确配置")

if __name__ == "__main__":
    analyze_checkpoint_issue()
    test_checkpoint_format()
    check_metric_logging()
    
    print("\n🎯 建议的解决方案:")
    print("1. 检查 self.log('val_mse', ...) 是否正确执行")
    print("2. 修复检查点文件名格式，避免重复的字段名")
    print("3. 确保 ModelCheckpoint.monitor 与实际记录的指标名称一致")
    print("4. 添加调试日志确认指标值是否正确传递给检查点回调") 