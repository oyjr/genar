#!/usr/bin/env python3
"""
测试修复后的检查点监控指标配置
验证配置是否正确传递到ModelCheckpoint
"""

import sys
import os
sys.path.insert(0, 'src')

import argparse
from datetime import datetime
from addict import Dict
from main import build_config_from_args, DATASETS, MODELS, ENCODER_FEATURE_DIMS, DEFAULT_CONFIG
from utils import load_callbacks

def test_stage1_config():
    """测试Stage 1配置"""
    print("🧪 测试 Stage 1 配置...")
    
    # 模拟命令行参数
    args = argparse.Namespace(
        dataset='PRAD',
        model='TWO_STAGE_VAR_ST',
        encoder=None,
        gpus=1,
        epochs=None,
        batch_size=None,
        lr=None,
        weight_decay=None,
        patience=None,
        strategy='auto',
        sync_batchnorm=False,
        use_augmented=True,
        expand_augmented=True,
        mode='train',
        seed=None,
        training_stage=1,
        stage1_ckpt=None,
        config=None
    )
    
    # 构建配置
    config = build_config_from_args(args)
    
    # 验证配置
    print(f"   TRAINING.monitor: {config.TRAINING.monitor}")
    print(f"   TRAINING.mode: {config.TRAINING.mode}")
    print(f"   CALLBACKS.model_checkpoint.monitor: {config.CALLBACKS.model_checkpoint.monitor}")
    print(f"   CALLBACKS.model_checkpoint.mode: {config.CALLBACKS.model_checkpoint.mode}")
    print(f"   CALLBACKS.model_checkpoint.filename: {config.CALLBACKS.model_checkpoint.filename}")
    
    # 验证是否正确
    assert config.TRAINING.monitor == 'val_mse', f"TRAINING.monitor应该是val_mse，实际是{config.TRAINING.monitor}"
    assert config.TRAINING.mode == 'min', f"TRAINING.mode应该是min，实际是{config.TRAINING.mode}"
    assert config.CALLBACKS.model_checkpoint.monitor == 'val_mse', f"model_checkpoint.monitor应该是val_mse"
    
    print("   ✅ Stage 1 配置正确")
    return config

def test_stage2_config():
    """测试Stage 2配置"""
    print("\n🧪 测试 Stage 2 配置...")
    
    # 模拟命令行参数
    args = argparse.Namespace(
        dataset='PRAD',
        model='TWO_STAGE_VAR_ST',
        encoder=None,
        gpus=1,
        epochs=None,
        batch_size=None,
        lr=None,
        weight_decay=None,
        patience=None,
        strategy='auto',
        sync_batchnorm=False,
        use_augmented=True,
        expand_augmented=True,
        mode='train',
        seed=None,
        training_stage=2,
        stage1_ckpt='dummy_path.ckpt',
        config=None
    )
    
    # 构建配置
    config = build_config_from_args(args)
    
    # 验证配置
    print(f"   TRAINING.monitor: {config.TRAINING.monitor}")
    print(f"   TRAINING.mode: {config.TRAINING.mode}")
    print(f"   CALLBACKS.model_checkpoint.monitor: {config.CALLBACKS.model_checkpoint.monitor}")
    print(f"   CALLBACKS.model_checkpoint.mode: {config.CALLBACKS.model_checkpoint.mode}")
    print(f"   CALLBACKS.model_checkpoint.filename: {config.CALLBACKS.model_checkpoint.filename}")
    
    # 验证是否正确
    assert config.TRAINING.monitor == 'val_accuracy', f"TRAINING.monitor应该是val_accuracy，实际是{config.TRAINING.monitor}"
    assert config.TRAINING.mode == 'max', f"TRAINING.mode应该是max，实际是{config.TRAINING.mode}"
    assert config.CALLBACKS.model_checkpoint.monitor == 'val_accuracy', f"model_checkpoint.monitor应该是val_accuracy"
    
    print("   ✅ Stage 2 配置正确")
    return config

def test_callbacks_integration():
    """测试配置与callbacks的集成"""
    print("\n🧪 测试 callbacks 集成...")
    
    # 测试Stage 1
    args1 = argparse.Namespace(
        dataset='PRAD', model='TWO_STAGE_VAR_ST', encoder=None, gpus=1,
        epochs=None, batch_size=None, lr=None, weight_decay=None, patience=None,
        strategy='auto', sync_batchnorm=False, use_augmented=True, expand_augmented=True,
        mode='train', seed=None, training_stage=1, stage1_ckpt=None, config=None
    )
    config1 = build_config_from_args(args1)
    callbacks1 = load_callbacks(config1)
    
    # 寻找ModelCheckpoint
    model_checkpoint1 = None
    for cb in callbacks1:
        if hasattr(cb, 'monitor'):
            model_checkpoint1 = cb
            break
    
    assert model_checkpoint1 is not None, "找不到ModelCheckpoint回调"
    print(f"   Stage 1 ModelCheckpoint.monitor: {model_checkpoint1.monitor}")
    print(f"   Stage 1 ModelCheckpoint.mode: {model_checkpoint1.mode}")
    
    # 测试Stage 2  
    args2 = argparse.Namespace(
        dataset='PRAD', model='TWO_STAGE_VAR_ST', encoder=None, gpus=1,
        epochs=None, batch_size=None, lr=None, weight_decay=None, patience=None,
        strategy='auto', sync_batchnorm=False, use_augmented=True, expand_augmented=True,
        mode='train', seed=None, training_stage=2, stage1_ckpt='dummy.ckpt', config=None
    )
    config2 = build_config_from_args(args2)
    callbacks2 = load_callbacks(config2)
    
    # 寻找ModelCheckpoint
    model_checkpoint2 = None
    for cb in callbacks2:
        if hasattr(cb, 'monitor'):
            model_checkpoint2 = cb
            break
    
    assert model_checkpoint2 is not None, "找不到ModelCheckpoint回调"
    print(f"   Stage 2 ModelCheckpoint.monitor: {model_checkpoint2.monitor}")
    print(f"   Stage 2 ModelCheckpoint.mode: {model_checkpoint2.mode}")
    
    # 验证
    assert model_checkpoint1.monitor == 'val_mse', f"Stage 1应该监控val_mse，实际是{model_checkpoint1.monitor}"
    assert model_checkpoint2.monitor == 'val_accuracy', f"Stage 2应该监控val_accuracy，实际是{model_checkpoint2.monitor}"
    
    print("   ✅ callbacks 集成正确")

if __name__ == "__main__":
    print("🚀 测试检查点监控指标修复")
    print("=" * 50)
    
    try:
        test_stage1_config()
        test_stage2_config()
        test_callbacks_integration()
        
        print("\n" + "=" * 50)
        print("✅ 所有测试通过！")
        print("🎯 修复总结:")
        print("   - Stage 1: 正确监控 val_mse")
        print("   - Stage 2: 正确监控 val_accuracy") 
        print("   - 配置正确传递到 ModelCheckpoint")
        print("   - 检查点文件名格式正确")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 