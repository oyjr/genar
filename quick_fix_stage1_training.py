#!/usr/bin/env python3
"""
快速修复Stage 1训练 - 解决Codebook Collapse

最小化修改，最大化效果的方案：
1. 调整关键训练参数
2. 修改主训练脚本的调用方式
3. 不改变现有架构，只优化训练策略
"""

import os
import sys

# 快速修复的训练命令
QUICK_FIX_COMMAND = """
python src/main.py \\
    --dataset PRAD \\
    --model TWO_STAGE_VAR_ST \\
    --training_stage 1 \\
    --epochs 200 \\
    --batch_size 128 \\
    --learning_rate 3e-4 \\
    --gpus 1 \\
    --additional_args '{
        "MODEL": {
            "beta": 1.0,
            "hierarchical_loss_weight": 0.2,
            "vq_loss_weight": 0.5,
            "use_ema_update": true,
            "ema_decay": 0.99,
            "restart_unused_codes": true,
            "restart_interval": 10
        },
        "OPTIMIZER": {
            "name": "AdamW",
            "weight_decay": 1e-5,
            "gradient_clip_val": 1.0
        },
        "SCHEDULER": {
            "name": "cosine_with_warmup",
            "warmup_epochs": 20,
            "min_lr": 1e-5
        }
    }'
"""

def print_quick_fix_guide():
    """打印快速修复指南"""
    print("🚀 Stage 1 Codebook Collapse快速修复方案")
    print("=" * 60)
    
    print("\n📊 问题诊断确认:")
    print("   - Global尺度: 仅2个tokens/4096 (利用率0.05%)")
    print("   - Pathway尺度: 仅2个tokens/4096 (利用率0.05%)")  
    print("   - Module尺度: 仅3个tokens/4096 (利用率0.07%)")
    print("   - Individual尺度: 仅4个tokens/4096 (利用率0.10%)")
    print("   ❌ 这是严重的Codebook Collapse！")
    
    print("\n🔧 关键参数修改:")
    print("   1. Beta: 0.25 → 1.0 (4倍commitment loss权重)")
    print("   2. Learning Rate: 默认 → 3e-4 (提高学习率)")
    print("   3. Batch Size: 256 → 128 (增加更新频率)")
    print("   4. VQ Loss Weight: 0.25 → 0.5 (强化量化学习)")
    print("   5. 添加EMA更新和代码重启机制")
    
    print(f"\n🚀 推荐训练命令:")
    print(QUICK_FIX_COMMAND)
    
    print("\n📈 预期改善:")
    print("   - Codebook利用率提升到10-30%")
    print("   - Token多样性显著增加")
    print("   - 推理结果回归正常范围")
    
    print("\n⏰ 预计训练时间:")
    print("   - 约200 epochs，根据GPU性能约4-8小时")
    print("   - 建议每20个epoch检查一次利用率")

def create_improved_main_args():
    """创建改进的主训练参数"""
    return {
        'dataset': 'PRAD',
        'model': 'TWO_STAGE_VAR_ST',
        'training_stage': 1,
        'epochs': 200,
        'batch_size': 128,
        'learning_rate': 3e-4,
        'gpus': 1,
        
        # 🔧 关键改进参数
        'model_config_overrides': {
            'vqvae_config': {
                'vocab_size': 4096,
                'embed_dim': 128,
                'beta': 1.0,  # 🔧 关键：4倍commitment loss权重
                'hierarchical_loss_weight': 0.2,  # 🔧 增强分层学习
                'vq_loss_weight': 0.5,  # 🔧 强化VQ学习
            }
        },
        
        'optimizer_config': {
            'name': 'AdamW',
            'lr': 3e-4,  # 🔧 提高学习率
            'weight_decay': 1e-5,
            'gradient_clip_val': 1.0  # 🔧 梯度裁剪
        },
        
        'scheduler_config': {
            'name': 'cosine_with_warmup',
            'warmup_epochs': 20,  # 🔧 长预热
            'min_lr': 1e-5
        }
    }

# ========================================
# 阶段性解决方案
# ========================================

def print_staged_solution():
    """打印分阶段解决方案"""
    print("\n🎯 分阶段解决方案路线图")
    print("=" * 50)
    
    print("\n【阶段1 - 立即实施】参数优化 (推荐)")
    print("   修改内容: 只改训练参数，不动架构")
    print("   实施难度: ⭐ (Very Easy)")
    print("   预期效果: ⭐⭐⭐ (High Impact)")
    print("   时间成本: 4-8小时重新训练")
    print("   风险: 极低")
    
    print("\n【阶段2 - 如果阶段1效果不佳】架构改进")
    print("   修改内容: 更深编码器+EMA更新+代码重启")
    print("   实施难度: ⭐⭐ (Easy)")
    print("   预期效果: ⭐⭐⭐⭐ (Very High)")
    print("   时间成本: 1-2天开发+重新训练")
    print("   风险: 中等")
    
    print("\n【阶段3 - 如果前两阶段无效】替换量化方法")
    print("   修改内容: 使用FSQ或Group Quantization")
    print("   实施难度: ⭐⭐⭐ (Medium)")
    print("   预期效果: ⭐⭐⭐⭐⭐ (Excellent)")
    print("   时间成本: 3-5天开发+重新训练")
    print("   风险: 较高(需要大幅改动)")

def print_monitoring_tips():
    """打印监控建议"""
    print("\n📊 训练监控要点")
    print("=" * 30)
    
    print("关键指标监控:")
    print("   1. Codebook利用率 (每个epoch)")
    print("   2. VQ Loss趋势 (应该逐渐下降)")
    print("   3. Reconstruction Loss (主要指标)")
    print("   4. Token分布熵 (越高越好)")
    
    print("\n早期停止条件:")
    print("   ✅ 利用率>10% 且稳定")
    print("   ✅ VQ Loss收敛")
    print("   ✅ 重建质量满意")
    
    print("\n异常信号:")
    print("   ❌ 利用率持续<5%")
    print("   ❌ VQ Loss不下降或上升")
    print("   ❌ 某些尺度tokens完全相同")

if __name__ == "__main__":
    print_quick_fix_guide()
    print_staged_solution()
    print_monitoring_tips()
    
    print("\n🎯 立即行动建议:")
    print("   1. 先尝试阶段1的参数优化方案")
    print("   2. 每20个epoch检查token多样性")
    print("   3. 如果100个epoch后利用率仍<5%，考虑阶段2")
    print("   4. 记录训练日志，便于后续分析") 