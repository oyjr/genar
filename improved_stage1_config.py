#!/usr/bin/env python3
"""
改进的Stage 1 VQVAE训练配置

解决Codebook Collapse问题的完整方案：
1. 优化训练参数
2. 改进量化策略  
3. 增强架构设计
4. 添加正则化技术
"""

# ========================================
# 方案A：训练参数优化
# ========================================

IMPROVED_TRAINING_CONFIG = {
    # 基础训练参数
    'batch_size': 128,  # 🔧 减小batch size，增加梯度更新频率
    'epochs': 200,
    'learning_rate': 3e-4,  # 🔧 提高学习率，帮助codebook快速分化
    'weight_decay': 1e-5,
    
    # 学习率调度
    'lr_scheduler': {
        'type': 'cosine_with_warmup',
        'warmup_epochs': 20,  # 🔧 长预热期
        'min_lr': 1e-5
    },
    
    # 梯度相关
    'gradient_clip_val': 1.0,  # 🔧 梯度裁剪防止训练不稳定
    'accumulate_grad_batches': 2,  # 🔧 梯度累积
}

IMPROVED_VQVAE_CONFIG = {
    'vocab_size': 4096,
    'embed_dim': 128,
    
    # 🔧 关键改进：commitment loss权重
    'beta': 1.0,  # 🔧 从0.25增加到1.0，强化commitment
    
    # 🔧 层级损失权重调整
    'hierarchical_loss_weight': 0.2,  # 从0.1增加到0.2
    'vq_loss_weight': 0.5,  # 从0.25增加到0.5
    
    # 🔧 新增：Codebook利用率正则化
    'utilization_loss_weight': 0.01,
    'target_utilization': 0.1,  # 目标10%利用率
    
    # 🔧 新增：EMA更新策略
    'use_ema_update': True,
    'ema_decay': 0.99,
    'ema_epsilon': 1e-5,
    
    # 🔧 新增：重启不活跃codes
    'restart_unused_codes': True,
    'restart_threshold': 1.0,  # epoch级别的重启阈值
    'restart_interval': 10,  # 每10个epoch检查一次
}

# ========================================
# 方案B：改进的VQVAE架构
# ========================================

IMPROVED_ARCHITECTURE_CONFIG = {
    # 🔧 更深的编码器
    'encoder_config': {
        'use_residual_blocks': True,
        'num_residual_blocks': 3,
        'hidden_dims': [256, 512, 256, 128],  # 更深的网络
        'activation': 'gelu',
        'dropout': 0.1,
        'layer_norm': True
    },
    
    # 🔧 更深的解码器
    'decoder_config': {
        'use_residual_blocks': True,
        'num_residual_blocks': 3,
        'hidden_dims': [128, 256, 512, 256],  # 对称设计
        'activation': 'gelu',
        'dropout': 0.1,
        'layer_norm': True
    },
    
    # 🔧 改进的量化策略
    'quantizer_config': {
        'distance_metric': 'euclidean',  # 欧几里得距离
        'normalize_embeddings': True,  # L2标准化
        'temperature': 1.0,  # 温度参数
        'use_cosine_similarity': False,  # 不使用余弦相似度
    }
}

# ========================================  
# 方案C：替代量化方法 (FSQ)
# ========================================

FSQ_CONFIG = {
    'use_fsq': True,  # 🔧 使用Finite Scalar Quantization
    'fsq_levels': [8, 5, 5, 5],  # 对应1000个codes，接近1024
    'fsq_dim': 4,  # FSQ维度
    'need_project': True,  # 启用维度投影
    'project_dim': 128,  # 投影到128维
}

# ========================================
# 方案D：Group/Product Quantization  
# ========================================

GROUP_QUANTIZATION_CONFIG = {
    'use_group_quantization': True,
    'num_groups': 4,  # 🔧 将128维分成4组，每组32维
    'group_vocab_size': 256,  # 每组词汇表大小256
    'total_combinations': 256**4,  # 总共4B+组合
}

# ========================================
# 方案E：Residual Vector Quantization
# ========================================

RVQ_CONFIG = {
    'use_rvq': True,
    'num_quantizers': 4,  # 🔧 4层残差量化
    'vocab_size_per_layer': 1024,  # 每层1024词汇
    'shared_codebook': False,  # 每层独立codebook
}

# ========================================
# 数据预处理改进
# ========================================

DATA_PREPROCESSING_CONFIG = {
    # 🔧 减少过度标准化
    'gene_normalization': {
        'method': 'log1p_only',  # 只做log1p，不做z-score
        'clip_outliers': True,
        'clip_percentile': [1, 99],  # 裁剪1%和99%分位数
    },
    
    # 🔧 数据增强
    'augmentation': {
        'add_noise': True,
        'noise_std': 0.05,  # 小幅噪声增强多样性
        'random_scale': True,
        'scale_range': [0.95, 1.05]
    }
}

# ========================================
# 训练策略
# ========================================

TRAINING_STRATEGY_CONFIG = {
    # 🔧 分阶段训练
    'staged_training': {
        'stage1_epochs': 50,   # 先训练重建
        'stage1_vq_weight': 0.1,  # 初期VQ权重较小
        
        'stage2_epochs': 100,  # 增强VQ训练  
        'stage2_vq_weight': 0.5,  # 后期VQ权重较大
        
        'stage3_epochs': 50,   # 精调阶段
        'stage3_vq_weight': 1.0,  # 最大VQ权重
    },
    
    # 🔧 自适应权重调整
    'adaptive_weights': {
        'monitor_utilization': True,
        'target_utilization': 0.1,
        'adjust_beta_dynamically': True,
        'beta_schedule': 'exponential'  # 指数增长
    },
    
    # 🔧 定期codebook重启
    'codebook_maintenance': {
        'reset_unused_codes': True,
        'reset_interval': 1000,  # 每1000步
        'usage_threshold': 10,   # 使用次数阈值
        'reset_method': 'kmeans'  # 使用k-means重新初始化
    }
}

# ========================================
# 监控和诊断
# ========================================

MONITORING_CONFIG = {
    'log_interval': 100,
    'validate_interval': 500,
    
    # 🔧 codebook监控
    'codebook_metrics': {
        'track_utilization': True,
        'track_entropy': True,
        'track_dead_codes': True,
        'save_usage_heatmap': True
    },
    
    # 🔧 重建质量监控
    'reconstruction_metrics': {
        'mse': True,
        'mae': True, 
        'pcc': True,  # Pearson相关系数
        'r2': True    # R方
    }
}

# ========================================
# 完整配置组合
# ========================================

def get_improved_config(method='enhanced_vqvae'):
    """
    获取改进配置
    
    Args:
        method: 改进方法
            - 'enhanced_vqvae': 增强的VQVAE
            - 'fsq': Finite Scalar Quantization  
            - 'group_vq': Group Vector Quantization
            - 'rvq': Residual Vector Quantization
    """
    base_config = {
        'training': IMPROVED_TRAINING_CONFIG,
        'architecture': IMPROVED_ARCHITECTURE_CONFIG,
        'data': DATA_PREPROCESSING_CONFIG,
        'strategy': TRAINING_STRATEGY_CONFIG,
        'monitoring': MONITORING_CONFIG
    }
    
    if method == 'enhanced_vqvae':
        base_config['vqvae'] = IMPROVED_VQVAE_CONFIG
        
    elif method == 'fsq':
        base_config['vqvae'] = {**IMPROVED_VQVAE_CONFIG, **FSQ_CONFIG}
        
    elif method == 'group_vq':
        base_config['vqvae'] = {**IMPROVED_VQVAE_CONFIG, **GROUP_QUANTIZATION_CONFIG}
        
    elif method == 'rvq':
        base_config['vqvae'] = {**IMPROVED_VQVAE_CONFIG, **RVQ_CONFIG}
    
    return base_config

# ========================================
# 使用示例
# ========================================

if __name__ == "__main__":
    # 获取不同的配置
    enhanced_config = get_improved_config('enhanced_vqvae')
    fsq_config = get_improved_config('fsq')
    
    print("🔧 Enhanced VQVAE配置:")
    print(f"   Beta: {enhanced_config['vqvae']['beta']}")
    print(f"   利用率权重: {enhanced_config['vqvae']['utilization_loss_weight']}")
    print(f"   EMA更新: {enhanced_config['vqvae']['use_ema_update']}")
    
    print("\n🎯 FSQ配置:")
    print(f"   FSQ levels: {fsq_config['vqvae']['fsq_levels']}")
    print(f"   总codes: {8*5*5*5}") 