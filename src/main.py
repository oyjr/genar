import os
import sys
import argparse
import logging
from datetime import datetime
from glob import glob
from pathlib import Path

from typing import Dict, Any

# 确保导入项目目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning import loggers as pl_loggers

# 导入项目模块
from dataset.data_interface import DataInterface
from model import ModelInterface
from utils import (
    load_callbacks,
    load_loggers,
    fix_seed
)

# 设置日志记录器
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision('high')


# 编码器特征维度映射
ENCODER_FEATURE_DIMS = {
    'uni': 1024,
    'conch': 512
}

# 数据集配置 - 包含路径和slide划分信息
DATASETS = {
    'PRAD': {
        'path': '/data/ouyangjiarui/stem/hest1k_datasets/PRAD/',
        'val_slides': 'MEND139',
        'test_slides': 'MEND140',
        'recommended_encoder': 'uni'
    },
    'her2st': {
        'path': '/data/ouyangjiarui/stem/hest1k_datasets/her2st/',
        'val_slides': 'A1,B1',
        'test_slides': 'C1,D1', 
        'recommended_encoder': 'conch'
    }
}

# 模型配置
MODELS = {
    'MFBP': {
        'model_name': 'MFBP',
        'num_genes': 200,
        'dropout_rate': 0.1
    }
}

# 默认训练配置 - 从base_config.yaml提取的核心配置
DEFAULT_CONFIG = {
    'GENERAL': {
        'seed': 2021,
        'log_path': './logs',  # Will be updated to dataset-specific path
        'debug': False
    },
    'DATA': {
        'normalize': True,  # STEm方式: log2(+1)变换
        'train_dataloader': {
            'batch_size': 256,
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': True,
            'persistent_workers': True
        },
        'val_dataloader': {
            'batch_size': 1,
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': False,
            'persistent_workers': True
        },
        'test_dataloader': {
            'batch_size': 1,
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': False,
            'persistent_workers': True
        }
    },
    'TRAINING': {
        'num_epochs': 200,
        'learning_rate': 1.0e-4,
        'weight_decay': 1.0e-4,
        'mode': 'min',
        'monitor': 'val_loss',
        'lr_scheduler': {
            'monitor': 'val_loss',
            'patience': 5,
            'factor': 0.5,
            'mode': 'min'
        },
        'gradient_clip_val': 1.0
    },
    'CALLBACKS': {
        'early_stopping': {
            'monitor': 'val_loss',
            'patience': 10,
            'mode': 'min',
            'min_delta': 0.0
        },
        'model_checkpoint': {
            'monitor': 'val_loss',
            'save_top_k': 1,
            'mode': 'min',
            'filename': 'epoch={epoch}-val_loss={val_loss:.4f}'
        },
        'learning_rate_monitor': {
            'logging_interval': 'epoch'
        }
    },
    'MULTI_GPU': {
        'strategy': 'ddp',
        'sync_batchnorm': True,
        'find_unused_parameters': False,
        'lr_scaling': 'linear',
        'base_lr': 1.0e-4,
        'accumulate_grad_batches': 1
    }
}





def get_parse():
    """
    Parse command line arguments for simplified MFBP training.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser with simplified parameters
    """
    parser = argparse.ArgumentParser(
        description='Simplified Training for Spatial Transcriptomics Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python src/main.py --dataset PRAD --gpus 4
  
  # With custom parameters
  python src/main.py --dataset PRAD --model MFBP --encoder uni \\
      --gpus 4 --epochs 200 --batch_size 256 --lr 1e-4
  
  # Single GPU training
  python src/main.py --dataset her2st --gpus 1
  
  # Test mode
  python src/main.py --dataset PRAD --gpus 1 --mode test
        """
    )
    
    # === 核心参数 ===
    parser.add_argument('--dataset', type=str, choices=list(DATASETS.keys()),
                        help='数据集名称 (PRAD, her2st)')
    parser.add_argument('--model', type=str, default='MFBP', choices=list(MODELS.keys()),
                        help='模型名称 (默认: MFBP)')
    parser.add_argument('--encoder', type=str, choices=list(ENCODER_FEATURE_DIMS.keys()),
                        help='编码器类型 (uni, conch)，默认使用数据集推荐编码器')
    
    # === 训练参数 ===
    parser.add_argument('--gpus', type=int, default=1,
                        help='GPU数量 (默认: 1)')
    parser.add_argument('--epochs', type=int,
                        help='训练轮数 (默认: 200)')
    parser.add_argument('--batch_size', type=int,
                        help='批次大小 (默认: 256)')
    parser.add_argument('--lr', type=float,
                        help='学习率 (默认: 1e-4)')
    parser.add_argument('--weight-decay', type=float,
                        help='权重衰减 (默认: 1e-4)')
    
    # === 多GPU参数 ===
    parser.add_argument('--strategy', type=str, default='auto',
                        choices=['auto', 'ddp', 'ddp_spawn', 'dp'],
                        help='多GPU策略 (默认: auto，多GPU时使用ddp)')
    parser.add_argument('--sync-batchnorm', action='store_true',
                        help='启用同步BatchNorm (多GPU训练推荐)')
    
    # === 数据增强参数 ===
    parser.add_argument('--use-augmented', action='store_true', default=True,
                        help='使用数据增强 (默认: True)')
    parser.add_argument('--expand-augmented', action='store_true', default=True,
                        help='展开增强数据为7倍样本 (默认: True)')
    
    # === 其他参数 ===
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='运行模式 (默认: train)')
    parser.add_argument('--seed', type=int,
                        help='随机种子 (默认: 2021)')
    
    # === 向后兼容参数 (保留最少必要的) ===
    parser.add_argument('--config', type=str,
                        help='[已弃用] 请使用 --dataset 参数替代')
    
    return parser


def build_config_from_args(args):
    """
    从简化的命令行参数构建完整配置
    
    Args:
        args: 解析后的命令行参数
        
    Returns:
        完整的配置对象
    """
    from addict import Dict
    
    # 如果使用了原有的config参数，则使用原有逻辑
    if args.config:
        print("🔄 使用原有配置文件模式")
        return None  # 返回None表示使用原有逻辑
    
    # 检查必需参数
    if not args.dataset:
        raise ValueError("必须指定 --dataset 参数")
    
    if args.dataset not in DATASETS:
        raise ValueError(f"不支持的数据集: {args.dataset}，支持的数据集: {list(DATASETS.keys())}")
    
    print(f"🚀 使用简化配置模式: 数据集={args.dataset}, 模型={args.model}")
    
    # 获取数据集信息
    dataset_info = DATASETS[args.dataset]
    
    # 获取模型信息
    model_info = MODELS[args.model]
    
    # 确定编码器
    encoder_name = args.encoder or dataset_info['recommended_encoder']
    
    # 确定GPU相关参数
    devices = args.gpus
    strategy = 'ddp' if devices > 1 and args.strategy == 'auto' else args.strategy
    sync_batchnorm = getattr(args, 'sync_batchnorm', False) or (devices > 1)
    
    # 构建完整配置
    config = Dict(DEFAULT_CONFIG)
    
    # 更新日志路径为数据集名称和模型名称
    config.GENERAL.log_path = f'./logs/{args.dataset}/{args.model}'
    
    # 更新模型配置
    config.MODEL = Dict(model_info)
    config.MODEL.feature_dim = ENCODER_FEATURE_DIMS[encoder_name]
    
    # 更新训练参数
    if args.epochs:
        config.TRAINING.num_epochs = args.epochs
    if args.lr:
        config.TRAINING.learning_rate = args.lr
    if args.weight_decay:
        config.TRAINING.weight_decay = args.weight_decay
    if getattr(args, 'batch_size', None):
        config.DATA.train_dataloader.batch_size = args.batch_size
    
    # 更新种子
    if args.seed:
        config.GENERAL.seed = args.seed
    
    # 设置数据集相关参数
    config.mode = args.mode
    config.expr_name = args.dataset
    config.data_path = dataset_info['path']
    config.slide_val = dataset_info['val_slides']
    config.slide_test = dataset_info['test_slides']
    config.encoder_name = encoder_name
    config.use_augmented = getattr(args, 'use_augmented', True)
    config.expand_augmented = getattr(args, 'expand_augmented', True)
    
    # 设置多GPU参数
    config.devices = devices
    config.strategy = strategy
    config.sync_batchnorm = sync_batchnorm
    
    # 设置时间戳和配置路径
    config.GENERAL.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.config = 'built-in'  # 标记为内置配置
    
    print(f"✅ 配置构建完成:")
    print(f"   - 数据集: {args.dataset} ({dataset_info['path']})")
    print(f"   - 模型: {args.model}")
    print(f"   - 编码器: {encoder_name} (特征维度: {ENCODER_FEATURE_DIMS[encoder_name]})")
    print(f"   - GPU: {devices}个 (策略: {strategy})")
    print(f"   - 训练轮数: {config.TRAINING.num_epochs}")
    print(f"   - 批次大小: {config.DATA.train_dataloader.batch_size}")
    print(f"   - 学习率: {config.TRAINING.learning_rate}")
    
    return config








def main(config):
    print("begin main.py")
    print("config_infomation")
    print(config)
     
    seed = config.GENERAL.seed
    print(f"---seed: {seed}---")
    fix_seed(seed)

    print(f'mode: {config.mode}')
    print(f'expr_name: {config.expr_name}')
    print(f'data_path: {config.data_path}')
    print(f'encoder_name: {config.encoder_name}')

    # initialize dataset
    print(f'intializing dataset...')
    dataset = DataInterface(config)
    print(f'dataset: {dataset}')

    print(f'intializing model...')
    model = ModelInterface(config)
    print(f'model: {model}')

    print(f'intializing logger...')
    logger = load_loggers(config)
    print(f'logger: {logger}')

    print(f'intializing callbacks...')
    callbacks = load_callbacks(config)
    print(f'callbacks: {callbacks}')

    print(f'intializing trainer...')
    
    # Configure multi-GPU training strategy based on user selection
    # DDP (DistributedDataParallel) is the recommended strategy for multi-GPU training
    # as it provides better performance and memory efficiency compared to DataParallel
    strategy_config = config.strategy
    if config.devices > 1 and config.strategy == 'ddp':
        # Configure DDP strategy with advanced optimization parameters
        from pytorch_lightning.strategies import DDPStrategy
        strategy_config = DDPStrategy(
            # find_unused_parameters: Set to False for better performance when all parameters are used
            # Setting to True can help with debugging but may slow down training
            find_unused_parameters=getattr(getattr(config, 'MULTI_GPU', None), 'find_unused_parameters', False),
            # gradient_as_bucket_view: Memory optimization that reduces GPU memory usage
            # by storing gradients as views into buckets rather than separate tensors
            gradient_as_bucket_view=True,
            # static_graph: Set to False to support dynamic computational graphs
            # Required for models with conditional execution paths
            static_graph=False
        )
        print(f"配置DDP策略: find_unused_parameters={getattr(getattr(config, 'MULTI_GPU', None), 'find_unused_parameters', False)}")
    
    # Configure gradient accumulation for handling large effective batch sizes
    # When GPU memory is limited, gradient accumulation allows training with larger
    # effective batch sizes by accumulating gradients over multiple mini-batches
    accumulate_grad_batches = 1
    if hasattr(config, 'MULTI_GPU') and hasattr(config.MULTI_GPU, 'accumulate_grad_batches'):
        accumulate_grad_batches = config.MULTI_GPU.accumulate_grad_batches
    
    # Initialize PyTorch Lightning Trainer with multi-GPU optimizations
    trainer = pl.Trainer(
        # Hardware configuration
        accelerator='gpu',  # Use GPU acceleration
        devices=config.devices,  # Number of GPUs to use
        max_epochs=config.TRAINING.num_epochs,  # Maximum training epochs
        
        # Logging and monitoring
        logger=logger,  # TensorBoard/WandB logger for experiment tracking
        check_val_every_n_epoch=1,  # Validate after every epoch
        callbacks=callbacks,  # Early stopping, checkpointing, etc.
        
        # Training optimizations
        precision='16-mixed',  # Mixed precision training for faster training and reduced memory
        strategy=strategy_config,  # Multi-GPU training strategy (DDP/DP)
        sync_batchnorm=config.sync_batchnorm,  # Synchronize BatchNorm statistics across GPUs
        accumulate_grad_batches=accumulate_grad_batches,  # Gradient accumulation steps
        
        # Progress monitoring and debugging options
        enable_progress_bar=True,  # Show training progress bar
        log_every_n_steps=50,  # Log metrics every N training steps
        gradient_clip_val=getattr(config.TRAINING, 'gradient_clip_val', 1.0),  # Gradient clipping for stability
        
        # Reproducibility settings
        deterministic=False,  # Set to True for fully deterministic training (may impact performance)
    )

    print(f'trainer: {trainer}')

    print(f'training...')
    if config.mode == 'train':
        trainer.fit(model, datamodule=dataset)

    return model

if __name__ == '__main__':
    parser = get_parse()
    args = parser.parse_args()
    
    # 构建配置并运行训练
    config = build_config_from_args(args)

    main(config)
