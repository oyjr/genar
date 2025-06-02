import os
import sys
import argparse
import logging
from datetime import datetime

# 确保导入项目目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import pytorch_lightning as pl

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
    },
    'VAR_ST': {
        'model_name': 'VAR_ST',
        'num_genes': 200,
        # VAR-ST 特定参数来自配置文件
        'spatial_size': 16,
        'vae_ch': 128,
        'vae_embed_dim': 256,
        'vae_num_embeddings': 1024,
        'var_depth': 16,
        'var_embed_dim': 1024,
        'var_num_heads': 16,
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
            'patience': 0,  # 默认禁用，只有命令行指定时才启用
            'factor': 0.5
        },
        'gradient_clip_val': 1.0
    },
    'CALLBACKS': {
        'early_stopping': {
            'monitor': 'val_loss',
            'patience': 10000,  # 默认设置很大值，实际禁用早停
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
        'find_unused_parameters': True,
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
    parser.add_argument('--patience', type=int,
                        help='学习率调度器耐心值 (默认: 禁用, 只有指定时才启用patience机制)')
    
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
    batch_size = getattr(args, 'batch_size', None)
    if batch_size:
        config.DATA.train_dataloader.batch_size = batch_size
    if args.patience is not None:
        # 只有明确指定patience时才启用patience机制
        config.TRAINING.lr_scheduler.patience = args.patience
        # 设置早停的patience（通常设为lr_scheduler patience的2倍）
        if args.patience == 0:
            # 如果明确设为0，禁用早停
            config.CALLBACKS.early_stopping.patience = 10000
        else:
            # 启用早停，设为patience的2倍
            config.CALLBACKS.early_stopping.patience = max(10, args.patience * 2)
    # 如果没有指定patience，保持默认的禁用状态（patience=0和early_stopping=10000）
    
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
    
    # 检查patience状态
    lr_patience = config.TRAINING.lr_scheduler.patience
    early_patience = config.CALLBACKS.early_stopping.patience
    patience_status = "禁用" if lr_patience == 0 else f"启用 (LR调度器: {lr_patience}, 早停: {early_patience})"
    
    print(f"✅ 配置构建完成:")
    print(f"   - 数据集: {args.dataset} ({dataset_info['path']})")
    print(f"   - 模型: {args.model}")
    print(f"   - 编码器: {encoder_name} (特征维度: {ENCODER_FEATURE_DIMS[encoder_name]})")
    print(f"   - GPU: {devices}个 (策略: {strategy})")
    print(f"   - 训练轮数: {config.TRAINING.num_epochs}")
    print(f"   - 批次大小: {config.DATA.train_dataloader.batch_size}")
    print(f"   - 学习率: {config.TRAINING.learning_rate}")
    print(f"   - Patience机制: {patience_status}")
    
    return config


def main(config):
    print("🚀 开始训练...")
    
    # 设置随机种子
    fix_seed(config.GENERAL.seed)

    # 初始化组件
    dataset = DataInterface(config)
    model = ModelInterface(config)
    logger = load_loggers(config)
    callbacks = load_callbacks(config)

    # 配置多GPU策略
    strategy_config = config.strategy
    if config.devices > 1 and config.strategy == 'ddp':
        from pytorch_lightning.strategies import DDPStrategy
        strategy_config = DDPStrategy(
            find_unused_parameters=config.MULTI_GPU.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=False
        )
    
    # 配置梯度累积
    accumulate_grad_batches = getattr(config.MULTI_GPU, 'accumulate_grad_batches', 1)
    
    # 初始化训练器
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=config.devices,
        max_epochs=config.TRAINING.num_epochs,
        logger=logger,
        callbacks=callbacks,
        precision='16-mixed',
        strategy=strategy_config,
        sync_batchnorm=config.sync_batchnorm,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_progress_bar=True,
        log_every_n_steps=50,
        gradient_clip_val=config.TRAINING.gradient_clip_val,
        deterministic=False,
    )

    # 开始训练
    if config.mode == 'train':
        trainer.fit(model, datamodule=dataset)

    return model

if __name__ == '__main__':
    parser = get_parse()
    args = parser.parse_args()
    
    # 构建配置并运行训练
    config = build_config_from_args(args)

    main(config)
