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
from dataset.hest_dataset import STDataset
from model import ModelInterface
from utils import (
    load_callbacks,
    load_loggers,
    fix_seed
)
from torch.utils.data import DataLoader

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
        'val_slides': 'SPA148',
        'test_slides': 'SPA148', 
        'recommended_encoder': 'conch'
    }
}

# Multi-Scale Gene VAR 模型配置
VAR_ST_CONFIG = {
        'model_name': 'VAR_ST',
        'num_genes': 200,
        'histology_feature_dim': 1024,  # 依赖编码器
        'spatial_coord_dim': 2,
        
        # Multi-Scale VAR 配置 (内存优化版本)
        'gene_patch_nums': (1, 2, 4, 6, 8, 10, 15),  # 7个尺度，最后一个改为14减少序列长度
        # vocab_size 将根据 max_gene_count 动态计算 (max_gene_count + 1)
        'embed_dim': 512,  # 减少嵌入维度 768->512
        'num_heads': 8,    # 减少注意力头数 12->8
        'num_layers': 8,   # 减少层数 12->8
        'mlp_ratio': 3.0,  # 减少MLP倍数 4.0->3.0
        
        # Dropout 参数
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
        
        # 条件相关参数
        'condition_embed_dim': 512,  # 匹配embed_dim
        'cond_drop_rate': 0.1,
        
        # 其他参数
        'norm_eps': 1e-6,
        'shared_aln': False,
        'attn_l2_norm': True
}

# 默认训练配置 
DEFAULT_CONFIG = {
    'GENERAL': {
        'seed': 2021,
        'log_path': './logs', 
        'debug': False
    },
    'DATA': {
        'normalize': True,  # 保留参数兼容性，实际使用原始基因计数
        'train_dataloader': {
            'batch_size': 256,
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': True,
            'persistent_workers': True
        },
        'val_dataloader': {
            'batch_size': 64,  # 🔧 进一步增加验证批次大小到64，显著加速验证
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': False,
            'persistent_workers': True
        },
        'test_dataloader': {
            'batch_size': 64,  # 🔧 同步增加测试批次大小到64
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
            'monitor': 'val_loss',  # 动态更新：Stage1用val_mse, Stage2用val_accuracy
            'patience': 10000,  # 默认设置很大值，实际禁用早停
            'mode': 'min',      # 动态更新：Stage1用min, Stage2用max
            'min_delta': 0.0
        },
        'model_checkpoint': {
            'monitor': 'val_loss',  # 动态更新：Stage1用val_mse, Stage2用val_accuracy  
            'save_top_k': 1,
            'mode': 'min',          # 动态更新：Stage1用min, Stage2用max
            'filename': 'best-epoch={epoch:02d}-{val_mse:.4f}'  # 动态更新：Stage1和Stage2使用不同命名
        },
        'learning_rate_monitor': {
            'logging_interval': 'epoch'
        }
    },
    'MULTI_GPU': {
        'find_unused_parameters': True,  # 🔧 启用未使用参数检测：VAR模型可能有未使用参数
        'accumulate_grad_batches': 1
    }
}


def get_parse():
    """
    Parse command line arguments for VAR_ST training.
    
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
  python src/main.py --dataset PRAD --encoder uni \\
      --gpus 4 --epochs 200 --batch_size 256 --lr 1e-4
  
  # Single GPU training
  python src/main.py --dataset her2st --gpus 1
  
  # Test mode with checkpoint
  python src/main.py --dataset her2st --mode test --gpus 1 \\
      --ckpt_path logs/her2st/VAR_ST/best-epoch=epoch=02-pcc50=val_pcc_50=0.7688.ckpt
        """
    )
    
    # === 核心参数 ===
    parser.add_argument('--dataset', type=str, choices=list(DATASETS.keys()),
                        help='数据集名称 (PRAD, her2st)')
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
    
    # === 🆕 基因计数参数 ===
    parser.add_argument('--max-gene-count', type=int, default=500,
                        help='最大基因计数值 (默认: 500)')
    
    # === 其他参数 ===
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='运行模式 (默认: train)')
    parser.add_argument('--seed', type=int,
                        help='随机种子 (默认: 2021)')
    
    # === 🆕 测试模式参数 ===
    parser.add_argument('--ckpt_path', type=str,
                        help='测试模式时使用的checkpoint路径 (必须在--mode test时指定)')
    
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
    
    # 🆕 检查测试模式参数
    if args.mode == 'test' and not args.ckpt_path:
        raise ValueError("测试模式必须指定 --ckpt_path 参数")
    
    if args.ckpt_path and not os.path.exists(args.ckpt_path):
        raise ValueError(f"Checkpoint文件不存在: {args.ckpt_path}")
    
    print(f"🚀 使用简化配置模式: 数据集={args.dataset}, 模型=VAR_ST, 模式={args.mode}")
    
    # 获取数据集信息
    dataset_info = DATASETS[args.dataset]
    
    # 获取模型信息
    model_info = VAR_ST_CONFIG
    
    # 确定编码器
    encoder_name = args.encoder or dataset_info['recommended_encoder']
    
    # 确定GPU相关参数
    devices = args.gpus
    strategy = 'ddp' if devices > 1 and args.strategy == 'auto' else args.strategy
    sync_batchnorm = getattr(args, 'sync_batchnorm', False) or (devices > 1)
    
    # 构建完整配置
    config = Dict(DEFAULT_CONFIG)
    
    # 更新日志路径为数据集名称和模型名称
    config.GENERAL.log_path = f'./logs/{args.dataset}/VAR_ST'
    
    # 更新模型配置
    config.MODEL = Dict(model_info)
    config.MODEL.feature_dim = ENCODER_FEATURE_DIMS[encoder_name]
    # 🔧 根据命令行参数更新基因数量
    max_gene_count = getattr(args, 'max_gene_count', 500)
    # num_genes保持200不变，不被max_gene_count影响
    
    # 🆕 动态计算vocab_size = max_gene_count + 1 (对应0到max_gene_count的计数范围)
    vocab_size = max_gene_count + 1
    config.MODEL.vocab_size = vocab_size
    config.MODEL.max_gene_count = max_gene_count
    
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
    config.gene_count_mode = 'discrete_tokens'  # 固定为离散token模式
    config.max_gene_count = getattr(args, 'max_gene_count', 500)
    
    # 🆕 设置checkpoint路径
    if args.ckpt_path:
        config.ckpt_path = args.ckpt_path
    
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
    
    # VAR-ST模型配置
    config.MODEL.histology_feature_dim = ENCODER_FEATURE_DIMS[encoder_name]
    config.MODEL.gene_count_mode = config.gene_count_mode
    config.MODEL.max_gene_count = config.max_gene_count
    # 🔧 暂时使用val_loss作为监控指标，避免第一个epoch的EarlyStopping错误
    # TODO: 后续可以改回val_pcc_50，但需要确保第一个epoch验证完成后才检查
    config.TRAINING.monitor = 'val_loss'
    config.TRAINING.mode = 'min'
    config.CALLBACKS.early_stopping.monitor = 'val_loss'
    config.CALLBACKS.early_stopping.mode = 'min'
    config.CALLBACKS.model_checkpoint.monitor = 'val_loss'
    config.CALLBACKS.model_checkpoint.mode = 'min'
    config.CALLBACKS.model_checkpoint.filename = 'best-epoch={epoch:02d}-loss={val_loss:.6f}'
    print(f"   - VAR-ST监控指标: val_loss (最小化) - 临时使用，避免第一个epoch错误")
    print(f"   - Checkpoint文件名模板: best-epoch={{epoch:02d}}-loss={{val_loss:.6f}}")
    print(f"   - 基因计数模式: discrete_tokens (保持原始计数)")
    print(f"   - 最大基因计数: {config.max_gene_count}")
    print(f"   - 词汇表大小: {vocab_size} (动态计算: {max_gene_count} + 1)")
    
    print(f"✅ 配置构建完成:")
    print(f"   - 数据集: {args.dataset} ({dataset_info['path']})")
    print(f"   - 模型: VAR_ST")
    print(f"   - 编码器: {encoder_name} (特征维度: {ENCODER_FEATURE_DIMS[encoder_name]})")
    print(f"   - GPU: {devices}个 (策略: {strategy})")
    print(f"   - 训练轮数: {config.TRAINING.num_epochs}")
    print(f"   - 批次大小: {config.DATA.train_dataloader.batch_size}")
    print(f"   - 学习率: {config.TRAINING.learning_rate}")
    print(f"   - Patience机制: {patience_status}")
    print(f"   - 基因计数范围: 0-{max_gene_count} (vocab_size: {vocab_size})")
    
    return config


def create_dataloaders(config):
    """创建数据加载器"""
    # 基础参数
    base_params = {
        'data_path': config.data_path,
        'expr_name': config.expr_name,
        'slide_val': config.slide_val,
        'slide_test': config.slide_test,
        'encoder_name': config.encoder_name,
        'use_augmented': config.use_augmented,
        'max_gene_count': getattr(config, 'max_gene_count', 500),
    }
    
    # 创建数据集
    train_dataset = STDataset(mode='train', expand_augmented=config.expand_augmented, **base_params)
    val_dataset = STDataset(mode='val', expand_augmented=False, **base_params)
    test_dataset = STDataset(mode='test', expand_augmented=False, **base_params)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.DATA.train_dataloader.batch_size,
        shuffle=config.DATA.train_dataloader.shuffle,
        num_workers=config.DATA.train_dataloader.num_workers,
        pin_memory=config.DATA.train_dataloader.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.DATA.val_dataloader.batch_size,
        shuffle=config.DATA.val_dataloader.shuffle,
        num_workers=config.DATA.val_dataloader.num_workers,
        pin_memory=config.DATA.val_dataloader.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.DATA.val_dataloader.batch_size,
        shuffle=False,
        num_workers=config.DATA.val_dataloader.num_workers,
        pin_memory=config.DATA.val_dataloader.pin_memory
    )
    
    return train_loader, val_loader, test_loader


def main(config):
    if config.mode == 'train':
        print("🚀 开始训练...")
    else:
        print("🧪 开始测试...")
    
    # 设置随机种子
    fix_seed(config.GENERAL.seed)

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    # 初始化组件
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
        precision=32,
        strategy=strategy_config,
        sync_batchnorm=config.sync_batchnorm,
        accumulate_grad_batches=accumulate_grad_batches,
        enable_progress_bar=True,
        log_every_n_steps=50,
        gradient_clip_val=config.TRAINING.gradient_clip_val,
        deterministic=False,
    )

    # 根据模式执行不同的操作
    if config.mode == 'train':
        trainer.fit(model, train_loader, val_loader)
    elif config.mode == 'test':
        print(f"📂 从checkpoint加载模型: {config.ckpt_path}")
        trainer.test(model, test_loader, ckpt_path=config.ckpt_path)
        print("✅ 测试完成！")

    return model

if __name__ == '__main__':
    parser = get_parse()
    args = parser.parse_args()
    
    # 构建配置并运行训练
    config = build_config_from_args(args)

    main(config)
