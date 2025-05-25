import os
import sys
import argparse
from datetime import datetime
from glob import glob
from pathlib import Path

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
    load_config,
    load_loggers,
    fix_seed
)


torch.set_float32_matmul_precision('high')


def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='运行模式')
    
    # 新增参数
    parser.add_argument('--expr_name', type=str, required=True, help='数据集名称 (PRAD, her2st, kidney, mouse_brain)')
    parser.add_argument('--data_path', type=str, required=True, help='数据集根目录路径')
    parser.add_argument('--slide_val', type=str, default='', help='验证集slide ID，逗号分隔')
    parser.add_argument('--slide_test', type=str, default='', help='测试集slide ID，逗号分隔')
    parser.add_argument('--encoder_name', type=str, default='uni', choices=['uni', 'conch'], help='编码器类型')
    parser.add_argument('--use_augmented', action='store_true', help='是否使用增强嵌入')
    parser.add_argument('--expand_augmented', action='store_true', help='是否展开3D增强嵌入为7倍训练样本（仅训练模式）')
    parser.add_argument('--aug_strategy', type=str, default='random', 
                        choices=['random', 'mean', 'attention', 'first', 'all'],
                        help='3D增强嵌入处理策略: random(推荐)|mean(取平均)|attention(注意力)|first(原图)|all(保留所有)')
    
    return parser


def validate_args(args):
    """验证命令行参数的有效性"""
    # 检查数据路径是否存在
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"数据路径不存在: {args.data_path}")
    
    # 确保data_path以'/'结尾
    if not args.data_path.endswith('/'):
        args.data_path += '/'
    
    # 检查关键目录是否存在
    st_dir = os.path.join(args.data_path, 'st')
    processed_dir = os.path.join(args.data_path, 'processed_data')
    
    if not os.path.exists(st_dir):
        raise FileNotFoundError(f"ST数据目录不存在: {st_dir}")
    
    if not os.path.exists(processed_dir):
        raise FileNotFoundError(f"处理数据目录不存在: {processed_dir}")
    
    print(f"✅ 数据路径验证通过: {args.data_path}")
    
    return args


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
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=config.TRAINING.num_epochs,
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        precision= '16-mixed',
    )

    print(f'trainer: {trainer}')

    print(f'training...')
    if config.mode == 'train':
        trainer.fit(model, datamodule=dataset)

    return model

if __name__ == '__main__':
    parser = get_parse()
    args = parser.parse_args()
    
    # 验证参数
    args = validate_args(args)

    config = load_config(args.config)
    
    # 更新配置对象
    config.mode = args.mode
    config.expr_name = args.expr_name
    config.data_path = args.data_path
    config.slide_val = args.slide_val
    config.slide_test = args.slide_test
    config.encoder_name = args.encoder_name
    config.use_augmented = args.use_augmented
    config.expand_augmented = args.expand_augmented
    config.aug_strategy = args.aug_strategy
    config.config = args.config
    
    # 根据编码器类型动态设置特征维度
    feature_dim = 1024 if args.encoder_name == 'uni' else 512
    config.MODEL.feature_dim = feature_dim
    print(f"✅ 根据编码器 '{args.encoder_name}' 设置特征维度为: {feature_dim}")
    
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.GENERAL.current_time = current_time

    main(config)
    #test_data_module(config)

    


# def test_data_module(cfg):
#     # 初始化数据模块
#     data_module = DataInterface(cfg)
    
#     # 手动调用 setup 进行测试
#     data_module.setup('fit')
    
#     # 检查数据集是否正确初始化
#     print(f"Train dataset: {data_module.train_dataset}")
#     print(f"Val dataset: {data_module.val_dataset}")
    
#     # 获取数据加载器并尝试迭代
#     train_loader = data_module.train_dataloader()
#     val_loader = data_module.val_dataloader()
#     batch = next(iter(train_loader))
#     print(f"Sample batch shape: {[b.shape if hasattr(b, 'shape') else type(b) for b in batch]}")

#      # 打印训练数据的第一个批次
#     print("\n=== 训练数据批次结构 ===")
#     train_batch = next(iter(train_loader))
#     if isinstance(train_batch, dict):
#         print("训练批次是字典类型")
#         for key, value in train_batch.items():
#             if hasattr(value, 'shape'):
#                 print(f"  - {key}: {value.shape}")
#             else:
#                 print(f"  - {key}: {type(value)}")
#     else:
#         print(f"训练批次类型: {type(train_batch)}")
#         print(f"训练批次形状: {[b.shape if hasattr(b, 'shape') else type(b) for b in train_batch]}")
    
#     # 打印验证数据的第一个批次
#     print("\n=== 验证数据批次结构 ===")
#     val_batch = next(iter(val_loader))
#     if isinstance(val_batch, dict):
#         print("验证批次是字典类型")
#         for key, value in val_batch.items():
#             if hasattr(value, 'shape'):
#                 print(f"  - {key}: {value.shape}")
#             else:
#                 print(f"  - {key}: {type(value)}")
#     else:
#         print(f"验证批次类型: {type(val_batch)}")
#         print(f"验证批次形状: {[b.shape if hasattr(b, 'shape') else type(b) for b in val_batch]}")
