#!/usr/bin/env python3
"""
VAR_ST模型推理脚本
用于加载训练好的checkpoint并在测试集上进行评估
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import pytorch_lightning as pl
from typing import Dict, List, Tuple
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入项目模块
from dataset.data_interface import DataInterface
from model import ModelInterface

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 最小方差阈值
MIN_VARIANCE_THRESHOLD = 1e-8

# 数据集配置
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

# 编码器特征维度映射
ENCODER_FEATURE_DIMS = {
    'uni': 1024,
    'conch': 512
}


def calculate_gene_correlations(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """计算基因级别的相关系数"""
    num_genes = y_true.shape[1]
    correlations = np.zeros(num_genes)
    
    for i in range(num_genes):
        true_gene = y_true[:, i]
        pred_gene = y_pred[:, i]
        
        # 处理常数值
        if np.std(true_gene) == 0 or np.std(pred_gene) == 0:
            correlations[i] = 0.0
        else:
            corr = np.corrcoef(true_gene, pred_gene)[0, 1]
            correlations[i] = 0.0 if np.isnan(corr) else corr
    
    return correlations


def calculate_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算综合评估指标"""
    # 确保输入是numpy数组
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # 🔧 关键修复：应用log2(x+1)变换用于评估指标计算
    # 这与训练时的评估保持一致
    logger.info("📊 应用log2(x+1)变换用于指标计算...")
    y_true_log2 = np.log2(y_true + 1.0)
    y_pred_log2 = np.log2(y_pred + 1.0)
    
    # 检查NaN值
    if np.isnan(y_true_log2).any() or np.isnan(y_pred_log2).any():
        logger.warning("⚠️ Log2变换后发现NaN值，将使用原始值")
        y_true_log2 = y_true
        y_pred_log2 = y_pred
    
    # 计算基因相关性（使用log2变换后的值）
    correlations = calculate_gene_correlations(y_true_log2, y_pred_log2)
    
    # 排序相关性
    sorted_corr = np.sort(correlations)[::-1]
    
    # 计算PCC指标
    pcc_10 = np.mean(sorted_corr[:10]) if len(sorted_corr) >= 10 else np.mean(sorted_corr)
    pcc_50 = np.mean(sorted_corr[:50]) if len(sorted_corr) >= 50 else np.mean(sorted_corr)
    pcc_200 = np.mean(sorted_corr[:200]) if len(sorted_corr) >= 200 else np.mean(sorted_corr)
    
    # 计算MSE和MAE（使用log2变换后的值）
    mse = np.mean((y_true_log2 - y_pred_log2) ** 2)
    mae = np.mean(np.abs(y_true_log2 - y_pred_log2))
    
    # 计算RVD (Relative Variance Difference)（使用log2变换后的值）
    pred_var = np.var(y_pred_log2, axis=0)
    true_var = np.var(y_true_log2, axis=0)
    
    valid_mask = true_var > MIN_VARIANCE_THRESHOLD
    if np.sum(valid_mask) > 0:
        rvd = np.mean(((pred_var[valid_mask] - true_var[valid_mask]) ** 2) / (true_var[valid_mask] ** 2))
    else:
        rvd = 0.0
    
    return {
        'PCC-10': float(pcc_10),
        'PCC-50': float(pcc_50), 
        'PCC-200': float(pcc_200),
        'MSE': float(mse),
        'MAE': float(mae),
        'RVD': float(rvd),
        'correlations': correlations
    }


def load_model_from_checkpoint(checkpoint_path: str, dataset_name: str, encoder_name: str = None) -> ModelInterface:
    """从checkpoint加载模型 - 严格验证配置一致性"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    
    # 获取数据集信息
    dataset_info = DATASETS[dataset_name]
    encoder_name = encoder_name or dataset_info['recommended_encoder']
    
    logger.info(f"🔄 加载模型从: {checkpoint_path}")
    logger.info(f"📊 数据集: {dataset_name}")
    logger.info(f"🔧 编码器: {encoder_name}")
    
    # 从checkpoint加载模型 - 严格模式，不允许参数不匹配
    model = ModelInterface.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=True  # 🔧 严格模式：配置必须完全匹配
    )
    
    logger.info("✅ 模型加载成功")
    return model


def create_config_for_inference(dataset_name: str, encoder_name: str = None):
    """为推理创建配置"""
    from addict import Dict
    
    # 获取数据集信息
    dataset_info = DATASETS[dataset_name]
    encoder_name = encoder_name or dataset_info['recommended_encoder']
    
    # 创建基础配置
    config = Dict({
        'GENERAL': {
            'seed': 2021,
            'log_path': f'./logs/{dataset_name}/VAR_ST',
            'debug': False
        },
        'DATA': {
            'normalize': True,
            'test_dataloader': {
                'batch_size': 128,  # 单样本推理以最大化节省内存
                'num_workers': 1,  # 最少worker数量
                'pin_memory': False,  # 关闭pin_memory节省内存
                'shuffle': False,
                'persistent_workers': False
            }
        },
        'MODEL': {
            'model_name': 'VAR_ST',
            # 🔧 关键修复：删除所有硬编码的模型配置
            # 模型配置将从checkpoint中自动读取，确保训练和推理完全一致
            'gene_count_mode': 'discrete_tokens',
            'max_gene_count': 200
        }
    })
    
    # 设置数据集相关参数
    config.mode = 'test'
    config.expr_name = dataset_name
    config.data_path = dataset_info['path']
    config.slide_val = dataset_info['val_slides']
    config.slide_test = dataset_info['test_slides']
    config.encoder_name = encoder_name
    config.use_augmented = True
    config.expand_augmented = True
    config.gene_count_mode = 'discrete_tokens'
    config.max_gene_count = 200
    
    return config


def run_inference(model: ModelInterface, dataloader, args, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray]:
    """运行推理并收集预测结果"""
    
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_targets = []
    
    logger.info(f"🔮 开始推理，共 {len(dataloader)} 个批次")
    logger.info(f"💾 使用批次大小: {dataloader.batch_size}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="推理中")):
            try:
                # 每个批次前都清理GPU缓存
                torch.cuda.empty_cache()
                
                # 🔧 关键修复：先将整个batch移动到设备，然后再预处理
                # 确保所有数据都在同一设备上
                batch_on_device = {}
                for key, value in batch.items():
                    if torch.is_tensor(value):
                        batch_on_device[key] = value.to(device)
                    else:
                        batch_on_device[key] = value
                
                # 使用ModelInterface的预处理逻辑
                processed_batch = model._preprocess_inputs(batch_on_device)
                
                # 严格验证预处理结果
                required_keys = ['histology_features', 'spatial_coords']
                for key in required_keys:
                    if key not in processed_batch:
                        raise ValueError(f"预处理后缺少必需的键: {key}")
                
                # 确保预处理后的数据也在正确设备上
                for key in required_keys:
                    if torch.is_tensor(processed_batch[key]):
                        processed_batch[key] = processed_batch[key].to(device)
                
                # 调用底层模型的inference方法，支持采样参数
                outputs = model.model.inference(
                    histology_features=processed_batch['histology_features'],
                    spatial_coords=processed_batch['spatial_coords'],
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    seed=args.seed
                )
                
                # 严格验证输出格式
                if not isinstance(outputs, dict):
                    raise ValueError(f"模型输出必须是字典格式，实际得到: {type(outputs)}")
                
                if 'predictions' not in outputs:
                    raise ValueError(f"模型输出中缺少'predictions'键，可用键: {list(outputs.keys())}")
                
                predictions = outputs['predictions']
                gene_expression = batch_on_device['target_genes']
                
                # 立即移动到CPU并收集结果
                all_predictions.append(predictions.cpu())
                all_targets.append(gene_expression.cpu())
                
                # 删除GPU上的临时变量
                del batch_on_device, processed_batch, predictions, gene_expression
                if 'outputs' in locals():
                    del outputs
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"❌ 批次 {batch_idx} GPU内存不足: {e}")
                logger.info("🔄 清理GPU缓存并跳过此批次...")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                logger.error(f"❌ 批次 {batch_idx} 推理失败: {e}")
                continue
    
    # 合并所有批次结果
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    
    logger.info(f"✅ 推理完成")
    logger.info(f"📊 预测形状: {all_predictions.shape}")
    logger.info(f"📊 目标形状: {all_targets.shape}")
    
    return all_targets, all_predictions


def main():
    parser = argparse.ArgumentParser(description='VAR_ST模型推理脚本')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, choices=list(DATASETS.keys()), required=True,
                        help='数据集名称')
    parser.add_argument('--encoder', type=str, choices=list(ENCODER_FEATURE_DIMS.keys()),
                        help='编码器类型，默认使用数据集推荐编码器')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备 (默认: cuda)')
    parser.add_argument('--output', type=str, default='inference_results.txt',
                        help='结果输出文件 (默认: inference_results.txt)')
    
    # 🔧 新增：采样参数控制
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='采样温度 (默认: 1.0)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k采样参数 (默认: None)')
    parser.add_argument('--top_p', type=float, default=None,
                        help='Top-p采样参数 (默认: None)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子，用于可重现结果 (默认: None)')
    
    args = parser.parse_args()
    
    # 检查checkpoint文件
    if not os.path.exists(args.checkpoint):
        logger.error(f"❌ Checkpoint文件不存在: {args.checkpoint}")
        return
    
    # 创建推理配置
    config = create_config_for_inference(args.dataset, args.encoder)
    
    # 加载模型
    try:
        model = load_model_from_checkpoint(args.checkpoint, args.dataset, args.encoder)
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        return
    
    # 创建数据加载器
    logger.info("📂 准备测试数据...")
    dataset = DataInterface(config)
    dataset.setup('test')
    test_dataloader = dataset.test_dataloader()
    
    logger.info(f"📊 测试数据集大小: {len(test_dataloader.dataset)}")
    
    # 🔧 新增：输出采样参数配置
    logger.info("🎯 采样参数配置:")
    logger.info(f"  - Temperature: {args.temperature}")
    logger.info(f"  - Top-k: {args.top_k}")
    logger.info(f"  - Top-p: {args.top_p}")
    logger.info(f"  - Seed: {args.seed}")
    
    # 运行推理
    try:
        y_true, y_pred = run_inference(model, test_dataloader, args, args.device)
    except Exception as e:
        logger.error(f"❌ 推理失败: {e}")
        return
    
    # 计算评估指标
    logger.info("📈 计算评估指标...")
    metrics = calculate_evaluation_metrics(y_true, y_pred)
    
    # 打印结果
    print("\n" + "="*60)
    print("🎯 VAR_ST模型推理结果")
    print("="*60)
    print(f"📁 Checkpoint: {args.checkpoint}")
    print(f"📊 数据集: {args.dataset}")
    print(f"🔧 编码器: {config.encoder_name}")
    print(f"📏 测试样本数: {y_true.shape[0]}")
    print(f"🧬 基因数量: {y_true.shape[1]}")
    print("-"*60)
    print("📈 评估指标:")
    print(f"   PCC-10:  {metrics['PCC-10']:.4f}")
    print(f"   PCC-50:  {metrics['PCC-50']:.4f}")
    print(f"   PCC-200: {metrics['PCC-200']:.4f}")
    print(f"   MSE:     {metrics['MSE']:.6f}")
    print(f"   MAE:     {metrics['MAE']:.6f}")
    print(f"   RVD:     {metrics['RVD']:.6f}")
    print("="*60)
    
    # 保存结果到文件
    with open(args.output, 'w') as f:
        f.write("VAR_ST模型推理结果\n")
        f.write("="*60 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"数据集: {args.dataset}\n")
        f.write(f"编码器: {config.encoder_name}\n")
        f.write(f"测试样本数: {y_true.shape[0]}\n")
        f.write(f"基因数量: {y_true.shape[1]}\n")
        f.write("-"*60 + "\n")
        f.write("评估指标:\n")
        f.write(f"PCC-10:  {metrics['PCC-10']:.4f}\n")
        f.write(f"PCC-50:  {metrics['PCC-50']:.4f}\n")
        f.write(f"PCC-200: {metrics['PCC-200']:.4f}\n")
        f.write(f"MSE:     {metrics['MSE']:.6f}\n")
        f.write(f"MAE:     {metrics['MAE']:.6f}\n")
        f.write(f"RVD:     {metrics['RVD']:.6f}\n")
        
        # 保存基因级别的相关性
        f.write("\n基因级别相关性统计:\n")
        correlations = metrics['correlations']
        f.write(f"平均相关性: {np.mean(correlations):.4f}\n")
        f.write(f"中位数相关性: {np.median(correlations):.4f}\n")
        f.write(f"标准差: {np.std(correlations):.4f}\n")
        f.write(f"最大相关性: {np.max(correlations):.4f}\n")
        f.write(f"最小相关性: {np.min(correlations):.4f}\n")
        
        # 相关性分布
        high_corr = np.sum(correlations > 0.5)
        medium_corr = np.sum((correlations > 0.3) & (correlations <= 0.5))
        low_corr = np.sum(correlations <= 0.3)
        
        f.write(f"\n相关性分布:\n")
        f.write(f"高相关性 (>0.5): {high_corr} 个基因 ({high_corr/len(correlations)*100:.1f}%)\n")
        f.write(f"中等相关性 (0.3-0.5): {medium_corr} 个基因 ({medium_corr/len(correlations)*100:.1f}%)\n")
        f.write(f"低相关性 (≤0.3): {low_corr} 个基因 ({low_corr/len(correlations)*100:.1f}%)\n")
    
    logger.info(f"💾 结果已保存到: {args.output}")


if __name__ == '__main__':
    main() 