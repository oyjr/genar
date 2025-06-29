#!/usr/bin/env python3
"""
VAR_ST模型推理脚本
用于加载训练好的模型checkpoint并对指定样本进行推理测试
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# 确保导入项目目录下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# 导入项目模块
from dataset.hest_dataset import STDataset
from model import ModelInterface
from model.model_metrics import ModelMetrics
from utils import fix_seed

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 数据集配置
DATASETS = {
    'PRAD': {
        'path': '/data/ouyangjiarui/stem/hest1k_datasets/PRAD/',
        'val_slides': 'MEND144',
        'test_slides': 'MEND144',
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='VAR_ST模型推理脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 对PRAD数据集的MEND144样本进行推理
  python src/inference.py \\
      --ckpt_path logs/PRAD/VAR_ST/best-epoch=epoch=01-val_loss_final=val_loss_final=101.7450.ckpt \\
      --dataset PRAD \\
      --slide_id MEND144 \\
      --output_dir ./inference_results
      
  # 使用GPU进行推理
  python src/inference.py \\
      --ckpt_path your_checkpoint.ckpt \\
      --dataset PRAD \\
      --slide_id MEND144 \\
      --gpu_id 0
        """
    )
    
    # 必需参数
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='模型checkpoint文件路径')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASETS.keys()),
                        help='数据集名称 (PRAD, her2st)')
    parser.add_argument('--slide_id', type=str, required=True,
                        help='要推理的slide ID (如: MEND144)')
    
    # 可选参数
    parser.add_argument('--encoder', type=str, choices=list(ENCODER_FEATURE_DIMS.keys()),
                        help='编码器类型，默认使用数据集推荐编码器')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='结果输出目录 (默认: ./inference_results)')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='使用的GPU ID (默认: 0, -1表示使用CPU)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='推理批次大小 (默认: 64)')
    parser.add_argument('--max_gene_count', type=int, default=500,
                        help='最大基因计数值 (默认: 500)')
    parser.add_argument('--seed', type=int, default=2021,
                        help='随机种子 (默认: 2021)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='是否保存详细的预测结果到文件')
    
    return parser.parse_args()


def setup_device(gpu_id: int):
    """设置计算设备"""
    if gpu_id == -1:
        device = torch.device('cpu')
        logger.info("使用CPU进行推理")
    else:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"使用GPU {gpu_id}进行推理")
        else:
            device = torch.device('cpu')
            logger.warning("CUDA不可用，使用CPU进行推理")
    
    return device


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """从checkpoint加载模型"""
    logger.info(f"从checkpoint加载模型: {ckpt_path}")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {ckpt_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 从checkpoint中提取配置信息
    if 'hyper_parameters' not in checkpoint:
        raise ValueError("Checkpoint中缺少hyper_parameters信息")
    
    config = checkpoint['hyper_parameters']['config']
    logger.info(f"加载的模型配置: {config.MODEL.model_name}")
    
    # 创建模型实例
    model = ModelInterface(config)
    
    # 加载权重
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    logger.info("模型加载完成")
    return model, config


def create_test_dataloader(config, slide_id: str, batch_size: int = 64):
    """创建测试数据加载器"""
    logger.info(f"创建测试数据加载器，slide_id: {slide_id}")
    
    # 基础参数
    base_params = {
        'data_path': config.data_path,
        'expr_name': config.expr_name,
        'slide_val': slide_id,  # 将指定slide作为验证集
        'slide_test': slide_id,  # 将指定slide作为测试集
        'encoder_name': config.encoder_name,
        'use_augmented': False,  # 推理时不使用数据增强
        'max_gene_count': getattr(config, 'max_gene_count', 500),
    }
    
    # 创建测试数据集
    test_dataset = STDataset(mode='test', expand_augmented=False, **base_params)
    
    # 创建DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"测试数据集大小: {len(test_dataset)} 个样本")
    return test_loader, test_dataset


def run_inference(model, test_loader, device: torch.device):
    """运行推理"""
    logger.info("开始推理...")
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_losses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 将数据移到设备上
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(device)
            
            # 预处理输入
            processed_batch = model.model_utils.preprocess_inputs(batch)
            
            # 推理模式：不使用teacher forcing
            inference_batch = processed_batch.copy()
            if 'target_genes' in inference_batch:
                targets = inference_batch.pop('target_genes')
            else:
                targets = batch['target_genes']
            
            # 执行推理
            results = model.model(**inference_batch, top_k=1)  # 使用top-k=1进行确定性推理
            
            # 获取预测结果
            if 'predictions' in results:
                predictions = results['predictions']
            elif 'generated_sequence' in results:
                predictions = results['generated_sequence']
            else:
                raise ValueError("模型输出中找不到预测结果")
            
            # 计算损失（用于监控）
            loss_batch = processed_batch.copy()
            loss_results = model.model(**loss_batch)
            loss = model._compute_loss(loss_results, batch)
            
            # 收集结果
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_losses.append(loss.item())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"已处理 {batch_idx + 1}/{len(test_loader)} 个批次")
    
    # 合并所有结果
    predictions = torch.cat(all_predictions, dim=0)  # [N, 200]
    targets = torch.cat(all_targets, dim=0)  # [N, 200]
    avg_loss = np.mean(all_losses)
    
    logger.info(f"推理完成，总样本数: {predictions.shape[0]}, 平均损失: {avg_loss:.6f}")
    
    return predictions, targets, avg_loss


def calculate_detailed_metrics(predictions: torch.Tensor, targets: torch.Tensor):
    """计算详细的评估指标"""
    logger.info("计算评估指标...")
    
    # 转换为numpy数组
    if torch.is_tensor(predictions):
        predictions = predictions.numpy()
    if torch.is_tensor(targets):
        targets = targets.numpy()
    
    # 创建ModelMetrics实例进行计算
    # 创建一个简单的配置对象
    class SimpleConfig:
        def __init__(self):
            self.MODEL = type('obj', (object,), {'num_genes': 200})()
    
    config = SimpleConfig()
    
    # 创建一个简单的lightning_module模拟对象
    class SimpleLightningModule:
        def log(self, *args, **kwargs):
            pass
    
    lightning_module = SimpleLightningModule()
    
    # 创建ModelMetrics实例
    model_metrics = ModelMetrics(config, lightning_module)
    
    # 计算PCC指标 - 应用log2变换
    pcc_metrics = model_metrics.calculate_comprehensive_pcc_metrics(
        predictions, targets, apply_log2=True
    )
    
    # 计算额外的统计指标
    # 原始数据统计
    pred_stats = {
        'pred_mean': float(np.mean(predictions)),
        'pred_std': float(np.std(predictions)),
        'pred_min': float(np.min(predictions)),
        'pred_max': float(np.max(predictions)),
    }
    
    target_stats = {
        'target_mean': float(np.mean(targets)),
        'target_std': float(np.std(targets)),
        'target_min': float(np.min(targets)),
        'target_max': float(np.max(targets)),
    }
    
    # 基因级别的相关性分析
    gene_correlations = model_metrics.calculate_gene_correlations(targets, predictions)
    
    # 计算每个基因的统计信息
    gene_stats = []
    for i in range(predictions.shape[1]):
        gene_pred = predictions[:, i]
        gene_target = targets[:, i]
        
        gene_stat = {
            'gene_idx': i,
            'correlation': float(gene_correlations[i]),
            'pred_mean': float(np.mean(gene_pred)),
            'target_mean': float(np.mean(gene_target)),
            'pred_std': float(np.std(gene_pred)),
            'target_std': float(np.std(gene_target)),
        }
        gene_stats.append(gene_stat)
    
    # 排序基因相关性
    sorted_gene_stats = sorted(gene_stats, key=lambda x: x['correlation'], reverse=True)
    
    return {
        'pcc_metrics': pcc_metrics,
        'pred_stats': pred_stats,
        'target_stats': target_stats,
        'gene_correlations': gene_correlations,
        'gene_stats': gene_stats,
        'sorted_gene_stats': sorted_gene_stats
    }


def print_results(metrics: dict, avg_loss: float):
    """打印推理结果"""
    pcc_metrics = metrics['pcc_metrics']
    pred_stats = metrics['pred_stats']
    target_stats = metrics['target_stats']
    sorted_gene_stats = metrics['sorted_gene_stats']
    
    print("\n" + "="*60)
    print("🎯 VAR_ST模型推理结果")
    print("="*60)
    
    # 主要指标
    print(f"\n📊 主要评估指标:")
    print(f"   损失 (Loss):      {avg_loss:.6f}")
    print(f"   PCC-10:          {pcc_metrics['pcc_10']:.4f}")
    print(f"   PCC-50:          {pcc_metrics['pcc_50']:.4f}")
    print(f"   PCC-200:         {pcc_metrics['pcc_200']:.4f}")
    print(f"   MSE:             {pcc_metrics['mse']:.6f}")
    print(f"   MAE:             {pcc_metrics['mae']:.6f}")
    print(f"   RVD:             {pcc_metrics['rvd']:.6f}")
    
    # 数据统计
    print(f"\n📈 预测值统计:")
    print(f"   均值:            {pred_stats['pred_mean']:.2f}")
    print(f"   标准差:          {pred_stats['pred_std']:.2f}")
    print(f"   范围:            [{pred_stats['pred_min']:.2f}, {pred_stats['pred_max']:.2f}]")
    
    print(f"\n📈 真实值统计:")
    print(f"   均值:            {target_stats['target_mean']:.2f}")
    print(f"   标准差:          {target_stats['target_std']:.2f}")
    print(f"   范围:            [{target_stats['target_min']:.2f}, {target_stats['target_max']:.2f}]")
    
    # Top表现基因
    print(f"\n🏆 Top-10表现最佳基因:")
    for i, gene_stat in enumerate(sorted_gene_stats[:10]):
        print(f"   {i+1:2d}. 基因{gene_stat['gene_idx']:3d}: PCC={gene_stat['correlation']:.4f}")
    
    # Bottom表现基因
    print(f"\n⚠️  Bottom-5表现最差基因:")
    for i, gene_stat in enumerate(sorted_gene_stats[-5:]):
        rank = len(sorted_gene_stats) - 4 + i
        print(f"   {rank:2d}. 基因{gene_stat['gene_idx']:3d}: PCC={gene_stat['correlation']:.4f}")
    
    print("\n" + "="*60)


def save_results(metrics: dict, predictions: torch.Tensor, targets: torch.Tensor, 
                avg_loss: float, output_dir: str, slide_id: str, save_predictions: bool = False):
    """保存推理结果"""
    logger.info(f"保存结果到: {output_dir}")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 保存主要指标
    results_summary = {
        'slide_id': slide_id,
        'timestamp': datetime.now().isoformat(),
        'num_samples': predictions.shape[0],
        'num_genes': predictions.shape[1],
        'avg_loss': avg_loss,
        **metrics['pcc_metrics'],
        **metrics['pred_stats'],
        **metrics['target_stats']
    }
    
    summary_file = os.path.join(output_dir, f'{slide_id}_inference_summary.json')
    import json
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # 保存基因级别统计
    gene_stats_df = pd.DataFrame(metrics['gene_stats'])
    gene_stats_file = os.path.join(output_dir, f'{slide_id}_gene_statistics.csv')
    gene_stats_df.to_csv(gene_stats_file, index=False)
    
    # 可选：保存详细预测结果
    if save_predictions:
        predictions_file = os.path.join(output_dir, f'{slide_id}_predictions.npz')
        np.savez_compressed(
            predictions_file,
            predictions=predictions.numpy() if torch.is_tensor(predictions) else predictions,
            targets=targets.numpy() if torch.is_tensor(targets) else targets
        )
        logger.info(f"详细预测结果已保存到: {predictions_file}")
    
    logger.info(f"结果摘要已保存到: {summary_file}")
    logger.info(f"基因统计已保存到: {gene_stats_file}")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    fix_seed(args.seed)
    
    # 设置设备
    device = setup_device(args.gpu_id)
    
    # 检查checkpoint文件
    if not os.path.exists(args.ckpt_path):
        logger.error(f"Checkpoint文件不存在: {args.ckpt_path}")
        return
    
    # 检查数据集配置
    if args.dataset not in DATASETS:
        logger.error(f"不支持的数据集: {args.dataset}")
        return
    
    dataset_info = DATASETS[args.dataset]
    
    # 确定编码器
    encoder_name = args.encoder or dataset_info['recommended_encoder']
    
    logger.info(f"推理配置:")
    logger.info(f"  数据集: {args.dataset}")
    logger.info(f"  Slide ID: {args.slide_id}")
    logger.info(f"  编码器: {encoder_name}")
    logger.info(f"  Checkpoint: {args.ckpt_path}")
    logger.info(f"  输出目录: {args.output_dir}")
    
    try:
        # 加载模型
        model, config = load_model_from_checkpoint(args.ckpt_path, device)
        
        # 更新配置中的数据集信息
        config.data_path = dataset_info['path']
        config.expr_name = args.dataset
        config.encoder_name = encoder_name
        config.max_gene_count = args.max_gene_count
        
        # 创建测试数据加载器
        test_loader, test_dataset = create_test_dataloader(config, args.slide_id, args.batch_size)
        
        # 运行推理
        predictions, targets, avg_loss = run_inference(model, test_loader, device)
        
        # 计算评估指标
        metrics = calculate_detailed_metrics(predictions, targets)
        
        # 打印结果
        print_results(metrics, avg_loss)
        
        # 保存结果
        save_results(metrics, predictions, targets, avg_loss, 
                    args.output_dir, args.slide_id, args.save_predictions)
        
        logger.info("推理完成！")
        
    except Exception as e:
        logger.error(f"推理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 