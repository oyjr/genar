#!/usr/bin/env python3
"""
Two-Stage VAR-ST 完整推理和评估脚本

此脚本实现：
1. 加载Stage 1 (VQVAE) 和 Stage 2 (VAR) 检查点
2. 构建完整的两阶段模型
3. 进行端到端的基因表达预测推理
4. 计算完整的评估指标 (PCC, MSE, MAE, RVD等)
5. 保存推理结果和评估报告

使用方法：
python two_stage_complete_inference.py \
    --stage1_ckpt logs/PRAD/TWO_STAGE_VAR_ST/stage1-best-epoch=epoch=143-val_mse=val_mse=0.5353.ckpt \
    --stage2_ckpt logs/PRAD/TWO_STAGE_VAR_ST/stage2-best-epoch=epoch=03-val_acc=val_accuracy=0.8263.ckpt \
    --dataset PRAD \
    --mode test \
    --save_results
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import numpy as np
import argparse
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import json

from model.VAR.two_stage_var_st import TwoStageVARST
from model.model_interface import ModelInterface
from dataset.data_interface import DataInterface
from main import DATASETS, ENCODER_FEATURE_DIMS
from addict import Dict as AddictDict


class TwoStageCompleteInference:
    """两阶段VAR-ST完整推理类"""
    
    def __init__(
        self, 
        stage1_ckpt_path: str,
        stage2_ckpt_path: str,
        device: str = 'cuda'
    ):
        """
        初始化两阶段推理器
        
        Args:
            stage1_ckpt_path: Stage 1 VQVAE检查点路径
            stage2_ckpt_path: Stage 2 VAR检查点路径  
            device: 计算设备
        """
        self.stage1_ckpt_path = stage1_ckpt_path
        self.stage2_ckpt_path = stage2_ckpt_path
        self.device = device
        self.model = None
        self.config = None
        
        print(f"🚀 初始化两阶段VAR-ST推理器")
        print(f"   - Stage 1 VQVAE: {stage1_ckpt_path}")
        print(f"   - Stage 2 VAR: {stage2_ckpt_path}")
        print(f"   - 设备: {device}")
    
    def load_model(self) -> TwoStageVARST:
        """加载完整的两阶段模型"""
        print(f"\n🔄 加载两阶段模型...")
        
        # 1. 从Stage 2检查点获取配置信息
        print(f"   步骤1: 从Stage 2检查点获取配置...")
        stage2_checkpoint = torch.load(self.stage2_ckpt_path, map_location='cpu')
        
        # 从hyperparameters获取配置
        if 'hyper_parameters' in stage2_checkpoint and 'config' in stage2_checkpoint['hyper_parameters']:
            self.config = stage2_checkpoint['hyper_parameters']['config']
            model_config = self.config.get('MODEL', {})
        else:
            # 使用默认配置
            print("   ⚠️  未找到完整配置，使用默认参数")
            model_config = {
                'num_genes': 200,
                'histology_feature_dim': 1024,
                'spatial_coord_dim': 2
            }
        
        # 2. 创建模型实例
        print(f"   步骤2: 创建模型实例...")
        self.model = TwoStageVARST(
            num_genes=model_config.get('num_genes', 200),
            histology_feature_dim=model_config.get('histology_feature_dim', 1024),
            spatial_coord_dim=model_config.get('spatial_coord_dim', 2),
            current_stage=1,  # 先设置为Stage 1
            device=self.device
        )
        
        # 3. 加载Stage 1权重
        print(f"   步骤3: 加载Stage 1 VQVAE权重...")
        stage1_checkpoint = torch.load(self.stage1_ckpt_path, map_location='cpu')
        
        # 提取Stage 1的模型权重
        stage1_state_dict = {}
        for key, value in stage1_checkpoint['state_dict'].items():
            if key.startswith('model.'):
                # 移除Lightning的前缀
                new_key = key[6:]  # 去掉'model.'
                stage1_state_dict[new_key] = value
        
        # 只加载Stage 1 VQVAE的权重
        stage1_vqvae_state_dict = {}
        for key, value in stage1_state_dict.items():
            if key.startswith('stage1_vqvae.'):
                new_key = key[13:]  # 去掉'stage1_vqvae.'
                stage1_vqvae_state_dict[new_key] = value
        
        self.model.stage1_vqvae.load_state_dict(stage1_vqvae_state_dict, strict=True)
        print(f"     ✅ Stage 1 VQVAE权重加载完成")
        
        # 4. 切换到Stage 2并加载权重
        print(f"   步骤4: 切换到Stage 2并加载VAR权重...")
        self.model.current_stage = 2
        self.model._set_vqvae_trainable(False)  # 冻结VQVAE
        
        # 加载Stage 2权重
        stage2_state_dict = {}
        for key, value in stage2_checkpoint['state_dict'].items():
            if key.startswith('model.'):
                new_key = key[6:]  # 去掉'model.'
                stage2_state_dict[new_key] = value
        
        # 加载Stage 2 VAR的权重
        stage2_var_state_dict = {}
        for key, value in stage2_state_dict.items():
            if key.startswith('stage2_var.'):
                new_key = key[11:]  # 去掉'stage2_var.'
                stage2_var_state_dict[new_key] = value
        
        self.model.stage2_var.load_state_dict(stage2_var_state_dict, strict=True)
        
        # 加载条件处理器权重
        condition_processor_state_dict = {}
        for key, value in stage2_state_dict.items():
            if key.startswith('condition_processor.'):
                new_key = key[20:]  # 去掉'condition_processor.'
                condition_processor_state_dict[new_key] = value
        
        if condition_processor_state_dict:
            self.model.condition_processor.load_state_dict(condition_processor_state_dict, strict=True)
            print(f"     ✅ 条件处理器权重加载完成")
        
        print(f"     ✅ Stage 2 VAR权重加载完成")
        
        # 5. 设置为推理模式
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ 两阶段模型加载完成！")
        print(f"   - 模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   - 可训练参数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        return self.model
    
    def predict_batch(
        self, 
        histology_features: torch.Tensor,
        spatial_coords: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9
    ) -> Dict[str, torch.Tensor]:
        """
        对一个批次进行预测
        
        Args:
            histology_features: [B, 1024] 组织学特征
            spatial_coords: [B, 2] 空间坐标
            temperature: 采样温度
            top_k: Top-k采样
            top_p: Nucleus采样
            
        Returns:
            预测结果字典
        """
        with torch.no_grad():
            results = self.model.inference(
                histology_features=histology_features,
                spatial_coords=spatial_coords,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        return results
    
    def evaluate_on_dataloader(
        self,
        dataloader,
        max_batches: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9
    ) -> Dict[str, float]:
        """
        在数据加载器上进行评估
        
        Args:
            dataloader: 数据加载器
            max_batches: 最大批次数（用于快速测试）
            temperature: 采样温度
            top_k: Top-k采样
            top_p: Nucleus采样
            
        Returns:
            评估指标字典
        """
        print(f"\n🧬 开始模型评估...")
        print(f"   - 数据批次数: {len(dataloader) if max_batches is None else min(max_batches, len(dataloader))}")
        print(f"   - 采样参数: temp={temperature}, top_k={top_k}, top_p={top_p}")
        
        all_predictions = []
        all_targets = []
        
        self.model.eval()
        
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            # 移动数据到设备
            histology_features = batch['img'].to(self.device)      # [B, 1024]
            spatial_coords = batch['positions'].to(self.device)   # [B, 2]
            target_genes = batch['target_genes'].to(self.device)  # [B, 200]
            
            # 进行预测
            results = self.predict_batch(
                histology_features=histology_features,
                spatial_coords=spatial_coords,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
            
            # 收集结果
            predictions = results['predicted_gene_expression']  # [B, 200]
            all_predictions.append(predictions.cpu())
            all_targets.append(target_genes.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   已处理 {batch_idx + 1} 个批次...")
        
        # 合并所有结果
        all_predictions = torch.cat(all_predictions, dim=0)  # [N, 200]
        all_targets = torch.cat(all_targets, dim=0)          # [N, 200]
        
        print(f"   总样本数: {all_predictions.shape[0]}")
        
        # 计算评估指标
        metrics = self._calculate_evaluation_metrics(
            all_targets.numpy(), 
            all_predictions.numpy()
        )
        
        return metrics, all_predictions.numpy(), all_targets.numpy()
    
    def _calculate_evaluation_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        计算完整的评估指标
        
        Args:
            y_true: [N, 200] 真实基因表达
            y_pred: [N, 200] 预测基因表达
            
        Returns:
            评估指标字典
        """
        from scipy.stats import pearsonr
        
        # 基础回归指标
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(mse)
        
        # 计算每个基因的皮尔逊相关系数
        gene_correlations = []
        for gene_idx in range(y_true.shape[1]):
            true_gene = y_true[:, gene_idx]
            pred_gene = y_pred[:, gene_idx]
            
            # 跳过方差为0的基因
            if np.var(true_gene) == 0 or np.var(pred_gene) == 0:
                gene_correlations.append(0.0)
            else:
                corr, _ = pearsonr(true_gene, pred_gene)
                gene_correlations.append(corr if not np.isnan(corr) else 0.0)
        
        gene_correlations = np.array(gene_correlations)
        
        # PCC指标
        pcc_mean = np.mean(gene_correlations)
        pcc_top10 = np.mean(np.sort(gene_correlations)[-10:])  # Top 10
        pcc_top50 = np.mean(np.sort(gene_correlations)[-50:])  # Top 50
        pcc_top200 = np.mean(gene_correlations)                # All genes
        
        # RVD (Relative Variance Difference)
        true_var = np.var(y_true, axis=0)
        pred_var = np.var(y_pred, axis=0)
        rvd = np.mean(np.abs(true_var - pred_var) / (true_var + 1e-8))
        
        # R²
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'PCC-Mean': pcc_mean,
            'PCC-10': pcc_top10,
            'PCC-50': pcc_top50,
            'PCC-200': pcc_top200,
            'RVD': rvd,
            'R2': r2,
            'correlations': gene_correlations  # 保存所有基因的相关性
        }
        
        return metrics
    
    def print_evaluation_results(self, metrics: Dict[str, float], prefix: str = ""):
        """打印评估结果"""
        print(f"\n📊 {prefix}评估结果:")
        print("=" * 50)
        print(f"🔹 回归指标:")
        print(f"   MSE:  {metrics['MSE']:.6f}")
        print(f"   MAE:  {metrics['MAE']:.6f}")
        print(f"   RMSE: {metrics['RMSE']:.6f}")
        print(f"   R²:   {metrics['R2']:.6f}")
        
        print(f"\n🔹 相关性指标:")
        print(f"   PCC-Mean: {metrics['PCC-Mean']:.6f}")
        print(f"   PCC-10:   {metrics['PCC-10']:.6f}")
        print(f"   PCC-50:   {metrics['PCC-50']:.6f}")
        print(f"   PCC-200:  {metrics['PCC-200']:.6f}")
        
        print(f"\n🔹 分布指标:")
        print(f"   RVD: {metrics['RVD']:.6f}")
        
        # 相关性分布统计
        correlations = metrics['correlations']
        print(f"\n🔹 基因相关性分布:")
        print(f"   正相关 (>0.1): {np.sum(correlations > 0.1)}/200 ({np.sum(correlations > 0.1)/200*100:.1f}%)")
        print(f"   中等相关 (>0.3): {np.sum(correlations > 0.3)}/200 ({np.sum(correlations > 0.3)/200*100:.1f}%)")
        print(f"   强相关 (>0.5): {np.sum(correlations > 0.5)}/200 ({np.sum(correlations > 0.5)/200*100:.1f}%)")
    
    def save_results(
        self, 
        metrics: Dict[str, float], 
        predictions: np.ndarray,
        targets: np.ndarray,
        save_dir: str = './inference_results'
    ):
        """保存推理结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存评估指标
        metrics_to_save = {}
        for k, v in metrics.items():
            if k != 'correlations':
                # 将numpy类型转换为Python原生类型
                if isinstance(v, np.floating):
                    metrics_to_save[k] = float(v)
                elif isinstance(v, np.integer):
                    metrics_to_save[k] = int(v)
                else:
                    metrics_to_save[k] = v
        
        with open(os.path.join(save_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # 保存基因相关性
        np.save(os.path.join(save_dir, 'gene_correlations.npy'), metrics['correlations'])
        
        # 保存预测和目标
        np.save(os.path.join(save_dir, 'predictions.npy'), predictions)
        np.save(os.path.join(save_dir, 'targets.npy'), targets)
        
        # 保存详细报告
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(os.path.join(save_dir, 'evaluation_report.txt'), 'w') as f:
            f.write("Two-Stage VAR-ST 评估报告\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"Stage 1 检查点: {self.stage1_ckpt_path}\n")
            f.write(f"Stage 2 检查点: {self.stage2_ckpt_path}\n")
            f.write("\n评估指标:\n")
            for key, value in metrics_to_save.items():
                f.write(f"  {key}: {value:.6f}\n")
        
        print(f"💾 结果已保存到: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Two-Stage VAR-ST 完整推理和评估')
    parser.add_argument('--stage1_ckpt', type=str, required=True,
                       help='Stage 1 VQVAE检查点路径')
    parser.add_argument('--stage2_ckpt', type=str, required=True,
                       help='Stage 2 VAR检查点路径')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=list(DATASETS.keys()),
                       help='数据集名称')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['val', 'test'],
                       help='评估模式：验证集或测试集')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--max_batches', type=int, default=None,
                       help='最大批次数（用于快速测试）')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='采样温度')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k采样参数')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Nucleus采样参数')
    parser.add_argument('--save_results', action='store_true',
                       help='是否保存详细结果')
    parser.add_argument('--save_dir', type=str, default='./inference_results',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    print("🚀 Two-Stage VAR-ST 完整推理和评估")
    print("=" * 60)
    
    # 检查设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        device = 'cpu'
    
    try:
        # 1. 初始化推理器
        inferencer = TwoStageCompleteInference(
            stage1_ckpt_path=args.stage1_ckpt,
            stage2_ckpt_path=args.stage2_ckpt,
            device=device
        )
        
        # 2. 加载模型
        model = inferencer.load_model()
        
        # 3. 准备数据
        print(f"\n📊 准备数据...")
        dataset_info = DATASETS[args.dataset]
        
        # 构建简化配置
        config = AddictDict({
            'data_path': dataset_info['path'],
            'slide_val': dataset_info['val_slides'],
            'slide_test': dataset_info['test_slides'],
            'encoder_name': dataset_info['recommended_encoder'],
            'use_augmented': False,  # 推理时不使用数据增强
            'expand_augmented': False,
            'expr_name': args.dataset,  # 添加缺失的字段
            'MODEL': AddictDict({  # 修复：使用MODEL结构
                'model_name': 'TWO_STAGE_VAR_ST'
            }),
            'DATA': {
                'normalize': True,
                f'{args.mode}_dataloader': {
                    'batch_size': args.batch_size,
                    'num_workers': 4,
                    'pin_memory': True,
                    'shuffle': False,
                    'persistent_workers': True
                }
            }
        })
        
        # 创建数据接口
        data_interface = DataInterface(config)
        data_interface.setup(stage=args.mode)  # 添加：调用setup方法
        
        if args.mode == 'val':
            dataloader = data_interface.val_dataloader()
        else:
            dataloader = data_interface.test_dataloader()
        
        print(f"   - 数据集: {args.dataset}")
        print(f"   - 模式: {args.mode}")
        print(f"   - 批次大小: {args.batch_size}")
        print(f"   - 总批次数: {len(dataloader)}")
        
        # 4. 进行评估
        metrics, predictions, targets = inferencer.evaluate_on_dataloader(
            dataloader=dataloader,
            max_batches=args.max_batches,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        
        # 5. 打印结果
        inferencer.print_evaluation_results(metrics, f"{args.dataset} {args.mode.upper()}")
        
        # 6. 保存结果
        if args.save_results:
            inferencer.save_results(metrics, predictions, targets, args.save_dir)
        
        print(f"\n✅ 推理和评估完成！")
        print(f"🎯 关键指标:")
        print(f"   - PCC-Mean: {metrics['PCC-Mean']:.4f}")
        print(f"   - MSE: {metrics['MSE']:.4f}")
        print(f"   - MAE: {metrics['MAE']:.4f}")
        
    except Exception as e:
        print(f"❌ 推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 