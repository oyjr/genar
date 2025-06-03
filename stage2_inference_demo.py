"""
Two-Stage VAR-ST Stage 2 推理演示脚本

此脚本展示如何：
1. 加载训练好的Stage 2模型
2. 进行端到端的基因表达预测推理
3. 可视化推理结果

使用方法：
python stage2_inference_demo.py --ckpt logs/PRAD/TWO_STAGE_VAR_ST/best-epoch=epoch=00-val_mse=0.0000.ckpt
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import numpy as np
import argparse
from typing import Dict, Tuple
import matplotlib.pyplot as plt

from model.VAR.two_stage_var_st import TwoStageVARST
from model.model_interface import ModelInterface
from dataset.data_interface import DataInterface


def load_stage2_model_from_lightning_ckpt(ckpt_path: str, device: str = 'cuda') -> TwoStageVARST:
    """
    从PyTorch Lightning checkpoint加载Stage 2模型
    
    Args:
        ckpt_path: Lightning checkpoint路径
        device: 目标设备
    
    Returns:
        加载好的Two-Stage VAR-ST模型，可用于推理
    """
    print(f"🔄 从 {ckpt_path} 加载Stage 2模型...")
    
    # 加载Lightning checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['state_dict']
    hyper_params = checkpoint.get('hyper_parameters', {})
    
    print(f"   Checkpoint信息:")
    print(f"   - Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   - Global step: {checkpoint.get('global_step', 'unknown')}")
    print(f"   - Lightning版本: {checkpoint.get('pytorch-lightning_version', 'unknown')}")
    
    # 检查是否包含Stage 2权重
    stage1_keys = [k for k in state_dict.keys() if 'stage1_vqvae' in k]
    stage2_keys = [k for k in state_dict.keys() if 'stage2_var' in k]
    condition_keys = [k for k in state_dict.keys() if 'condition_processor' in k]
    
    print(f"   - Stage 1权重: {len(stage1_keys)}个")
    print(f"   - Stage 2权重: {len(stage2_keys)}个")
    print(f"   - 条件处理器权重: {len(condition_keys)}个")
    
    if len(stage2_keys) == 0:
        raise ValueError("Checkpoint中未找到Stage 2权重！请确保使用Stage 2训练的checkpoint。")
    
    # 获取配置信息
    model_config = hyper_params.get('config', {}).get('MODEL', {})
    stage1_ckpt_path = model_config.get('stage1_ckpt_path')
    
    print(f"   - 原始Stage 1 checkpoint: {stage1_ckpt_path}")
    
    # 创建模型实例 - 由于Lightning checkpoint包含完整状态，我们可以直接加载
    model = TwoStageVARST(
        num_genes=model_config.get('num_genes', 200),
        histology_feature_dim=model_config.get('histology_feature_dim', 1024),
        spatial_coord_dim=model_config.get('spatial_coord_dim', 2),
        current_stage=1,  # 先设置为Stage 1，避免要求stage1_ckpt_path
        device=device
    )
    
    # 提取并加载模型权重（去掉Lightning的前缀）
    model_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            # 去掉 'model.' 前缀
            new_key = key[6:]
            model_state_dict[new_key] = value
        else:
            model_state_dict[key] = value
    
    # 加载权重
    model.load_state_dict(model_state_dict, strict=False)
    model = model.to(device)
    
    # 现在设置为Stage 2模式（推理模式，不需要重新加载checkpoint）
    model.current_stage = 2
    model._set_vqvae_trainable(False)  # VQVAE冻结
    model._set_var_trainable(False)    # VAR也冻结（推理模式）
    model.eval()
    
    print(f"✅ Stage 2模型加载成功！")
    return model


def demo_inference(model: TwoStageVARST, num_samples: int = 5, device: str = 'cuda'):
    """
    演示Stage 2推理功能
    
    Args:
        model: 加载好的Stage 2模型
        num_samples: 生成样本数量
        device: 设备
    """
    print(f"\n🧬 开始Stage 2推理演示 (生成 {num_samples} 个样本)...")
    
    # 创建模拟输入数据
    batch_size = num_samples
    histology_features = torch.randn(batch_size, 1024, device=device)  # 模拟组织学特征
    spatial_coords = torch.randn(batch_size, 2, device=device)         # 模拟空间坐标
    
    print(f"   输入数据:")
    print(f"   - 组织学特征: {histology_features.shape}")
    print(f"   - 空间坐标: {spatial_coords.shape}")
    
    # 进行推理
    with torch.no_grad():
        # 基础推理
        results = model.inference(
            histology_features=histology_features,
            spatial_coords=spatial_coords,
            temperature=1.0,  # 控制生成的随机性
            top_k=50,         # Top-k采样
            top_p=0.9         # Nucleus采样
        )
    
    # 解析结果
    predicted_genes = results['predicted_gene_expression']  # [B, 200]
    generated_tokens = results['generated_tokens']          # [B, 241]
    multi_scale_tokens = results['multi_scale_tokens']      # Dict
    
    print(f"\n📊 推理结果:")
    print(f"   - 预测基因表达: {predicted_genes.shape}")
    print(f"   - 生成的tokens: {generated_tokens.shape}")
    print(f"   - 多尺度token结构:")
    for scale, tokens in multi_scale_tokens.items():
        print(f"     {scale}: {tokens.shape}")
    
    # 统计信息
    gene_stats = {
        'mean': predicted_genes.mean(dim=1),     # [B]
        'std': predicted_genes.std(dim=1),       # [B]
        'min': predicted_genes.min(dim=1)[0],    # [B]
        'max': predicted_genes.max(dim=1)[0],    # [B]
    }
    
    print(f"\n📈 预测基因表达统计:")
    for i in range(batch_size):
        print(f"   样本 {i+1}: mean={gene_stats['mean'][i]:.4f}, "
              f"std={gene_stats['std'][i]:.4f}, "
              f"range=[{gene_stats['min'][i]:.4f}, {gene_stats['max'][i]:.4f}]")
    
    return results


def compare_sampling_strategies(model: TwoStageVARST, device: str = 'cuda'):
    """
    比较不同采样策略的效果
    
    Args:
        model: Stage 2模型
        device: 设备
    """
    print(f"\n🎯 比较不同采样策略...")
    
    # 固定输入
    histology_features = torch.randn(1, 1024, device=device)
    spatial_coords = torch.randn(1, 2, device=device)
    
    # 不同采样策略
    sampling_configs = [
        {'name': 'Greedy', 'temperature': 0.1, 'top_k': None, 'top_p': None},
        {'name': 'Low Temp', 'temperature': 0.7, 'top_k': None, 'top_p': None},
        {'name': 'High Temp', 'temperature': 1.5, 'top_k': None, 'top_p': None},
        {'name': 'Top-K', 'temperature': 1.0, 'top_k': 50, 'top_p': None},
        {'name': 'Nucleus', 'temperature': 1.0, 'top_k': None, 'top_p': 0.9},
    ]
    
    results_comparison = {}
    
    for config in sampling_configs:
        with torch.no_grad():
            result = model.inference(
                histology_features=histology_features,
                spatial_coords=spatial_coords,
                temperature=config['temperature'],
                top_k=config['top_k'],
                top_p=config['top_p']
            )
        
        pred_genes = result['predicted_gene_expression'][0]  # [200]
        results_comparison[config['name']] = {
            'predictions': pred_genes.cpu().numpy(),
            'mean': pred_genes.mean().item(),
            'std': pred_genes.std().item(),
            'entropy': -torch.sum(torch.softmax(pred_genes, dim=0) * 
                                 torch.log_softmax(pred_genes, dim=0)).item()
        }
    
    # 打印比较结果
    print(f"\n📊 采样策略比较:")
    print(f"{'Strategy':<10} {'Mean':<8} {'Std':<8} {'Entropy':<10}")
    print("-" * 40)
    for name, stats in results_comparison.items():
        print(f"{name:<10} {stats['mean']:<8.4f} {stats['std']:<8.4f} {stats['entropy']:<10.4f}")
    
    return results_comparison


def save_inference_results(results: Dict, save_dir: str = './inference_results'):
    """
    保存推理结果
    
    Args:
        results: 推理结果字典
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存基因表达预测
    predicted_genes = results['predicted_gene_expression'].cpu().numpy()
    np.save(os.path.join(save_dir, 'predicted_gene_expression.npy'), predicted_genes)
    
    # 保存tokens
    generated_tokens = results['generated_tokens'].cpu().numpy()
    np.save(os.path.join(save_dir, 'generated_tokens.npy'), generated_tokens)
    
    # 保存多尺度tokens
    for scale, tokens in results['multi_scale_tokens'].items():
        np.save(os.path.join(save_dir, f'tokens_{scale}.npy'), tokens.cpu().numpy())
    
    print(f"💾 推理结果已保存到: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Two-Stage VAR-ST Stage 2 推理演示')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='PyTorch Lightning checkpoint路径')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='计算设备')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='生成样本数量')
    parser.add_argument('--save_results', action='store_true',
                       help='是否保存推理结果')
    
    args = parser.parse_args()
    
    print("🚀 Two-Stage VAR-ST Stage 2 推理演示")
    print("=" * 50)
    
    # 检查设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        device = 'cpu'
    
    try:
        # 1. 加载模型
        model = load_stage2_model_from_lightning_ckpt(args.ckpt, device)
        
        # 2. 演示基础推理
        results = demo_inference(model, args.num_samples, device)
        
        # 3. 比较采样策略
        comparison = compare_sampling_strategies(model, device)
        
        # 4. 保存结果（可选）
        if args.save_results:
            save_inference_results(results)
        
        print(f"\n✅ Stage 2推理演示完成！")
        print(f"💡 Tips:")
        print(f"   - 调整temperature参数控制生成的随机性")
        print(f"   - 使用top_k/top_p参数改善生成质量")
        print(f"   - 组织学特征和空间坐标会影响基因表达预测")
        
    except Exception as e:
        print(f"❌ 推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 