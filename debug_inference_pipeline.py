#!/usr/bin/env python3
"""
调试两阶段VAR-ST推理管道

检查每个步骤的输入输出，找出指标异常的原因：
1. 检查模型权重加载
2. 检查数据预处理
3. 检查Stage 2生成的tokens
4. 检查Stage 1重建的基因表达
5. 检查最终预测值的分布
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import numpy as np
from two_stage_complete_inference import TwoStageCompleteInference
from main import DATASETS
from addict import Dict as AddictDict
from dataset.data_interface import DataInterface

def debug_inference_pipeline():
    """调试完整的推理管道"""
    print("🔍 开始调试两阶段VAR-ST推理管道")
    print("=" * 60)
    
    # 1. 初始化推理器
    stage1_ckpt = "logs/PRAD/TWO_STAGE_VAR_ST/stage1-best-epoch=epoch=143-val_mse=val_mse=0.5353.ckpt"
    stage2_ckpt = "logs/PRAD/TWO_STAGE_VAR_ST/stage2-best-epoch=epoch=03-val_acc=val_accuracy=0.8263.ckpt"
    
    inferencer = TwoStageCompleteInference(
        stage1_ckpt_path=stage1_ckpt,
        stage2_ckpt_path=stage2_ckpt,
        device='cuda'
    )
    
    # 2. 加载模型
    model = inferencer.load_model()
    
    # 3. 准备一小批数据
    print(f"\n📊 准备调试数据...")
    dataset_info = DATASETS['PRAD']
    
    config = AddictDict({
        'data_path': dataset_info['path'],
        'slide_val': dataset_info['val_slides'],
        'slide_test': dataset_info['test_slides'],
        'encoder_name': dataset_info['recommended_encoder'],
        'use_augmented': False,
        'expand_augmented': False,
        'expr_name': 'PRAD',
        'MODEL': AddictDict({'model_name': 'TWO_STAGE_VAR_ST'}),
        'DATA': {
            'normalize': True,
            'test_dataloader': {
                'batch_size': 4,  # 小批次用于调试
                'num_workers': 0,
                'pin_memory': True,
                'shuffle': False,
                'persistent_workers': False
            }
        }
    })
    
    data_interface = DataInterface(config)
    data_interface.setup(stage='test')
    dataloader = data_interface.test_dataloader()
    
    # 4. 获取一个批次进行详细调试
    batch = next(iter(dataloader))
    histology_features = batch['img'].cuda()      # [B, 1024]
    spatial_coords = batch['positions'].cuda()   # [B, 2]
    target_genes = batch['target_genes'].cuda()  # [B, 200]
    
    print(f"\n🔍 调试批次信息:")
    print(f"   批次大小: {histology_features.shape[0]}")
    print(f"   组织学特征: {histology_features.shape}")
    print(f"   空间坐标: {spatial_coords.shape}")
    print(f"   目标基因: {target_genes.shape}")
    print(f"   目标基因范围: [{target_genes.min().item():.4f}, {target_genes.max().item():.4f}]")
    print(f"   目标基因均值: {target_genes.mean().item():.4f}")
    print(f"   目标基因标准差: {target_genes.std().item():.4f}")
    
    # 5. 逐步调试推理过程
    print(f"\n🔧 开始逐步调试推理过程...")
    
    model.eval()
    with torch.no_grad():
        # Step 1: 条件处理
        print(f"\n   步骤1: 条件处理...")
        condition_embed = model.condition_processor(histology_features, spatial_coords)
        print(f"   条件嵌入形状: {condition_embed.shape}")
        print(f"   条件嵌入范围: [{condition_embed.min().item():.4f}, {condition_embed.max().item():.4f}]")
        print(f"   条件嵌入均值: {condition_embed.mean().item():.4f}")
        
        # Step 2: VAR生成tokens
        print(f"\n   步骤2: VAR生成tokens...")
        try:
            generated_tokens = model.stage2_var.generate(
                condition_embed=condition_embed,
                max_length=241,
                temperature=1.0,
                top_k=50,
                top_p=0.9
            )
            print(f"   生成tokens形状: {generated_tokens.shape}")
            print(f"   生成tokens范围: [{generated_tokens.min().item()}, {generated_tokens.max().item()}]")
            print(f"   生成tokens前10个: {generated_tokens[0, :10].cpu().tolist()}")
            
            # 检查tokens分布
            unique_tokens, counts = torch.unique(generated_tokens, return_counts=True)
            print(f"   唯一tokens数量: {len(unique_tokens)}/{model.stage2_var.vocab_size}")
            print(f"   最常见的5个tokens: {unique_tokens[:5].cpu().tolist()}")
            
        except Exception as e:
            print(f"   ❌ VAR生成失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 3: 重构多尺度tokens
        print(f"\n   步骤3: 重构多尺度tokens...")
        tokens = {
            'global': generated_tokens[:, 0:1],         # [B, 1]
            'pathway': generated_tokens[:, 1:9],        # [B, 8]
            'module': generated_tokens[:, 9:41],        # [B, 32]
            'individual': generated_tokens[:, 41:241]   # [B, 200]
        }
        
        for scale, scale_tokens in tokens.items():
            print(f"   {scale} tokens: {scale_tokens.shape}, 范围: [{scale_tokens.min().item()}, {scale_tokens.max().item()}]")
        
        # Step 4: VQVAE解码
        print(f"\n   步骤4: VQVAE解码...")
        try:
            decoded_output = model.stage1_vqvae.decode_from_tokens(tokens)
            predicted_gene_expression = decoded_output['final_reconstruction']
            
            print(f"   预测基因表达形状: {predicted_gene_expression.shape}")
            print(f"   预测基因表达范围: [{predicted_gene_expression.min().item():.4f}, {predicted_gene_expression.max().item():.4f}]")
            print(f"   预测基因表达均值: {predicted_gene_expression.mean().item():.4f}")
            print(f"   预测基因表达标准差: {predicted_gene_expression.std().item():.4f}")
            
        except Exception as e:
            print(f"   ❌ VQVAE解码失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 5: 与目标对比
        print(f"\n   步骤5: 预测与目标对比...")
        
        pred_flat = predicted_gene_expression.view(-1).cpu().numpy()
        target_flat = target_genes.view(-1).cpu().numpy()
        
        print(f"   预测统计:")
        print(f"     均值: {pred_flat.mean():.4f}")
        print(f"     标准差: {pred_flat.std():.4f}")
        print(f"     最小值: {pred_flat.min():.4f}")
        print(f"     最大值: {pred_flat.max():.4f}")
        
        print(f"   目标统计:")
        print(f"     均值: {target_flat.mean():.4f}")
        print(f"     标准差: {target_flat.std():.4f}")
        print(f"     最小值: {target_flat.min():.4f}")
        print(f"     最大值: {target_flat.max():.4f}")
        
        # 计算基本指标
        mse = np.mean((pred_flat - target_flat) ** 2)
        mae = np.mean(np.abs(pred_flat - target_flat))
        
        from scipy.stats import pearsonr
        pcc, _ = pearsonr(pred_flat, target_flat)
        
        print(f"   基本指标:")
        print(f"     MSE: {mse:.4f}")
        print(f"     MAE: {mae:.4f}")
        print(f"     PCC: {pcc:.4f}")
        
        # Step 6: 检查是否有异常值
        print(f"\n   步骤6: 异常值检查...")
        
        # 检查预测中的异常值
        pred_q99 = np.percentile(pred_flat, 99)
        pred_q01 = np.percentile(pred_flat, 1)
        target_q99 = np.percentile(target_flat, 99)
        target_q01 = np.percentile(target_flat, 1)
        
        print(f"   预测值分位数: 1%={pred_q01:.4f}, 99%={pred_q99:.4f}")
        print(f"   目标值分位数: 1%={target_q01:.4f}, 99%={target_q99:.4f}")
        
        # 检查NaN和Inf
        pred_nan = np.isnan(pred_flat).sum()
        pred_inf = np.isinf(pred_flat).sum()
        target_nan = np.isnan(target_flat).sum()
        target_inf = np.isinf(target_flat).sum()
        
        print(f"   预测值异常: NaN={pred_nan}, Inf={pred_inf}")
        print(f"   目标值异常: NaN={target_nan}, Inf={target_inf}")
        
        # Step 7: 检查几个具体基因的预测
        print(f"\n   步骤7: 具体基因预测检查...")
        for i in range(min(5, target_genes.shape[1])):
            pred_gene = predicted_gene_expression[:, i].cpu().numpy()
            target_gene = target_genes[:, i].cpu().numpy()
            gene_pcc, _ = pearsonr(pred_gene, target_gene)
            
            print(f"   基因{i}: 预测均值={pred_gene.mean():.4f}, 目标均值={target_gene.mean():.4f}, PCC={gene_pcc:.4f}")

def debug_stage1_reconstruction():
    """调试Stage 1的重建能力"""
    print(f"\n🧪 调试Stage 1重建能力...")
    
    # 加载Stage 1检查点
    stage1_ckpt = "logs/PRAD/TWO_STAGE_VAR_ST/stage1-best-epoch=epoch=143-val_mse=val_mse=0.5353.ckpt"
    checkpoint = torch.load(stage1_ckpt, map_location='cpu')
    
    print(f"   Stage 1 checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Stage 1 val_mse: {checkpoint.get('val_mse', 'unknown')}")
    
    # TODO: 可以直接测试Stage 1的编码-解码能力

def debug_stage2_generation():
    """调试Stage 2的生成能力"""
    print(f"\n🧪 调试Stage 2生成能力...")
    
    # 加载Stage 2检查点
    stage2_ckpt = "logs/PRAD/TWO_STAGE_VAR_ST/stage2-best-epoch=epoch=03-val_acc=val_accuracy=0.8263.ckpt"
    checkpoint = torch.load(stage2_ckpt, map_location='cpu')
    
    print(f"   Stage 2 checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   Stage 2 val_accuracy: {checkpoint.get('val_accuracy', 'unknown')}")
    
    # TODO: 可以直接测试Stage 2的token生成能力

if __name__ == "__main__":
    debug_inference_pipeline()
    debug_stage1_reconstruction()
    debug_stage2_generation() 