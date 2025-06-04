#!/usr/bin/env python3
"""
测试Stage 1 VQVAE的编码-解码能力

目的：
1. 验证VQVAE本身的重建能力是否正常
2. 对比直接编码-解码 vs 从VAR生成的tokens解码
3. 找出推理问题的真正原因
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import numpy as np
from addict import Dict as AddictDict
from dataset.data_interface import DataInterface
from model.VAR.multi_scale_gene_vqvae import MultiScaleGeneVQVAE
from main import DATASETS

def test_vqvae_direct_reconstruction():
    """测试VQVAE直接编码-解码能力"""
    print("🧪 测试VQVAE直接编码-解码能力")
    print("=" * 50)
    
    # 1. 加载VQVAE模型
    stage1_ckpt = "logs/PRAD/TWO_STAGE_VAR_ST/stage1-best-epoch=epoch=143-val_mse=val_mse=0.5353.ckpt"
    
    # 从checkpoint加载配置
    checkpoint = torch.load(stage1_ckpt, map_location='cpu')
    if 'hyper_parameters' in checkpoint and 'config' in checkpoint['hyper_parameters']:
        # PyTorch Lightning格式
        model_config = checkpoint['hyper_parameters']['config']
        if hasattr(model_config, 'MODEL'):
            vqvae_config = {
                'vocab_size': getattr(model_config.MODEL, 'vocab_size', 4096),
                'embed_dim': getattr(model_config.MODEL, 'embed_dim', 128),
                'beta': getattr(model_config.MODEL, 'beta', 0.25),
                'hierarchical_loss_weight': getattr(model_config.MODEL, 'hierarchical_loss_weight', 0.1),
                'vq_loss_weight': getattr(model_config.MODEL, 'vq_loss_weight', 0.25)
            }
        else:
            vqvae_config = {}
    else:
        vqvae_config = {}
    
    # 使用默认配置
    vqvae = MultiScaleGeneVQVAE(**vqvae_config)
    
    # 加载权重
    if 'state_dict' in checkpoint:
        # PyTorch Lightning格式
        state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('model.stage1_vqvae.'):
                new_key = key.replace('model.stage1_vqvae.', '')
                state_dict[new_key] = value
        vqvae.load_state_dict(state_dict, strict=False)
    else:
        vqvae.load_state_dict(checkpoint['model_state_dict'])
    
    vqvae.cuda()
    vqvae.eval()
    
    print(f"✅ VQVAE模型加载完成")
    
    # 2. 准备测试数据
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
                'batch_size': 8,
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
    
    # 3. 测试直接编码-解码
    batch = next(iter(dataloader))
    gene_expression = batch['target_genes'].cuda()  # [B, 200]
    
    print(f"\n📊 测试数据:")
    print(f"   批次大小: {gene_expression.shape[0]}")
    print(f"   基因数量: {gene_expression.shape[1]}")
    print(f"   目标范围: [{gene_expression.min().item():.4f}, {gene_expression.max().item():.4f}]")
    print(f"   目标均值: {gene_expression.mean().item():.4f}")
    print(f"   目标标准差: {gene_expression.std().item():.4f}")
    
    with torch.no_grad():
        # 测试1: 完整前向传播（训练时的方式）
        print(f"\n🔄 测试1: 完整前向传播（训练方式）")
        full_result = vqvae(gene_expression)
        reconstructed = full_result['final_reconstruction']
        
        print(f"   重建范围: [{reconstructed.min().item():.4f}, {reconstructed.max().item():.4f}]")
        print(f"   重建均值: {reconstructed.mean().item():.4f}")
        print(f"   重建标准差: {reconstructed.std().item():.4f}")
        
        # 计算重建误差
        mse_loss = torch.nn.functional.mse_loss(reconstructed, gene_expression)
        mae_loss = torch.nn.functional.l1_loss(reconstructed, gene_expression)
        
        print(f"   重建MSE: {mse_loss.item():.6f}")
        print(f"   重建MAE: {mae_loss.item():.6f}")
        
        # 测试2: 分步编码-解码
        print(f"\n🔄 测试2: 分步编码-解码")
        encode_result = vqvae.encode(gene_expression)
        tokens = encode_result['tokens']
        
        print(f"   生成的tokens:")
        for scale, scale_tokens in tokens.items():
            unique_tokens = torch.unique(scale_tokens)
            print(f"     {scale}: 形状{scale_tokens.shape}, 唯一tokens数量{len(unique_tokens)}, 范围[{scale_tokens.min().item()}, {scale_tokens.max().item()}]")
        
        # 从tokens重建
        decode_result = vqvae.decode_from_tokens(tokens)
        tokens_reconstructed = decode_result['final_reconstruction']
        
        print(f"   从tokens重建:")
        print(f"   重建范围: [{tokens_reconstructed.min().item():.4f}, {tokens_reconstructed.max().item():.4f}]")
        print(f"   重建均值: {tokens_reconstructed.mean().item():.4f}")
        print(f"   重建标准差: {tokens_reconstructed.std().item():.4f}")
        
        tokens_mse = torch.nn.functional.mse_loss(tokens_reconstructed, gene_expression)
        tokens_mae = torch.nn.functional.l1_loss(tokens_reconstructed, gene_expression)
        
        print(f"   tokens重建MSE: {tokens_mse.item():.6f}")
        print(f"   tokens重建MAE: {tokens_mae.item():.6f}")
        
        # 测试3: 验证两种方式的一致性
        print(f"\n🔄 测试3: 验证重建一致性")
        consistency_error = torch.nn.functional.mse_loss(reconstructed, tokens_reconstructed)
        print(f"   完整前向 vs tokens重建的MSE: {consistency_error.item():.8f}")
        
        if consistency_error.item() < 1e-5:
            print("   ✅ 两种重建方式完全一致")
        else:
            print("   ❌ 两种重建方式不一致，可能有问题")
            
        # 测试4: 与训练时的val_mse对比
        print(f"\n📈 与训练指标对比:")
        print(f"   训练时val_mse: 0.5353")
        print(f"   当前测试MSE: {mse_loss.item():.6f}")
        
        if abs(mse_loss.item() - 0.5353) < 0.1:
            print("   ✅ 重建质量与训练时一致")
        else:
            print("   ⚠️ 重建质量与训练时有差异")

def test_vqvae_token_diversity():
    """测试VQVAE的token多样性"""
    print(f"\n🎲 测试token多样性")
    print("=" * 30)
    
    # 加载模型（复用上面的代码）
    stage1_ckpt = "logs/PRAD/TWO_STAGE_VAR_ST/stage1-best-epoch=epoch=143-val_mse=val_mse=0.5353.ckpt"
    checkpoint = torch.load(stage1_ckpt, map_location='cpu')
    
    vqvae = MultiScaleGeneVQVAE()
    
    if 'state_dict' in checkpoint:
        state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('model.stage1_vqvae.'):
                new_key = key.replace('model.stage1_vqvae.', '')
                state_dict[new_key] = value
        vqvae.load_state_dict(state_dict, strict=False)
    
    vqvae.cuda()
    vqvae.eval()
    
    # 准备更多数据
    dataset_info = DATASETS['PRAD']
    config = AddictDict({
        'data_path': dataset_info['path'],
        'slide_test': dataset_info['test_slides'],
        'encoder_name': dataset_info['recommended_encoder'],
        'use_augmented': False,
        'expr_name': 'PRAD',
        'MODEL': AddictDict({'model_name': 'TWO_STAGE_VAR_ST'}),
        'DATA': {
            'normalize': True,
            'test_dataloader': {
                'batch_size': 32,
                'num_workers': 0,
                'shuffle': False,
                'persistent_workers': False
            }
        }
    })
    
    data_interface = DataInterface(config)
    data_interface.setup(stage='test')
    dataloader = data_interface.test_dataloader()
    
    all_tokens = {scale: [] for scale in ['global', 'pathway', 'module', 'individual']}
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:  # 测试10个批次
                break
                
            gene_expression = batch['target_genes'].cuda()
            encode_result = vqvae.encode(gene_expression)
            tokens = encode_result['tokens']
            
            for scale in ['global', 'pathway', 'module', 'individual']:
                all_tokens[scale].append(tokens[scale].cpu())
    
    # 分析token分布
    for scale in ['global', 'pathway', 'module', 'individual']:
        all_scale_tokens = torch.cat(all_tokens[scale], dim=0).flatten()
        unique_tokens, counts = torch.unique(all_scale_tokens, return_counts=True)
        
        print(f"   {scale}尺度:")
        print(f"     总tokens数: {len(all_scale_tokens)}")
        print(f"     唯一tokens数: {len(unique_tokens)}/4096")
        print(f"     利用率: {len(unique_tokens)/4096:.3f}")
        print(f"     最频繁的5个tokens: {unique_tokens[:5].tolist()}")

if __name__ == "__main__":
    test_vqvae_direct_reconstruction()
    test_vqvae_token_diversity() 