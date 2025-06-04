#!/usr/bin/env python3
"""
详细VQVAE诊断脚本

深入检查：
1. 量化过程是否正常工作
2. 直通估计器是否有效
3. 权重加载是否正确
4. codebook是否有意义的分布
5. 距离计算是否正确
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

def detailed_quantization_analysis():
    """详细分析量化过程"""
    print("🔬 详细量化过程分析")
    print("=" * 60)
    
    # 1. 加载VQVAE
    stage1_ckpt = "logs/PRAD/TWO_STAGE_VAR_ST/stage1-best-epoch=epoch=143-val_mse=val_mse=0.5353.ckpt"
    checkpoint = torch.load(stage1_ckpt, map_location='cpu')
    
    vqvae = MultiScaleGeneVQVAE()
    
    # 检查权重加载
    print("🔍 检查权重加载...")
    if 'state_dict' in checkpoint:
        state_dict = {}
        stage1_keys_found = []
        all_keys = list(checkpoint['state_dict'].keys())
        
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('model.stage1_vqvae.'):
                new_key = key.replace('model.stage1_vqvae.', '')
                state_dict[new_key] = value
                stage1_keys_found.append(key)
        
        print(f"   总键数: {len(all_keys)}")
        print(f"   Stage1键数: {len(stage1_keys_found)}")
        print(f"   Stage1键示例: {stage1_keys_found[:3]}")
        
        # 检查是否有codebook权重
        codebook_keys = [k for k in state_dict.keys() if 'embedding' in k]
        print(f"   Codebook相关键: {codebook_keys}")
        
        missing_keys, unexpected_keys = vqvae.load_state_dict(state_dict, strict=False)
        print(f"   缺失键: {len(missing_keys)}")
        print(f"   多余键: {len(unexpected_keys)}")
        if missing_keys:
            print(f"   缺失键示例: {missing_keys[:3]}")
        if unexpected_keys:
            print(f"   多余键示例: {unexpected_keys[:3]}")
    
    vqvae.cuda()
    vqvae.eval()
    
    # 2. 检查codebook权重
    print(f"\n🧮 检查Codebook权重...")
    codebook_weight = vqvae.shared_quantizer.embedding.weight
    print(f"   Codebook形状: {codebook_weight.shape}")
    print(f"   Codebook范围: [{codebook_weight.min().item():.6f}, {codebook_weight.max().item():.6f}]")
    print(f"   Codebook均值: {codebook_weight.mean().item():.6f}")
    print(f"   Codebook标准差: {codebook_weight.std().item():.6f}")
    
    # 检查codebook是否有意义的分布
    if torch.allclose(codebook_weight, torch.zeros_like(codebook_weight)):
        print("   ❌ Codebook全为0！")
    elif torch.allclose(codebook_weight, codebook_weight[0:1].expand_as(codebook_weight)):
        print("   ❌ Codebook所有向量相同！")
    else:
        print("   ✅ Codebook有正常分布")
    
    # 3. 准备测试数据
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
                'batch_size': 4,
                'num_workers': 0,
                'shuffle': False,
                'persistent_workers': False
            }
        }
    })
    
    data_interface = DataInterface(config)
    data_interface.setup(stage='test')
    dataloader = data_interface.test_dataloader()
    batch = next(iter(dataloader))
    gene_expression = batch['target_genes'].cuda()[:2]  # 只取2个样本方便调试
    
    print(f"\n📊 测试数据:")
    print(f"   批次大小: {gene_expression.shape[0]}")
    print(f"   基因表达范围: [{gene_expression.min().item():.4f}, {gene_expression.max().item():.4f}]")
    
    # 4. 逐步分析编码过程
    print(f"\n🔬 逐步分析编码过程...")
    
    with torch.no_grad():
        # Step 1: 多尺度分解
        decomposed = vqvae.decomposer(gene_expression)
        print(f"   多尺度分解:")
        for scale, features in decomposed.items():
            print(f"     {scale}: {features.shape}, 范围[{features.min().item():.4f}, {features.max().item():.4f}]")
        
        # Step 2: 编码到128维
        encoded = {}
        for scale in ['global', 'pathway', 'module', 'individual']:
            encoded[scale] = vqvae.encoders[scale](decomposed[scale])
            print(f"   {scale}编码: {encoded[scale].shape}, 范围[{encoded[scale].min().item():.4f}, {encoded[scale].max().item():.4f}]")
        
        # Step 3: 详细分析量化过程
        print(f"\n🎯 详细量化过程分析...")
        
        for scale in ['global', 'pathway', 'module', 'individual']:
            print(f"\n   --- {scale.upper()}尺度量化 ---")
            x = encoded[scale]
            print(f"   输入特征: {x.shape}")
            print(f"   输入范围: [{x.min().item():.6f}, {x.max().item():.6f}]")
            
            # 手动执行量化步骤
            input_shape = x.shape
            
            # 处理维度
            if x.dim() == 2:
                x = x.unsqueeze(1)
                squeeze_output = True
            else:
                squeeze_output = False
            
            B, N, D = x.shape
            flat_x = x.view(-1, D)  # [B*N, D]
            print(f"   展平后: {flat_x.shape}")
            
            # 计算距离
            distances = torch.cdist(flat_x, codebook_weight)  # [B*N, vocab_size]
            print(f"   距离矩阵: {distances.shape}")
            print(f"   距离范围: [{distances.min().item():.6f}, {distances.max().item():.6f}]")
            
            # 获取最近的tokens
            tokens_flat = torch.argmin(distances, dim=1)  # [B*N]
            tokens = tokens_flat.view(B, N)  # [B, N]
            print(f"   选中tokens: {tokens.flatten()[:10].tolist()}")
            
            # 获取量化特征
            quantized = vqvae.shared_quantizer.embedding(tokens)  # [B, N, embed_dim]
            print(f"   量化特征: {quantized.shape}")
            print(f"   量化范围: [{quantized.min().item():.6f}, {quantized.max().item():.6f}]")
            
            # 🔍 关键检查：量化前后的差异
            quantization_error = torch.nn.functional.mse_loss(quantized, x)
            print(f"   ⚠️ 量化误差: {quantization_error.item():.8f}")
            
            if quantization_error.item() < 1e-6:
                print(f"   ❌ 量化误差过小，可能没有真正量化！")
                
                # 详细检查是否输入特征与codebook某些向量完全匹配
                for i in range(min(3, flat_x.shape[0])):
                    input_vec = flat_x[i]  # [128]
                    selected_token = tokens_flat[i].item()
                    codebook_vec = codebook_weight[selected_token]  # [128]
                    vec_diff = torch.nn.functional.mse_loss(input_vec, codebook_vec)
                    print(f"     样本{i}: token={selected_token}, 向量差异={vec_diff.item():.8f}")
                    
                    if vec_diff.item() < 1e-6:
                        print(f"     ❌ 输入向量与codebook向量几乎完全相同！")
            else:
                print(f"   ✅ 有正常的量化误差")
            
            # 检查直通估计器
            quantized_with_grad = x + (quantized - x).detach()
            straight_through_diff = torch.nn.functional.mse_loss(quantized_with_grad, quantized)
            print(f"   直通估计器差异: {straight_through_diff.item():.8f}")

def check_training_vs_inference_consistency():
    """检查训练模式vs推理模式的一致性"""
    print(f"\n🔄 检查训练vs推理模式一致性")
    print("=" * 50)
    
    # 加载模型
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
    
    # 准备数据
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
                'batch_size': 2,
                'num_workers': 0,
                'shuffle': False,
                'persistent_workers': False
            }
        }
    })
    
    data_interface = DataInterface(config)
    data_interface.setup(stage='test')
    dataloader = data_interface.test_dataloader()
    batch = next(iter(dataloader))
    gene_expression = batch['target_genes'].cuda()[:2]
    
    # 测试训练模式
    print("🎓 训练模式测试:")
    vqvae.train()
    with torch.no_grad():  # 即使在训练模式也用no_grad，因为只是测试
        train_result = vqvae(gene_expression)
        train_recon = train_result['final_reconstruction']
        train_tokens = train_result['tokens']
        print(f"   重建范围: [{train_recon.min().item():.4f}, {train_recon.max().item():.4f}]")
        print(f"   VQ损失: {train_result['total_vq_loss'].item():.6f}")
    
    # 测试推理模式
    print("🔍 推理模式测试:")
    vqvae.eval()
    with torch.no_grad():
        eval_result = vqvae(gene_expression)
        eval_recon = eval_result['final_reconstruction']
        eval_tokens = eval_result['tokens']
        print(f"   重建范围: [{eval_recon.min().item():.4f}, {eval_recon.max().item():.4f}]")
        print(f"   VQ损失: {eval_result['total_vq_loss'].item():.6f}")
    
    # 对比差异
    mode_diff = torch.nn.functional.mse_loss(train_recon, eval_recon)
    print(f"📊 训练vs推理模式差异: {mode_diff.item():.8f}")
    
    if mode_diff.item() < 1e-6:
        print("   ✅ 训练和推理模式结果一致")
    else:
        print("   ⚠️ 训练和推理模式有差异")
    
    # 检查tokens是否相同
    tokens_same = True
    for scale in ['global', 'pathway', 'module', 'individual']:
        scale_diff = torch.allclose(train_tokens[scale], eval_tokens[scale])
        print(f"   {scale} tokens相同: {scale_diff}")
        if not scale_diff:
            tokens_same = False
    
    if tokens_same:
        print("   ✅ 所有尺度tokens在训练和推理模式下相同")
    else:
        print("   ❌ 不同模式下tokens有差异")

def test_random_input_quantization():
    """用随机输入测试量化是否工作"""
    print(f"\n🎲 随机输入量化测试")
    print("=" * 40)
    
    # 加载模型
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
    
    # 创建随机输入
    random_gene_expression = torch.randn(2, 200).cuda() * 2.0  # 随机基因表达
    print(f"随机输入范围: [{random_gene_expression.min().item():.4f}, {random_gene_expression.max().item():.4f}]")
    
    with torch.no_grad():
        # 编码
        encode_result = vqvae.encode(random_gene_expression)
        tokens = encode_result['tokens']
        vq_loss = encode_result['vq_loss']
        
        print(f"VQ损失: {vq_loss.item():.6f}")
        
        # 解码
        decode_result = vqvae.decode_from_tokens(tokens)
        reconstructed = decode_result['final_reconstruction']
        
        # 计算重建误差
        recon_error = torch.nn.functional.mse_loss(reconstructed, random_gene_expression)
        print(f"重建误差: {recon_error.item():.6f}")
        
        print(f"重建范围: [{reconstructed.min().item():.4f}, {reconstructed.max().item():.4f}]")
        
        # 如果重建误差为0，说明有问题
        if recon_error.item() < 1e-6:
            print("❌ 随机输入的重建误差也为0，量化过程有问题！")
        else:
            print("✅ 随机输入有正常的重建误差")

if __name__ == "__main__":
    detailed_quantization_analysis()
    check_training_vs_inference_consistency()
    test_random_input_quantization() 