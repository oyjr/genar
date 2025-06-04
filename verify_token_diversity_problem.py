#!/usr/bin/env python3
"""
验证Token多样性问题

对比：
1. Stage 1 VQVAE编码产生的tokens分布
2. Stage 2 VAR生成的tokens分布  
3. 验证这是否是推理异常的根本原因
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import numpy as np
from collections import Counter
from addict import Dict as AddictDict
from dataset.data_interface import DataInterface
from model.VAR.multi_scale_gene_vqvae import MultiScaleGeneVQVAE
from model.VAR.two_stage_var_st import TwoStageVARST
from main import DATASETS

def analyze_stage1_token_distribution():
    """分析Stage 1产生的token分布"""
    print("🔬 分析Stage 1 VQVAE的Token分布")
    print("=" * 50)
    
    # 加载Stage 1模型
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
    
    # 收集所有tokens
    all_tokens = {scale: [] for scale in ['global', 'pathway', 'module', 'individual']}
    
    print("🔄 处理测试数据...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 20:  # 处理20个批次，足够统计
                break
            
            gene_expression = batch['target_genes'].cuda()
            encode_result = vqvae.encode(gene_expression)
            tokens = encode_result['tokens']
            
            for scale in ['global', 'pathway', 'module', 'individual']:
                all_tokens[scale].append(tokens[scale].cpu())
            
            if i % 5 == 0:
                print(f"   处理批次 {i}/20")
    
    # 分析token分布
    print(f"\n📊 Stage 1 Token分布统计:")
    for scale in ['global', 'pathway', 'module', 'individual']:
        all_scale_tokens = torch.cat(all_tokens[scale], dim=0).flatten().numpy()
        token_counts = Counter(all_scale_tokens)
        unique_tokens = len(token_counts)
        
        print(f"\n{scale.upper()}尺度:")
        print(f"   总token数: {len(all_scale_tokens)}")
        print(f"   唯一token数: {unique_tokens}/4096")
        print(f"   利用率: {unique_tokens/4096:.4f}")
        
        # 显示最频繁的tokens
        most_common = token_counts.most_common(10)
        print(f"   最频繁的10个tokens:")
        for token, count in most_common:
            print(f"     Token {token}: {count}次 ({count/len(all_scale_tokens)*100:.1f}%)")
        
        # 计算集中度（前10个token占比）
        top10_count = sum([count for _, count in most_common])
        concentration = top10_count / len(all_scale_tokens)
        print(f"   前10个token占比: {concentration*100:.1f}%")
        
        if concentration > 0.8:
            print(f"   ❌ 高度集中！大部分输入被映射到少数tokens")
        elif concentration > 0.5:
            print(f"   ⚠️ 较为集中，多样性不足")
        else:
            print(f"   ✅ 分布相对均匀")

def compare_stage1_vs_var_tokens():
    """对比Stage 1编码的tokens vs VAR生成的tokens"""
    print(f"\n🆚 对比Stage 1编码 vs VAR生成的Token分布")
    print("=" * 60)
    
    # 加载完整的两阶段模型
    stage1_ckpt = "logs/PRAD/TWO_STAGE_VAR_ST/stage1-best-epoch=epoch=epoch=143-val_mse=val_mse=0.5353.ckpt"
    stage2_ckpt = "logs/PRAD/TWO_STAGE_VAR_ST/stage2-best-epoch=epoch=03-val_acc=val_accuracy=0.8263.ckpt"
    
    model_config = AddictDict({
        'MODEL': {
            'model_name': 'TWO_STAGE_VAR_ST',
            'stage1_ckpt': stage1_ckpt,
            'stage2_ckpt': stage2_ckpt,
            'feature_dim': 1024,
            'vocab_size': 4096,
            'embed_dim': 128,
            'depth': 16,
            'num_heads': 16,
            'num_genes': 200,
            'beta': 0.25,
            'hierarchical_loss_weight': 0.1,
            'vq_loss_weight': 0.25
        }
    })
    
    model = TwoStageVARST(model_config)
    model.cuda()
    model.eval()
    
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
                'batch_size': 16,
                'num_workers': 0,
                'shuffle': False,
                'persistent_workers': False
            }
        }
    })
    
    data_interface = DataInterface(config)
    data_interface.setup(stage='test')
    dataloader = data_interface.test_dataloader()
    
    # 收集tokens
    stage1_tokens = {scale: [] for scale in ['global', 'pathway', 'module', 'individual']}
    var_tokens = {scale: [] for scale in ['global', 'pathway', 'module', 'individual']}
    
    print("🔄 处理对比数据...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:  # 处理10个批次
                break
            
            histology_features = batch['feat'].cuda()
            spatial_coords = batch['coords'].cuda()
            gene_expression = batch['target_genes'].cuda()
            
            # 1. Stage 1直接编码
            encode_result = model.stage1_vqvae.encode(gene_expression)
            s1_tokens = encode_result['tokens']
            
            # 2. VAR生成tokens
            var_result = model.stage2_var.generate_tokens(histology_features, spatial_coords)
            v_tokens = var_result['tokens']
            
            # 收集
            for scale in ['global', 'pathway', 'module', 'individual']:
                stage1_tokens[scale].append(s1_tokens[scale].cpu())
                var_tokens[scale].append(v_tokens[scale].cpu())
            
            if i % 2 == 0:
                print(f"   处理批次 {i}/10")
    
    # 对比分析
    print(f"\n📊 Stage 1编码 vs VAR生成 对比:")
    
    for scale in ['global', 'pathway', 'module', 'individual']:
        s1_all = torch.cat(stage1_tokens[scale], dim=0).flatten().numpy()
        var_all = torch.cat(var_tokens[scale], dim=0).flatten().numpy()
        
        s1_unique = len(set(s1_all))
        var_unique = len(set(var_all))
        
        print(f"\n{scale.upper()}尺度:")
        print(f"   Stage 1唯一tokens: {s1_unique}/4096 ({s1_unique/4096:.4f})")
        print(f"   VAR生成唯一tokens: {var_unique}/4096 ({var_unique/4096:.4f})")
        
        # 重叠分析
        s1_set = set(s1_all)
        var_set = set(var_all)
        overlap = len(s1_set.intersection(var_set))
        
        print(f"   Token重叠数: {overlap}")
        print(f"   重叠率: {overlap/max(s1_unique, var_unique)*100:.1f}%")
        
        # 分布集中度对比
        s1_counter = Counter(s1_all)
        var_counter = Counter(var_all)
        
        s1_top5 = [count for _, count in s1_counter.most_common(5)]
        var_top5 = [count for _, count in var_counter.most_common(5)]
        
        s1_concentration = sum(s1_top5) / len(s1_all)
        var_concentration = sum(var_top5) / len(var_all)
        
        print(f"   Stage 1前5个token占比: {s1_concentration*100:.1f}%")
        print(f"   VAR前5个token占比: {var_concentration*100:.1f}%")
        
        if var_concentration > s1_concentration + 0.1:
            print(f"   ❌ VAR生成的tokens更加集中，多样性损失严重")
        elif var_concentration > s1_concentration:
            print(f"   ⚠️ VAR略微增加了集中度")
        else:
            print(f"   ✅ VAR保持了token多样性")

def test_codebook_utilization_during_training():
    """测试训练过程中codebook的实际利用情况"""
    print(f"\n🎯 测试训练数据的Codebook利用情况")
    print("=" * 50)
    
    # 加载Stage 1模型
    stage1_ckpt = "logs/PRAD/TWO_STAGE_VAR_ST/stage1-best-epoch=epoch=epoch=143-val_mse=val_mse=0.5353.ckpt"
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
    
    # 使用训练集数据测试
    dataset_info = DATASETS['PRAD']
    config = AddictDict({
        'data_path': dataset_info['path'],
        'slide_val': dataset_info['val_slides'],  # 使用训练数据
        'encoder_name': dataset_info['recommended_encoder'],
        'use_augmented': False,
        'expr_name': 'PRAD',
        'MODEL': AddictDict({'model_name': 'TWO_STAGE_VAR_ST'}),
        'DATA': {
            'normalize': True,
            'val_dataloader': {
                'batch_size': 32,
                'num_workers': 0,
                'shuffle': True,
                'persistent_workers': False
            }
        }
    })
    
    data_interface = DataInterface(config)
    data_interface.setup(stage='fit')
    train_dataloader = data_interface.val_dataloader()  # 使用验证集作为训练数据的代表
    
    print("📈 分析训练数据的token分布...")
    
    all_tokens = {scale: [] for scale in ['global', 'pathway', 'module', 'individual']}
    
    with torch.no_grad():
        for i, batch in enumerate(train_dataloader):
            if i >= 30:  # 更多批次
                break
            
            gene_expression = batch['target_genes'].cuda()
            encode_result = vqvae.encode(gene_expression)
            tokens = encode_result['tokens']
            
            for scale in ['global', 'pathway', 'module', 'individual']:
                all_tokens[scale].append(tokens[scale].cpu())
            
            if i % 10 == 0:
                print(f"   处理训练批次 {i}/30")
    
    print(f"\n📊 训练数据Token利用统计:")
    for scale in ['global', 'pathway', 'module', 'individual']:
        all_scale_tokens = torch.cat(all_tokens[scale], dim=0).flatten().numpy()
        unique_tokens = len(set(all_scale_tokens))
        
        print(f"\n{scale.upper()}尺度 (训练数据):")
        print(f"   唯一token数: {unique_tokens}/4096")
        print(f"   利用率: {unique_tokens/4096:.4f}")
        
        if unique_tokens < 100:
            print(f"   ❌ 严重欠利用！只使用了{unique_tokens}个codebook向量")
        elif unique_tokens < 500:
            print(f"   ⚠️ 利用不足，大量codebook向量未被使用")
        else:
            print(f"   ✅ 合理利用codebook")

if __name__ == "__main__":
    analyze_stage1_token_distribution()
    compare_stage1_vs_var_tokens()
    test_codebook_utilization_during_training() 