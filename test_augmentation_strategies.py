#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同3D增强嵌入处理策略
演示random、mean、attention、first、all策略的效果差异
"""

import os
import sys
import torch
import tempfile
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, 'src')

from dataset.hest_dataset import STDataset
from utils import load_config


def create_test_3d_embedding(slide_id, encoder_name, num_spots=10, seed=42):
    """创建测试3D嵌入文件，模拟7种增强变换"""
    torch.manual_seed(seed)
    
    feature_dim = 1024 if encoder_name == 'uni' else 512
    
    # 创建具有特定模式的3D嵌入，便于观察策略差异
    base_emb = torch.randn(num_spots, feature_dim) * 0.5  # 基础特征
    
    augmented_emb = torch.zeros(num_spots, 7, feature_dim)
    
    for i in range(7):
        # 为每个增强版本添加不同的变化模式
        if i == 0:
            # 第一个是原图（变化最小）
            augmented_emb[:, i, :] = base_emb + torch.randn(num_spots, feature_dim) * 0.1
        else:
            # 其他增强版本有更大变化
            noise_scale = 0.2 + i * 0.1  # 递增噪声
            augmented_emb[:, i, :] = base_emb + torch.randn(num_spots, feature_dim) * noise_scale
    
    return augmented_emb  # [num_spots, 7, feature_dim]


def analyze_strategy_differences():
    """分析不同策略的差异"""
    print("🔍 分析不同增强策略的差异")
    
    # 创建临时测试目录
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_path = f"{temp_dir}/test_dataset/"
        
        # 创建目录结构
        os.makedirs(f"{test_data_path}st", exist_ok=True)
        os.makedirs(f"{test_data_path}processed_data", exist_ok=True)
        os.makedirs(f"{test_data_path}processed_data/1spot_uni_ebd_aug", exist_ok=True)
        
        # 创建测试文件
        slide_id = "TEST001"
        
        # 创建基因列表
        genes = [f"GENE{i}" for i in range(50)]
        with open(f"{test_data_path}processed_data/selected_gene_list.txt", 'w') as f:
            f.write('\n'.join(genes))
        
        # 创建slide列表
        with open(f"{test_data_path}processed_data/all_slide_lst.txt", 'w') as f:
            f.write(slide_id)
        
        # 创建ST数据文件
        import scanpy as sc
        import anndata as ad
        
        num_spots = 5  # 减少spot数量便于观察
        num_genes = len(genes)
        
        X = np.random.randn(num_spots, num_genes)
        obs = {'array_row': np.arange(num_spots), 'array_col': np.arange(num_spots)}
        var = {'gene_names': genes}
        spatial = np.random.rand(num_spots, 2)
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.var_names = genes
        adata.obsm['spatial'] = spatial
        adata.write_h5ad(f"{test_data_path}st/{slide_id}.h5ad")
        
        # 创建具有特定模式的3D嵌入
        uni_emb_3d = create_test_3d_embedding(slide_id, 'uni', num_spots)
        torch.save(uni_emb_3d, f"{test_data_path}processed_data/1spot_uni_ebd_aug/{slide_id}_uni_aug.pt")
        
        print(f"📊 创建的3D嵌入形状: {uni_emb_3d.shape}")
        print(f"📊 前2个spot的7个增强版本的第一个特征维度:")
        for spot_idx in range(min(2, num_spots)):
            print(f"  Spot {spot_idx}: {uni_emb_3d[spot_idx, :, 0].numpy()}")
        
        # 测试所有策略
        strategies = ['random', 'mean', 'attention', 'first', 'all']
        results = {}
        
        for strategy in strategies:
            print(f"\n🔬 测试策略: {strategy}")
            
            try:
                dataset = STDataset(
                    mode='test',
                    data_path=test_data_path,
                    expr_name='test',
                    encoder_name='uni',
                    use_augmented=True,
                    aug_strategy=strategy
                )
                
                # 加载第一个spot的嵌入
                if strategy == 'all':
                    emb = dataset.load_emb(slide_id, 0, strategy)  # [7, 1024]
                    results[strategy] = emb
                    print(f"  ✅ 输出形状: {emb.shape}")
                    print(f"  📊 7个增强版本的第一个特征: {emb[:, 0].numpy()}")
                else:
                    emb = dataset.load_emb(slide_id, 0, strategy)  # [1024]
                    results[strategy] = emb
                    print(f"  ✅ 输出形状: {emb.shape}")
                    print(f"  📊 第一个特征值: {emb[0].item():.6f}")
                
            except Exception as e:
                print(f"  ❌ 策略 {strategy} 测试失败: {e}")
                return False
        
        # 比较不同策略的结果
        print(f"\n📈 策略比较分析:")
        print(f"  - 原始3D数据: spot 0的7个增强版本第一特征 = {uni_emb_3d[0, :, 0].numpy()}")
        
        if 'first' in results:
            print(f"  - 'first'策略结果: {results['first'][0].item():.6f} (应该等于原始第0个增强)")
            
        if 'mean' in results:
            expected_mean = uni_emb_3d[0, :, 0].mean().item()
            print(f"  - 'mean'策略结果: {results['mean'][0].item():.6f}")
            print(f"  - 期望平均值: {expected_mean:.6f}")
            
        # 运行多次random策略看随机性
        print(f"\n🎲 'random'策略的随机性测试 (spot 0, 第一特征):")
        random_results = []
        for i in range(5):
            dataset_random = STDataset(
                mode='test',
                data_path=test_data_path,
                expr_name='test',
                encoder_name='uni',
                use_augmented=True,
                aug_strategy='random'
            )
            emb = dataset_random.load_emb(slide_id, 0, 'random')
            random_results.append(emb[0].item())
            print(f"  运行 {i+1}: {emb[0].item():.6f}")
        
        print(f"  随机结果变化范围: [{min(random_results):.6f}, {max(random_results):.6f}]")
    
    return True


def test_training_impact():
    """测试不同策略对训练的潜在影响"""
    print("\n💡 不同策略的优缺点分析:")
    
    strategies_info = {
        'random': {
            '优点': ['保持数据增强的多样性', '每次训练看到不同变换', '有利于泛化'],
            '缺点': ['增加训练随机性', '可能需要更多epochs收敛'],
            '推荐': '✅ 推荐用于训练，特别是数据量不大时'
        },
        'mean': {
            '优点': ['训练稳定', '减少噪声', '保持接口一致'],
            '缺点': ['丢失增强多样性', '可能欠拟合', '违背增强目的'],
            '推荐': '❌ 不推荐，除非需要稳定性'
        },
        'attention': {
            '优点': ['自适应权重', '保留重要信息', '比mean更智能'],
            '缺点': ['计算稍复杂', '仍有信息丢失'],
            '推荐': '🔀 适中选择，平衡性能和稳定性'
        },
        'first': {
            '优点': ['使用原图', '无增强噪声', '训练稳定'],
            '缺点': ['完全浪费增强数据', '没有数据多样性'],
            '推荐': '❌ 不推荐，除非测试baseline'
        },
        'all': {
            '优点': ['保留所有信息', '可用于特殊模型'],
            '缺点': ['需要修改模型结构', '计算开销大'],
            '推荐': '🚀 高级用法，需要定制化开发'
        }
    }
    
    for strategy, info in strategies_info.items():
        print(f"\n📋 {strategy.upper()}策略:")
        print(f"  优点: {', '.join(info['优点'])}")
        print(f"  缺点: {', '.join(info['缺点'])}")
        print(f"  推荐: {info['推荐']}")


def main():
    """主测试函数"""
    print("🚀 开始3D增强嵌入策略测试")
    
    success = analyze_strategy_differences()
    
    if success:
        test_training_impact()
        
        print("\n" + "="*60)
        print("🎯 推荐使用方案:")
        print("  1. 🥇 训练阶段: --aug_strategy random")
        print("     └─ 保持数据增强多样性，提高泛化能力")
        print("  2. 🥈 验证阶段: --aug_strategy attention 或 first")
        print("     └─ 稳定的推理结果，便于模型评估")
        print("  3. 🥉 对比实验: --aug_strategy mean")
        print("     └─ 与原方案对比，看取平均的影响")
        
        print("\n🎮 使用示例:")
        print("# 推荐训练命令")
        print("python src/main.py \\")
        print("    --expr_name PRAD \\")
        print("    --data_path /data/path/ \\") 
        print("    --encoder_name uni \\")
        print("    --use_augmented \\")
        print("    --aug_strategy random \\")
        print("    --mode train")
        
    else:
        print("❌ 测试失败！")
        sys.exit(1)


if __name__ == '__main__':
    main() 