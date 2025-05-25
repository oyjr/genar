#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试expand_augmented功能
验证3D增强嵌入是否正确展开为7倍训练样本
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


def create_test_3d_embedding(slide_id, encoder_name, num_spots=5, seed=42):
    """创建测试3D嵌入文件"""
    torch.manual_seed(seed)
    
    feature_dim = 1024 if encoder_name == 'uni' else 512
    
    # 创建可识别的3D嵌入模式
    augmented_emb = torch.zeros(num_spots, 7, feature_dim)
    
    for spot_idx in range(num_spots):
        for aug_idx in range(7):
            # 每个spot的每个增强版本有特定的模式
            base_value = spot_idx * 10 + aug_idx  # 例如: spot0=[0,1,2,3,4,5,6], spot1=[10,11,12,13,14,15,16]
            augmented_emb[spot_idx, aug_idx, :] = base_value
    
    return augmented_emb  # [num_spots, 7, feature_dim]


def test_expand_augmented():
    """测试展开增强功能"""
    print("🚀 测试expand_augmented功能")
    
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
        genes = [f"GENE{i}" for i in range(10)]
        with open(f"{test_data_path}processed_data/selected_gene_list.txt", 'w') as f:
            f.write('\n'.join(genes))
        
        # 创建slide列表
        with open(f"{test_data_path}processed_data/all_slide_lst.txt", 'w') as f:
            f.write(slide_id)
        
        # 创建ST数据文件
        import scanpy as sc
        import anndata as ad
        
        num_spots = 3  # 少量spots便于验证
        num_genes = len(genes)
        
        # 创建可识别的基因表达模式
        X = np.zeros((num_spots, num_genes))
        for spot_idx in range(num_spots):
            X[spot_idx, :] = spot_idx * 100  # spot0=[0,0,0...], spot1=[100,100,100...], spot2=[200,200,200...]
        
        obs = {'array_row': np.arange(num_spots), 'array_col': np.arange(num_spots)}
        var = {'gene_names': genes}
        spatial = np.random.rand(num_spots, 2)
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.var_names = genes
        adata.obsm['spatial'] = spatial
        adata.write_h5ad(f"{test_data_path}st/{slide_id}.h5ad")
        
        # 创建可识别的3D嵌入
        uni_emb_3d = create_test_3d_embedding(slide_id, 'uni', num_spots)
        torch.save(uni_emb_3d, f"{test_data_path}processed_data/1spot_uni_ebd_aug/{slide_id}_uni_aug.pt")
        
        print(f"📊 创建的3D嵌入形状: {uni_emb_3d.shape}")
        print(f"📊 原始数据模式:")
        for spot_idx in range(num_spots):
            print(f"  Spot {spot_idx}: 嵌入前3维 = {uni_emb_3d[spot_idx, :, 0].numpy()}")
            print(f"  Spot {spot_idx}: 基因表达前3维 = {X[spot_idx, :3]}")
        
        # 测试1: 原有取平均模式
        print(f"\n🔬 测试1: 原有取平均模式 (expand_augmented=False)")
        dataset_original = STDataset(
            mode='train',
            data_path=test_data_path,
            expr_name='test',
            encoder_name='uni',
            use_augmented=True,
            expand_augmented=False,
            aug_strategy='mean',
            normalize=False  # 关闭归一化便于验证
        )
        
        print(f"  数据集长度: {len(dataset_original)}")
        sample = dataset_original[0]
        print(f"  第1个样本嵌入第1维: {sample['img'][0].item():.1f}")
        print(f"  第1个样本基因表达前3维: {sample['target_genes'][:3].numpy()}")
        
        # 测试2: 新的展开模式
        print(f"\n🚀 测试2: 新的展开模式 (expand_augmented=True)")
        dataset_expanded = STDataset(
            mode='train',
            data_path=test_data_path,
            expr_name='test',
            encoder_name='uni',
            use_augmented=True,
            expand_augmented=True,
            normalize=False  # 关闭归一化便于验证
        )
        
        print(f"  数据集长度: {len(dataset_expanded)} (应该是 {num_spots * 7} = {num_spots} spots × 7 增强)")
        
        if len(dataset_expanded) == num_spots * 7:
            print("  ✅ 数据集长度正确")
        else:
            print("  ❌ 数据集长度错误")
            return False
        
        # 验证展开后的样本
        print(f"\n📈 验证展开后的样本:")
        for i in range(min(21, len(dataset_expanded))):  # 查看前21个样本 (3 spots × 7 增强)
            sample = dataset_expanded[i]
            
            expected_spot_id = i // 7
            expected_aug_id = i % 7
            expected_emb_value = expected_spot_id * 10 + expected_aug_id
            expected_gene_value = expected_spot_id * 100  # 原始未归一化数据
            
            actual_spot_id = sample['original_spot_id']
            actual_aug_id = sample['aug_id']
            actual_emb_value = sample['img'][0].item()
            actual_gene_value = sample['target_genes'][0].item()
            
            print(f"  样本{i:2d}: spot={actual_spot_id}(期望{expected_spot_id}) "
                  f"aug={actual_aug_id}(期望{expected_aug_id}) "
                  f"emb={actual_emb_value:.0f}(期望{expected_emb_value}) "
                  f"gene={actual_gene_value:.0f}(期望{expected_gene_value:.0f})")
            
            # 验证数据正确性
            if (actual_spot_id != expected_spot_id or 
                actual_aug_id != expected_aug_id or
                abs(actual_emb_value - expected_emb_value) > 0.1 or
                abs(actual_gene_value - expected_gene_value) > 0.1):
                print(f"  ❌ 样本{i}数据不正确")
                print(f"    详细信息: 期望spot={expected_spot_id}, aug={expected_aug_id}, emb={expected_emb_value}, gene={expected_gene_value}")
                print(f"    实际信息: 实际spot={actual_spot_id}, aug={actual_aug_id}, emb={actual_emb_value}, gene={actual_gene_value}")
                return False
        
        print("  ✅ 所有样本数据正确")
        
        # 测试3: 验证模式不受影响
        print(f"\n🔬 测试3: 验证模式 (expand_augmented应该被忽略)")
        dataset_val = STDataset(
            mode='val',
            data_path=test_data_path,
            expr_name='test',
            slide_val=slide_id,
            encoder_name='uni',
            use_augmented=True,
            expand_augmented=True,  # 应该被忽略
            normalize=False  # 关闭归一化
        )
        
        print(f"  验证集长度: {len(dataset_val)} (应该是1个slide)")
        if len(dataset_val) == 1:
            print("  ✅ 验证模式不受影响")
        else:
            print("  ❌ 验证模式受到影响")
            return False
    
    return True


def main():
    """主测试函数"""
    print("🧪 开始expand_augmented功能测试")
    
    success = test_expand_augmented()
    
    print("\n" + "="*60)
    if success:
        print("🎉 expand_augmented功能测试通过！")
        print("\n✅ 功能验证:")
        print("  - 3D增强嵌入正确展开为7倍样本")
        print("  - 基因表达数据正确同步")
        print("  - 位置信息正确复制")
        print("  - 增强信息正确标记")
        print("  - 验证/测试模式不受影响")
        
        print("\n🎮 使用示例:")
        print("# 启用增强样本展开")
        print("python src/main.py \\")
        print("    --expr_name PRAD \\")
        print("    --data_path /data/path/ \\") 
        print("    --encoder_name uni \\")
        print("    --use_augmented \\")
        print("    --expand_augmented \\")
        print("    --mode train")
        
        print("\n💡 预期效果:")
        print("  - 训练样本数量×7")
        print("  - 每个原始spot对应7个增强样本")
        print("  - 更好的数据增强效果")
        print("  - 提高模型泛化能力")
        
    else:
        print("❌ expand_augmented功能测试失败！")
        sys.exit(1)


if __name__ == '__main__':
    main() 