#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同编码器维度适配
验证STDataset和MFBP模型是否能正确处理UNI(1024维)和CONCH(512维)编码器
"""

import os
import sys
import torch
import tempfile
from pathlib import Path

# 添加项目路径
sys.path.insert(0, 'src')

from dataset.hest_dataset import STDataset
from model.MFBP.MFBP import MFBP
from utils import load_config


def create_test_embedding(slide_id, encoder_name, num_spots=10, is_augmented=False, is_3d=False):
    """创建测试嵌入文件"""
    feature_dim = 1024 if encoder_name == 'uni' else 512
    
    if is_3d:
        # 创建3D格式: [num_spots, num_patches, feature_dim]
        emb = torch.randn(num_spots, 7, feature_dim)
    else:
        # 创建2D格式: [num_spots, feature_dim]
        emb = torch.randn(num_spots, feature_dim)
    
    return emb


def test_encoder_dimensions():
    """测试不同编码器维度适配"""
    print("🧪 开始测试不同编码器维度适配")
    
    # 创建临时测试目录
    with tempfile.TemporaryDirectory() as temp_dir:
        test_data_path = f"{temp_dir}/test_dataset/"
        
        # 创建目录结构
        os.makedirs(f"{test_data_path}st", exist_ok=True)
        os.makedirs(f"{test_data_path}processed_data", exist_ok=True)
        os.makedirs(f"{test_data_path}processed_data/1spot_uni_ebd", exist_ok=True)
        os.makedirs(f"{test_data_path}processed_data/1spot_conch_ebd", exist_ok=True)
        os.makedirs(f"{test_data_path}processed_data/1spot_uni_ebd_aug", exist_ok=True)
        os.makedirs(f"{test_data_path}processed_data/1spot_conch_ebd_aug", exist_ok=True)
        
        # 创建测试文件
        slide_id = "TEST001"
        
        # 创建基因列表
        genes = [f"GENE{i}" for i in range(50)]
        with open(f"{test_data_path}processed_data/selected_gene_list.txt", 'w') as f:
            f.write('\n'.join(genes))
        
        # 创建slide列表
        with open(f"{test_data_path}processed_data/all_slide_lst.txt", 'w') as f:
            f.write(slide_id)
        
        # 创建ST数据文件 (使用pandas创建简单的h5ad文件)
        import scanpy as sc
        import anndata as ad
        import numpy as np
        
        num_spots = 10
        num_genes = len(genes)
        
        # 创建AnnData对象
        X = np.random.randn(num_spots, num_genes)
        obs = {'array_row': np.arange(num_spots), 'array_col': np.arange(num_spots)}
        var = {'gene_names': genes}
        spatial = np.random.rand(num_spots, 2)
        
        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.var_names = genes
        adata.obsm['spatial'] = spatial
        
        # 保存ST数据
        adata.write_h5ad(f"{test_data_path}st/{slide_id}.h5ad")
        
        # 测试1: UNI编码器 (1024维)
        print("\n📊 测试1: UNI编码器 (1024维)")
        
        # 创建UNI嵌入文件 (2D格式)
        uni_emb_2d = create_test_embedding(slide_id, 'uni', num_spots, is_3d=False)
        torch.save(uni_emb_2d, f"{test_data_path}processed_data/1spot_uni_ebd/{slide_id}_uni.pt")
        print(f"  ✅ 创建UNI 2D嵌入: {uni_emb_2d.shape}")
        
        # 创建UNI增强嵌入文件 (3D格式)
        uni_emb_3d = create_test_embedding(slide_id, 'uni', num_spots, is_augmented=True, is_3d=True)
        torch.save(uni_emb_3d, f"{test_data_path}processed_data/1spot_uni_ebd_aug/{slide_id}_uni_aug.pt")
        print(f"  ✅ 创建UNI 3D增强嵌入: {uni_emb_3d.shape}")
        
        # 测试STDataset with UNI
        try:
            uni_dataset = STDataset(
                mode='test',
                data_path=test_data_path,
                expr_name='test',
                encoder_name='uni',
                use_augmented=False
            )
            print("  ✅ STDataset UNI标准模式初始化成功")
            
            # 测试加载嵌入
            emb = uni_dataset.load_emb(slide_id)
            assert emb.shape == (num_spots, 1024), f"UNI嵌入维度错误: {emb.shape}"
            print(f"  ✅ UNI嵌入加载成功: {emb.shape}")
            
        except Exception as e:
            print(f"  ❌ STDataset UNI测试失败: {e}")
            return False
        
        # 测试UNI增强模式
        try:
            uni_aug_dataset = STDataset(
                mode='test',
                data_path=test_data_path,
                expr_name='test',
                encoder_name='uni',
                use_augmented=True
            )
            print("  ✅ STDataset UNI增强模式初始化成功")
            
            # 测试加载3D嵌入 (应该自动取平均)
            emb_aug = uni_aug_dataset.load_emb(slide_id)
            assert emb_aug.shape == (num_spots, 1024), f"UNI增强嵌入维度错误: {emb_aug.shape}"
            print(f"  ✅ UNI增强嵌入加载成功 (3D->2D): {emb_aug.shape}")
            
        except Exception as e:
            print(f"  ❌ STDataset UNI增强测试失败: {e}")
            return False
        
        # 测试2: CONCH编码器 (512维)
        print("\n📊 测试2: CONCH编码器 (512维)")
        
        # 创建CONCH嵌入文件
        conch_emb_2d = create_test_embedding(slide_id, 'conch', num_spots, is_3d=False)
        torch.save(conch_emb_2d, f"{test_data_path}processed_data/1spot_conch_ebd/{slide_id}_conch.pt")
        print(f"  ✅ 创建CONCH 2D嵌入: {conch_emb_2d.shape}")
        
        # 创建CONCH增强嵌入文件 (3D格式)
        conch_emb_3d = create_test_embedding(slide_id, 'conch', num_spots, is_augmented=True, is_3d=True)
        torch.save(conch_emb_3d, f"{test_data_path}processed_data/1spot_conch_ebd_aug/{slide_id}_conch_aug.pt")
        print(f"  ✅ 创建CONCH 3D增强嵌入: {conch_emb_3d.shape}")
        
        # 测试STDataset with CONCH
        try:
            conch_dataset = STDataset(
                mode='test',
                data_path=test_data_path,
                expr_name='test',
                encoder_name='conch',
                use_augmented=False
            )
            print("  ✅ STDataset CONCH标准模式初始化成功")
            
            # 测试加载嵌入
            emb = conch_dataset.load_emb(slide_id)
            assert emb.shape == (num_spots, 512), f"CONCH嵌入维度错误: {emb.shape}"
            print(f"  ✅ CONCH嵌入加载成功: {emb.shape}")
            
        except Exception as e:
            print(f"  ❌ STDataset CONCH测试失败: {e}")
            return False
        
        # 测试CONCH增强模式
        try:
            conch_aug_dataset = STDataset(
                mode='test',
                data_path=test_data_path,
                expr_name='test',
                encoder_name='conch',
                use_augmented=True
            )
            print("  ✅ STDataset CONCH增强模式初始化成功")
            
            # 测试加载3D嵌入 (应该自动取平均)
            emb_aug = conch_aug_dataset.load_emb(slide_id)
            assert emb_aug.shape == (num_spots, 512), f"CONCH增强嵌入维度错误: {emb_aug.shape}"
            print(f"  ✅ CONCH增强嵌入加载成功 (3D->2D): {emb_aug.shape}")
            
        except Exception as e:
            print(f"  ❌ STDataset CONCH增强测试失败: {e}")
            return False
        
        # 测试3: MFBP模型维度适配
        print("\n📊 测试3: MFBP模型维度适配")
        
        # 测试UNI模型
        try:
            # 创建配置对象
            config = load_config('config/hest/base_config.yaml')
            config.MODEL.feature_dim = 1024
            config.MODEL.num_genes = num_genes
            
            uni_model = MFBP(config=config)
            
            # 测试前向传播
            test_input_uni = torch.randn(1, num_spots, 1024)
            output_uni = uni_model(test_input_uni)
            
            assert output_uni['logits'].shape == (1, num_spots, num_genes), f"UNI模型输出维度错误: {output_uni['logits'].shape}"
            print(f"  ✅ UNI模型测试成功: 输入{test_input_uni.shape} -> 输出{output_uni['logits'].shape}")
            
        except Exception as e:
            print(f"  ❌ UNI模型测试失败: {e}")
            return False
        
        # 测试CONCH模型
        try:
            # 更新配置为CONCH
            config.MODEL.feature_dim = 512
            
            conch_model = MFBP(config=config)
            
            # 测试前向传播
            test_input_conch = torch.randn(1, num_spots, 512)
            output_conch = conch_model(test_input_conch)
            
            assert output_conch['logits'].shape == (1, num_spots, num_genes), f"CONCH模型输出维度错误: {output_conch['logits'].shape}"
            print(f"  ✅ CONCH模型测试成功: 输入{test_input_conch.shape} -> 输出{output_conch['logits'].shape}")
            
        except Exception as e:
            print(f"  ❌ CONCH模型测试失败: {e}")
            return False
    
    return True


def main():
    """主测试函数"""
    print("🚀 开始编码器维度适配测试")
    
    success = test_encoder_dimensions()
    
    print("\n" + "="*50)
    if success:
        print("🎉 所有测试通过！编码器维度适配成功！")
        print("\n✅ 已支持的编码器维度:")
        print("  - UNI: 1024维 (2D和3D格式)")
        print("  - CONCH: 512维 (2D和3D格式)")
        print("  - 自动检测和处理3D->2D维度转换")
        print("  - 动态模型参数适配")
    else:
        print("❌ 测试失败！请检查代码修改")
        sys.exit(1)


if __name__ == '__main__':
    main() 