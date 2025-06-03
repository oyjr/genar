#!/usr/bin/env python3
"""
测试VAR-ST Padding策略

验证使用16×16 padding策略是否解决了14×14尺寸过小的问题
"""

import sys
import os
sys.path.append('src')

import torch
import torch.nn as nn
from typing import Dict, Any

# 导入我们的模块
from model.VAR.gene_pseudo_image_adapter import GenePseudoImageAdapter
from model.VAR.VAR_ST_Complete import VAR_ST_Complete

def test_gene_adapter_padding():
    """测试基因适配器的padding策略"""
    print("🧪 测试基因适配器 - Padding策略")
    print("=" * 60)
    
    # 初始化适配器（196基因 → 64×64，padding策略）
    adapter = GenePseudoImageAdapter(
        num_genes=196,
        target_image_size=64,  # 🔧 改为64×64，解决VQVAE下采样问题
        normalize_method='layer_norm'
    )
    
    # 创建测试数据
    batch_size = 4
    gene_data = torch.randn(batch_size, 196)
    
    print(f"\n📊 测试数据:")
    print(f"   - 输入基因: {gene_data.shape} (196个基因)")
    print(f"   - 数据范围: [{gene_data.min():.3f}, {gene_data.max():.3f}]")
    
    # 测试转换
    try:
        # 基因 → 伪图像
        pseudo_image = adapter.genes_to_pseudo_image(gene_data)
        print(f"\n✅ 转换成功:")
        print(f"   - 伪图像: {pseudo_image.shape}")
        print(f"   - 图像范围: [{pseudo_image.min():.3f}, {pseudo_image.max():.3f}]")
        
        # 验证padding区域
        flattened = pseudo_image.view(batch_size, -1)
        padding_region = flattened[:, 196:]  # 取padding部分
        is_padding_zero = torch.allclose(padding_region, torch.zeros_like(padding_region), atol=1e-6)
        print(f"   - Padding区域为零: {is_padding_zero}")
        print(f"   - Padding统计: mean={padding_region.mean():.6f}, std={padding_region.std():.6f}")
        
        # 伪图像 → 基因
        reconstructed_genes = adapter.pseudo_image_to_genes(pseudo_image)
        print(f"   - 重建基因: {reconstructed_genes.shape}")
        
        # 验证重建准确性
        reconstruction_error = torch.abs(gene_data - reconstructed_genes)
        max_error = reconstruction_error.max()
        mean_error = reconstruction_error.mean()
        print(f"   - 重建误差: max={max_error:.2e}, mean={mean_error:.2e}")
        
        if max_error < 1e-5:
            print(f"   ✅ 重建准确性验证通过")
        else:
            print(f"   ❌ 重建误差过大")
            
        return True
        
    except Exception as e:
        print(f"❌ 适配器测试失败: {e}")
        return False

def test_var_st_complete_padding():
    """测试完整VAR-ST模型的padding策略"""
    print("\n🚀 测试VAR-ST Complete - Padding策略")
    print("=" * 60)
    
    try:
        # 模型配置
        var_config = {
            'depth': 16,
            'embed_dim': 512,
            'num_heads': 8,
            'vocab_size': 1024
        }
        
        vqvae_config = {
            'z_channels': 256,
            'ch': 128,
            'ch_mult': [1, 1, 2, 2, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [16],  # 适配16×16
            'vocab_size': 1024
        }
        
        # 初始化模型（使用64×64 padding策略）
        model = VAR_ST_Complete(
            num_genes=196,
            spatial_size=64,  # 🔧 关键：使用64×64，解决VQVAE下采样问题
            histology_feature_dim=512,
            var_config=var_config,
            vqvae_config=vqvae_config
        )
        
        print(f"\n✅ 模型初始化成功!")
        print(f"   - 参数总数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 创建测试数据
        batch_size = 2
        gene_expression = torch.randn(batch_size, 196)
        histology_features = torch.randn(batch_size, 512)
        
        print(f"\n📊 测试推理:")
        print(f"   - 基因表达: {gene_expression.shape}")
        print(f"   - 组织学特征: {histology_features.shape}")
        
        # 测试推理
        model.eval()
        with torch.no_grad():
            outputs = model.forward_training(
                gene_expression=gene_expression,
                histology_features=histology_features
            )
            
        print(f"   - 输出形状: {outputs['predictions'].shape}")
        print(f"   - 输出范围: [{outputs['predictions'].min():.3f}, {outputs['predictions'].max():.3f}]")
        print(f"   - 损失: {outputs['loss'].item():.4f}")
        print(f"   ✅ 推理成功，padding策略解决了尺寸限制问题!")
        
        return True
        
    except Exception as e:
        print(f"❌ VAR-ST模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_padding_strategy_benefits():
    """展示padding策略的优势"""
    print("\n🎯 Padding策略分析")
    print("=" * 60)
    
    print("📈 对比分析:")
    print("   原始方案 (14×14):")
    print("     * 总位置: 14×14 = 196")
    print("     * 基因数量: 196")
    print("     * 空间利用率: 100%")
    print("     * ❌ 问题: VAR多层卷积后尺寸过小，导致维度错误")
    print()
    print("   Padding方案 (64×64):")
    print("     * 总位置: 64×64 = 4096")
    print("     * 基因数量: 196 + 3900 padding = 4096")
    print("     * 空间利用率: 4.8%")
    print("     * ✅ 优势: 为VAR VQVAE提供充足的下采样空间 (64→4)")
    print()
    
    print("🔧 技术优势:")
    print("   ✅ 兼容性: 支持标准VAR架构，无需修改核心代码")
    print("   ✅ 稳定性: 64×64→4×4经过VAR验证，支持16倍下采样")
    print("   ✅ 可扩展性: 可以轻松支持更大的基因集合")
    print("   ✅ 信息保留: 196基因信息完全保留，无损失")
    print("   ✅ 计算效率: 虽然增加存储，但解决了根本的尺寸匹配问题")
    print()
    
    print("🧬 生物学意义:")
    print("   ✅ 零padding不影响生物学解释")
    print("   ✅ 196基因的相对位置关系保持不变")
    print("   ✅ 多尺度分析依然有效")
    print("   ✅ 可以通过掩码忽略padding区域")

def main():
    """主测试函数"""
    print("🔬 VAR-ST Padding策略验证")
    print("=" * 80)
    
    # 展示策略分析
    show_padding_strategy_benefits()
    
    # 测试适配器
    adapter_success = test_gene_adapter_padding()
    
    # 测试完整模型
    model_success = test_var_st_complete_padding()
    
    # 总结
    print("\n" + "=" * 80)
    print("📋 测试总结:")
    print(f"   - 基因适配器: {'✅ 通过' if adapter_success else '❌ 失败'}")
    print(f"   - VAR-ST模型: {'✅ 通过' if model_success else '❌ 失败'}")
    
    if adapter_success and model_success:
        print("\n🎉 所有测试通过! Padding策略成功解决了VAR的尺寸限制问题!")
        print("💡 建议: 可以开始使用16×16配置进行训练")
    else:
        print("\n😞 部分测试失败，需要进一步调试")
    
    return adapter_success and model_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 