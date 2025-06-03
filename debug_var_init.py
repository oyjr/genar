#!/usr/bin/env python
import sys
sys.path.insert(0, 'src')
import torch

# 直接测试VAR_ST_Complete的初始化
print("=== 直接测试 VAR_ST_Complete 初始化 ===")

from model.VAR.VAR_ST_Complete import VAR_ST_Complete

# 使用正确的参数创建模型
model = VAR_ST_Complete(
    num_genes=196,
    spatial_size=64,  # 🔧 修复：使用64×64 padding策略，解决VQVAE下采样问题
    histology_feature_dim=None,  # 不传递这个参数
    feature_dim=1024  # 传递feature_dim=1024
)

print(f"\n=== 检查模型的histology_feature_dim ===")
print(f"model.histology_feature_dim: {model.histology_feature_dim}")

print(f"\n=== 检查VARGeneWrapper的histology_feature_dim ===") 
print(f"model.var_gene_wrapper.histology_feature_dim: {model.var_gene_wrapper.histology_feature_dim}")

print(f"\n=== 检查条件处理器的第一层权重形状 ===")
first_layer = model.var_gene_wrapper.condition_processor[0]
print(f"第一个线性层权重形状: {first_layer.weight.shape}")
print(f"期望输入维度: {first_layer.weight.shape[1]}")
print(f"期望输出维度: {first_layer.weight.shape[0]}")

# 测试一个1024维的输入
print(f"\n=== 测试1024维输入 ===")
test_input = torch.randn(1, 1024)
print(f"测试输入形状: {test_input.shape}")
try:
    output = model.var_gene_wrapper.condition_processor(test_input)
    print(f"✅ 成功！输出形状: {output.shape}")
except Exception as e:
    print(f"❌ 失败！错误: {e}") 