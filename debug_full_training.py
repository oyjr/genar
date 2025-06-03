#!/usr/bin/env python
import sys
sys.path.insert(0, 'src')
import torch
from main import build_config_from_args
from model import ModelInterface

class Args:
    dataset = 'PRAD'
    model = 'VAR_ST'
    encoder = None
    gpus = 1
    epochs = 1
    batch_size = 2
    lr = None
    weight_decay = None
    patience = None
    strategy = 'auto'
    sync_batchnorm = False
    use_augmented = True
    expand_augmented = True
    mode = 'train'
    seed = None
    config = None

print("=== 🔧 完整训练流程调试 ===")

# Step 1: 构建配置
args = Args()
config = build_config_from_args(args)

print(f"\n=== Step 1: 配置检查 ===")
print(f"spatial_size: {config.MODEL.spatial_size}")
print(f"feature_dim: {config.MODEL.feature_dim}")

# Step 2: 创建ModelInterface
print(f"\n=== Step 2: 创建ModelInterface ===")
model_interface = ModelInterface(config)

print(f"\n=== Step 3: 检查实际模型 ===")
actual_model = model_interface.model
print(f"实际模型类型: {type(actual_model)}")
print(f"实际模型的histology_feature_dim: {actual_model.histology_feature_dim}")
print(f"实际模型的spatial_size: {actual_model.spatial_size}")

print(f"\n=== Step 4: 检查VARGeneWrapper ===")
var_wrapper = actual_model.var_gene_wrapper
print(f"VARGeneWrapper的histology_feature_dim: {var_wrapper.histology_feature_dim}")
print(f"VARGeneWrapper的image_size: {var_wrapper.image_size}")

print(f"\n=== Step 5: 检查条件处理器 ===")
condition_processor = var_wrapper.condition_processor
first_layer = condition_processor[0]
print(f"条件处理器第一层权重形状: {first_layer.weight.shape}")
print(f"期望输入维度: {first_layer.weight.shape[1]}")

# Step 6: 模拟真实的前向传播
print(f"\n=== Step 6: 模拟前向传播 ===")
batch_size = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建模拟数据
gene_expression = torch.randn(batch_size, 196, device=device)
histology_features = torch.randn(batch_size, 1024, device=device)  # 1024维特征

print(f"gene_expression形状: {gene_expression.shape}")
print(f"histology_features形状: {histology_features.shape}")

# 移动模型到设备
actual_model = actual_model.to(device)

# 尝试前向传播
try:
    print(f"\n=== 尝试前向传播 ===")
    with torch.no_grad():
        outputs = actual_model.forward_training(
            gene_expression=gene_expression,
            histology_features=histology_features
        )
    print(f"✅ 前向传播成功！")
    print(f"输出类型: {type(outputs)}")
    if isinstance(outputs, dict):
        for key, value in outputs.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
except Exception as e:
    print(f"❌ 前向传播失败！")
    print(f"错误: {e}")
    import traceback
    traceback.print_exc() 