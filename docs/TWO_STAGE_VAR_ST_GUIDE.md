# Two-Stage VAR-ST模型使用指南

## 概述

Two-Stage VAR-ST是一个用于空间转录组学基因表达预测的两阶段深度学习模型。该模型结合了Vector-Quantized Variational AutoEncoder (VQVAE)和VAR Transformer，通过组织学特征和空间坐标预测基因表达。

## 模型架构

### Stage 1: Multi-Scale Gene VQVAE
- **目标**: 学习基因表达的离散表示
- **输入**: 基因表达数据 [B, 200]
- **输出**: 多尺度离散tokens (Global, Pathway, Module, Individual)
- **训练**: 仅使用基因表达数据，无监督学习

### Stage 2: Gene VAR Transformer  
- **目标**: 从组织学特征条件生成基因表达tokens
- **输入**: 组织学特征 [B, 1024] + 空间坐标 [B, 2]
- **输出**: 基因表达tokens序列 [B, 241]
- **训练**: 使用冻结的Stage 1 VQVAE，有监督学习

### 推理流程
组织学特征 + 空间坐标 → VAR Transformer → tokens → 冻结VQVAE → 基因表达预测

## 快速开始

### 1. Stage 1训练（VQVAE）

```bash
python src/main.py \
    --dataset PRAD \
    --model TWO_STAGE_VAR_ST \
    --training_stage 1 \
    --epochs 150 \
    --batch_size 256 \
    --lr 1e-4 \
    --gpus 1
```

### 2. Stage 2训练（VAR Transformer）

```bash
python src/main.py \
    --dataset PRAD \
    --model TWO_STAGE_VAR_ST \
    --training_stage 2 \
    --stage1_ckpt logs/PRAD/TWO_STAGE_VAR_ST/stage1_vqvae/best_vqvae.ckpt \
    --epochs 300 \
    --batch_size 64 \
    --lr 1e-4 \
    --gpus 1
```

## 配置参数

### 模型配置

```python
# Stage 1 VQVAE配置
vqvae_config = {
    'vocab_size': 4096,           # 词汇表大小
    'embed_dim': 128,             # 嵌入维度
    'beta': 0.25,                 # VQ损失权重
    'hierarchical_loss_weight': 0.1,  # 分层损失权重
    'vq_loss_weight': 0.25        # VQ损失权重
}

# Stage 2 VAR配置
var_config = {
    'vocab_size': 4096,           # 词汇表大小（与Stage 1一致）
    'embed_dim': 640,             # 嵌入维度
    'num_heads': 8,               # 注意力头数
    'num_layers': 12,             # Transformer层数
    'feedforward_dim': 2560,      # 前馈网络维度
    'dropout': 0.1,               # Dropout率
    'max_sequence_length': 1500,  # 最大序列长度
    'condition_embed_dim': 640    # 条件嵌入维度
}
```

### 训练建议

#### Stage 1 (VQVAE)
- **Batch Size**: 256-512 (根据GPU内存调整)
- **Learning Rate**: 1e-4
- **Epochs**: 100-200
- **优化器**: AdamW
- **监控指标**: Reconstruction Loss, VQ Loss, Codebook Utilization

#### Stage 2 (VAR Transformer)  
- **Batch Size**: 64-128 (VAR Transformer内存需求更大)
- **Learning Rate**: 1e-4
- **Epochs**: 200-400  
- **优化器**: AdamW
- **监控指标**: Cross-entropy Loss, Token Accuracy

## 数据流示例

### Stage 1数据流
```
基因表达 [64, 200] 
    ↓ Multi-scale Encoding
Global[64,1] + Pathway[64,8] + Module[64,32] + Individual[64,200]
    ↓ 统一量化 (vocab_size=4096)
Quantized Tokens [64, 241] 
    ↓ Multi-scale Decoding  
重建基因表达 [64, 200]
```

### Stage 2数据流
```
组织学特征[64,1024] + 空间坐标[64,2]
    ↓ 条件处理
条件嵌入[64,640]
    ↓ + 目标tokens[64,241] 
VAR Transformer自回归训练
    ↓ 输出
Logits[64,241,4096]
```

### 推理数据流
```
组织学特征[64,1024] + 空间坐标[64,2]
    ↓ VAR生成
生成tokens[64,241] 
    ↓ 冻结VQVAE重建
预测基因表达[64,200]
```

## 高级用法

### 1. 自定义配置训练

```python
import sys
sys.path.insert(0, 'src')

from model.VAR.two_stage_var_st import TwoStageVARST

# 自定义配置
custom_vqvae_config = {
    'vocab_size': 2048,
    'embed_dim': 64, 
    'beta': 0.5
}

custom_var_config = {
    'vocab_size': 2048,
    'embed_dim': 320,
    'num_heads': 4,
    'num_layers': 6
}

# 创建模型
model = TwoStageVARST(
    num_genes=200,
    vqvae_config=custom_vqvae_config,
    var_config=custom_var_config,
    current_stage=1
)
```

### 2. 模型推理

```python
# 加载完整模型
model = TwoStageVARST.load_complete_model(
    'path/to/complete_model.ckpt', 
    device='cuda'
)

# 推理
histology_features = torch.randn(1, 1024)  
spatial_coords = torch.randn(1, 2)

results = model.inference(
    histology_features=histology_features,
    spatial_coords=spatial_coords,
    temperature=1.0,
    top_k=50,
    top_p=0.9
)

predicted_genes = results['predicted_gene_expression']  # [1, 200]
```

### 3. 模型状态管理

```python
# 检查模型信息
info = model.get_model_info()
print(f"Total Parameters: {info['total_parameters']:,}")
print(f"Current Stage: {info['current_stage']}")

# 切换训练阶段
model.set_training_stage(2, stage1_ckpt_path='stage1.ckpt')

# 保存checkpoint
model.save_stage_checkpoint('stage1.ckpt', stage=1)
model.save_complete_model('complete_model.ckpt')
```

## 性能优化

### 1. 内存优化
- Stage 1: 使用较大的batch size（256-512）
- Stage 2: 使用较小的batch size（64-128）
- 使用梯度累积增加有效batch size

### 2. 训练加速  
- 使用多GPU并行训练（DDP）
- 混合精度训练（FP16）
- 数据加载器优化（num_workers, pin_memory）

### 3. 监控建议
- Stage 1: 监控codebook利用率，避免code collapse
- Stage 2: 监控token生成准确率和损失下降
- 使用TensorBoard可视化训练过程

## 故障排除

### 常见问题

1. **Stage 2训练时提示需要stage1_ckpt**
   - 确保Stage 1已完成训练并保存checkpoint
   - 检查checkpoint路径是否正确

2. **内存不足**
   - 减少batch size
   - 使用梯度累积
   - 检查序列长度设置

3. **收敛问题**
   - Stage 1: 检查VQ损失权重β
   - Stage 2: 调整学习率和dropout

4. **维度错误**
   - 确保组织学特征维度与编码器匹配
   - 检查基因数量设置（默认200）

### 调试模式

```bash
# 启用详细日志
export CUDA_LAUNCH_BLOCKING=1
python src/main.py --dataset PRAD --model TWO_STAGE_VAR_ST --training_stage 1 --debug
```

## 引用

如果使用此模型，请引用相关论文：

```bibtex
@article{two_stage_var_st,
  title={Two-Stage VAR-ST: A Novel Approach for Spatial Transcriptomics Gene Expression Prediction},
  author={},
  journal={},
  year={2024}
}
```

## 许可证

MIT License 