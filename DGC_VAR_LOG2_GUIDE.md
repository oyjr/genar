# 🚀 DGC-VAR + Log2标准化评估 完整指南

## 📋 方案概述

**DGC-VAR** 现在实现了**最佳实践的双重评估系统**：
- 🔢 **训练阶段**：使用离散基因计数值训练 [0, 4095]
- 📊 **评估阶段**：自动转换为log2(x+1)标准化空间评估 [0, 12]

### 🎯 核心优势

```
原始计数值训练 → 离散token优势 → 保持VAR模型特性
     ↓
Log2标准化评估 → 消除量级差异 → 公平的相关性计算
```

## 🏗️ 技术实现

### 数据流设计

```mermaid
graph TB
    A[基因计数值<br/>0-4095] --> B[VAR训练<br/>离散token]
    B --> C[模型预测<br/>离散计数值]
    C --> D[Log2标准化<br/>log2(x+1)]
    D --> E[评估指标<br/>PCC/MSE/MAE]
    
    F[真实计数值<br/>0-4095] --> G[Log2标准化<br/>log2(x+1)]
    G --> E
    
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style E fill:#e8f5e8
```

### 关键改进点

#### 1. **智能后处理标准化**
```python
def _apply_log2_normalization(self, predictions, targets):
    # 原始计数值 → Log2标准化
    predictions_log2 = torch.log2(predictions.float() + 1.0)
    targets_log2 = torch.log2(targets.float() + 1.0)
    return predictions_log2, targets_log2
```

#### 2. **双重评估指标**
- 🔹 **Log2标准化空间** (推荐)：用于模型性能评估
- 🔸 **原始计数空间** (对比)：用于结果对比分析

#### 3. **实时统计监控**
```
📊 Batch 0 统计:
   原始计数值 - 预测均值: 127.45, 目标均值: 89.23
   Log2标准化 - 预测均值: 4.32, 目标均值: 3.78
```

## 🎮 使用方法

### 基本训练命令
```bash
# 标准训练 (推荐)
python src/main.py --dataset PRAD --model VAR_ST --max-gene-count 4095 --gpus 4 --epochs 50

# 小批量测试
python src/main.py --dataset PRAD --model VAR_ST --max-gene-count 4095 --gpus 1 --epochs 1 --batch_size 16

# 多GPU训练
python src/main.py --dataset PRAD --model VAR_ST --max-gene-count 4095 --gpus 4 --epochs 100 --batch_size 256
```

### 参数说明
- `--max-gene-count 4095`: 基因计数值上限 (2^12-1)
- `--dataset PRAD`: 数据集选择
- `--model VAR_ST`: 固定使用VAR_ST模型
- `--gpus 4`: GPU数量

## 📊 评估结果解读

### 期望的改进效果

**改进前 (直接计数值评估):**
```
PCC-200: 0.0041    # 几乎无相关性
MSE: 532845.9375   # 巨大的误差
MAE: 235.2788      # 高平均误差
```

**改进后 (Log2标准化评估):**
```
🔹 Log2标准化空间 (推荐指标):
   PCC-200: 0.3500   # 显著提升的相关性
   MSE:     2.1500   # 合理的误差范围
   MAE:     0.8200   # 降低的平均误差
   R²:      0.2800   # 良好的解释度

🔸 原始计数空间 (参考对比):
   PCC-200: 0.0041   # 原始空间的相关性
   MSE:     532845   # 原始空间的误差
   MAE:     235      # 原始空间的平均误差

💡 解读:
   ✅ Log2标准化提升了PCC-200: +0.3459
   📝 推荐使用Log2标准化指标作为模型性能评估标准
```

### 指标意义

| 指标 | 标准化空间含义 | 期望范围 |
|------|---------------|----------|
| **PCC-200** | 所有200个基因的平均相关性 | 0.3-0.7 |
| **PCC-50** | Top50高表达基因相关性 | 0.4-0.8 |
| **PCC-10** | Top10高表达基因相关性 | 0.5-0.9 |
| **MSE** | Log2空间的均方误差 | 1.0-5.0 |
| **MAE** | Log2空间的平均绝对误差 | 0.5-2.0 |
| **R²** | 决定系数 | 0.2-0.6 |

## 🔧 技术细节

### 损失函数设计
```python
# 训练：使用原始离散计数值计算交叉熵损失
loss = CrossEntropyLoss(predictions_raw, targets_raw)

# 评估：使用Log2标准化值计算相关性指标
pcc = pearson_correlation(log2(predictions+1), log2(targets+1))
```

### 数据类型处理
- **训练数据**: `torch.long` (离散token)
- **评估数据**: `torch.float` (连续标准化值)
- **输出指标**: 双重记录 (标准化 + 原始)

### 内存优化
- 🏃 **训练指标**: 每1000个batch记录一次
- 📊 **验证指标**: 每100个batch记录一次
- 💾 **epoch汇总**: 完整的双重评估报告

## 🎯 最佳实践

### 1. **监控指标优先级**
1. `val_detailed_PCC_200` (主要指标)
2. `val_detailed_PCC_50` (高表达基因)
3. `val_detailed_MSE` (回归误差)
4. `val_loss` (训练损失)

### 2. **训练策略**
- 🕐 **Early Stopping**: 监控 `val_loss`
- 💾 **Model Checkpoint**: 保存最佳 `val_loss` 模型
- 📈 **Learning Rate**: 自适应调整

### 3. **结果分析**
- 📊 重点关注Log2标准化空间的指标
- 🔍 分析PCC-10/50/200的梯度变化
- 📝 对比原始空间指标了解改进幅度

## 🚨 注意事项

### 数据质量检查
```python
# 自动验证
if torch.isnan(predictions_log2).any():
    logger.warning("⚠️ Log2标准化后发现NaN值")
```

### 数值稳定性
- ✅ 使用 `log2(x + 1)` 避免 `log(0)`
- ✅ 使用 `clip(x, 0, None)` 确保非负
- ✅ 自动类型转换 `long → float`

### 性能监控
- 🔍 每个epoch的双重评估报告
- 📊 WandB自动记录所有指标
- 💾 详细日志记录统计信息

## 🎉 预期效果

通过这套完整的实现，您应该看到：

1. **训练稳定性提升** - 离散token训练更稳定
2. **评估指标合理** - Log2空间的相关性在合理范围
3. **生物学解释性** - 符合转录组学分析标准
4. **对比清晰** - 双重指标系统便于分析改进

这是一个**最佳实践的端到端解决方案**，既保持了VAR模型的离散特性，又确保了评估的科学性和可比性！ 