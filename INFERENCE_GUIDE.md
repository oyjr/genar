# VAR_ST模型推理指南

## 概述

本指南介绍如何使用训练好的VAR_ST模型checkpoint进行推理，并计算PCC-10, PCC-50, PCC-200, RVD, MSE, MAE等评估指标。

## 快速开始

### 方法一：使用shell脚本 (推荐)

最简单的使用方式：

```bash
# 使用你的checkpoint路径
./run_inference.sh --checkpoint /home/ouyangjiarui/project/ST/genar/logs/PRAD/VAR_ST/best-epoch=epoch=59-train_loss=train_loss=0.0054.ckpt --dataset PRAD
```

### 方法二：直接使用Python脚本

```bash
python inference.py \
    --checkpoint /home/ouyangjiarui/project/ST/genar/logs/PRAD/VAR_ST/best-epoch=epoch=59-train_loss=train_loss=0.0054.ckpt \
    --dataset PRAD \
    --encoder uni \
    --device cuda \
    --output results_prad.txt
```

## 参数说明

### 必需参数

- `--checkpoint`: 训练好的模型checkpoint路径
- `--dataset`: 数据集名称 (`PRAD` 或 `her2st`)

### 可选参数

- `--encoder`: 编码器类型 (`uni` 或 `conch`)，默认使用数据集推荐编码器
- `--device`: 推理设备，默认 `cuda`
- `--output`: 结果输出文件名，默认 `inference_results.txt`

## 数据集配置

### PRAD数据集
- 推荐编码器: `uni` (1024维特征)
- 测试集: `MEND140` slide
- 验证集: `MEND139` slide

### HER2ST数据集
- 推荐编码器: `conch` (512维特征)
- 测试集: `C1,D1` slides
- 验证集: `A1,B1` slides

## 评估指标说明

### PCC指标
- **PCC-10**: 相关性最高的前10个基因的平均Pearson相关系数
- **PCC-50**: 相关性最高的前50个基因的平均Pearson相关系数  
- **PCC-200**: 相关性最高的前200个基因的平均Pearson相关系数

### 基础指标
- **MSE**: 均方误差 (Mean Squared Error)
- **MAE**: 平均绝对误差 (Mean Absolute Error)
- **RVD**: 相对方差差异 (Relative Variance Difference)

## 输出结果

推理完成后，会在控制台显示结果并保存到指定文件中：

```
============================================================
🎯 VAR_ST模型推理结果
============================================================
📁 Checkpoint: /path/to/your/checkpoint.ckpt
📊 数据集: PRAD
🔧 编码器: uni
📏 测试样本数: XXXX
🧬 基因数量: 200
------------------------------------------------------------
📈 评估指标:
   PCC-10:  X.XXXX
   PCC-50:  X.XXXX
   PCC-200: X.XXXX
   MSE:     X.XXXXXX
   MAE:     X.XXXXXX
   RVD:     X.XXXXXX
============================================================
```

输出文件还包含详细的基因级别相关性统计和分布信息。

## 常见问题

### 1. CUDA内存不足

如果遇到CUDA内存不足，可以：

```bash
# 减少批次大小或使用CPU
python inference.py --checkpoint your_checkpoint.ckpt --dataset PRAD --device cpu
```

### 2. 模型加载失败

确保：
- Checkpoint文件路径正确
- 模型配置与训练时一致
- 数据集参数正确

### 3. 数据路径问题

确保数据集路径正确：
- PRAD: `/data/ouyangjiarui/stem/hest1k_datasets/PRAD/`
- HER2ST: `/data/ouyangjiarui/stem/hest1k_datasets/her2st/`

## 高级使用

### 批量评估多个checkpoints

```bash
# 创建批量评估脚本
for ckpt in logs/PRAD/VAR_ST/*.ckpt; do
    echo "评估: $ckpt"
    ./run_inference.sh --checkpoint "$ckpt" --dataset PRAD --output "results_$(basename $ckpt .ckpt).txt"
done
```

### 自定义输出格式

如需要其他输出格式或指标，可以修改 `inference.py` 中的 `calculate_evaluation_metrics` 函数。

## 性能基准

典型的推理性能（PRAD数据集）：
- 测试样本数: ~1000-2000个spots
- 推理时间: 1-5分钟 (取决于GPU)
- 内存使用: ~2-4GB GPU内存

## 注意事项

1. 确保使用的编码器与训练时一致
2. 推理时会自动使用测试集（不是验证集）
3. 模型会自动设置为eval模式
4. 结果文件会覆盖同名文件
5. 推理过程中模型参数不会改变 