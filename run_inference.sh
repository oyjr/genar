#!/bin/bash

# VAR_ST模型推理脚本
# 用于测试PRAD数据集的MEND144样本

echo "🚀 开始VAR_ST模型推理..."

# 设置参数
CKPT_PATH="/home/ouyangjiarui/project/ST/genar/logs/PRAD/VAR_ST/best-epoch=epoch=01-val_loss_final=val_loss_final=101.7450.ckpt"
DATASET="PRAD"
SLIDE_ID="MEND144"
OUTPUT_DIR="./inference_results"
GPU_ID=0

# 检查checkpoint文件是否存在
if [ ! -f "$CKPT_PATH" ]; then
    echo "❌ 错误: Checkpoint文件不存在: $CKPT_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行推理
echo "📊 运行推理..."
echo "   - Checkpoint: $CKPT_PATH"
echo "   - 数据集: $DATASET"
echo "   - Slide ID: $SLIDE_ID"
echo "   - 输出目录: $OUTPUT_DIR"
echo "   - GPU ID: $GPU_ID"
echo ""

python src/inference.py \
    --ckpt_path "$CKPT_PATH" \
    --dataset "$DATASET" \
    --slide_id "$SLIDE_ID" \
    --output_dir "$OUTPUT_DIR" \
    --gpu_id $GPU_ID \
    --batch_size 64 \
    --save_predictions \
    --seed 2021

# 检查运行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 推理完成！"
    echo "📁 结果保存在: $OUTPUT_DIR"
    echo ""
    echo "📋 生成的文件:"
    ls -la $OUTPUT_DIR/${SLIDE_ID}_*
else
    echo ""
    echo "❌ 推理失败！"
    exit 1
fi 