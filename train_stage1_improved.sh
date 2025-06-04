#!/bin/bash
# 🔧 改进的Stage 1训练脚本

echo "🚀 开始改进的Stage 1训练..."

# 🔧 关键改进参数：
# - 增大batch_size到128 (更稳定的梯度)
# - 降低学习率到1e-4 (避免训练过快)
# - 增加epochs到200 (充分训练)
# - 添加weight_decay (正则化)

python src/main.py \
    --dataset PRAD \
    --model TWO_STAGE_VAR_ST \
    --training_stage 1 \
    --epochs 200 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --gpus 1

echo "✅ Stage 1训练完成"
echo "请检查logs中的codebook利用率"
