#!/bin/bash

# MFBP Multi-GPU Training Script
# Simplified distributed training script for MFBP model using 4x RTX 4090 GPUs
# Updated to use the new simplified command interface

# Configure CUDA environment variables for optimal multi-GPU performance
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Specify which GPUs to use (0-3 for 4 GPUs)
export NCCL_DEBUG=INFO  # Enable NCCL debugging output for troubleshooting communication issues
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available (use Ethernet instead)
export NCCL_P2P_DISABLE=1  # Disable peer-to-peer GPU communication if causing issues

# Training configuration parameters
DATASET="PRAD"  # Dataset to train on (PRAD, her2st)
GPUS=4  # Number of GPUs to use for distributed training
EPOCHS=200  # Total number of training epochs (optional override)
BATCH_SIZE=256  # Batch size per GPU (optional override)

# Display training configuration summary
echo "🚀 开始4卡分布式训练..."
echo "📊 数据集: $DATASET"
echo "💻 GPU数量: $GPUS"
echo "📦 批次大小: $BATCH_SIZE"
echo "⏰ 训练轮数: $EPOCHS"

# Execute distributed training with new simplified command
# The new interface automatically handles:
# - Dataset paths and slide splits
# - Encoder selection (uni for PRAD, conch for her2st)
# - Multi-GPU strategy (DDP)
# - Data augmentation settings
python src/main.py \
    --dataset $DATASET \
    --gpus $GPUS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --strategy ddp \
    --sync-batchnorm

echo "✅ 训练完成！"

# Optional: Run inference on test set using single GPU for consistency
echo "🧪 开始测试..."
python src/main.py \
    --dataset $DATASET \
    --gpus 1 \
    --mode test

echo "🎉 所有任务完成！" 