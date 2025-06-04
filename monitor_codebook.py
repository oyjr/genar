#!/usr/bin/env python3
"""
Codebook利用率监控脚本

训练期间运行此脚本来监控codebook利用情况
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
from model.VAR.multi_scale_gene_vqvae import MultiScaleGeneVQVAE

def monitor_codebook_usage(checkpoint_path):
    """监控checkpoint中的codebook使用情况"""
    print(f"🔍 监控checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 加载模型
        vqvae = MultiScaleGeneVQVAE()
        if 'state_dict' in checkpoint:
            state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('model.stage1_vqvae.'):
                    new_key = key.replace('model.stage1_vqvae.', '')
                    state_dict[new_key] = value
            vqvae.load_state_dict(state_dict, strict=False)
        
        # 检查codebook利用率
        if hasattr(vqvae.shared_quantizer, 'usage_count'):
            usage_count = vqvae.shared_quantizer.usage_count
            used_codes = (usage_count > 0).sum().item()
            total_codes = len(usage_count)
            utilization = used_codes / total_codes
            
            print(f"📊 Codebook利用率: {used_codes}/{total_codes} ({utilization:.4f})")
            
            if utilization < 0.1:
                print("❌ 利用率过低！可能存在codebook collapse")
            elif utilization < 0.3:
                print("⚠️ 利用率偏低，建议继续训练")
            else:
                print("✅ 利用率正常")
        else:
            print("⚠️ 模型不支持utilization统计")
            
    except Exception as e:
        print(f"❌ 监控失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # 默认监控最新的checkpoint
        checkpoint_path = "logs/PRAD/TWO_STAGE_VAR_ST/stage1-best-epoch=*.ckpt"
        
    monitor_codebook_usage(checkpoint_path)
