import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any

from .module_no_density import MFBPModule_no_density


class MFBP(nn.Module):
    def __init__(self, config=None, num_genes=200, feature_dim=1024):
        super().__init__()
        
        # 从配置中获取参数，如果没有则使用默认值
        if config is not None:
            self.num_genes = getattr(config.MODEL, 'num_genes', num_genes)
            self.feature_dim = getattr(config.MODEL, 'feature_dim', feature_dim)
        else:
            self.num_genes = num_genes
            self.feature_dim = feature_dim
        
        print(f"初始化MFBP模型:")
        print(f"  - 基因数量: {self.num_genes}")
        print(f"  - 特征维度: {self.feature_dim}")
        
        # 使用简化的模块
        self.model = MFBPModule_no_density(
            num_genes=self.num_genes,
            feature_dim=self.feature_dim
        )
        
    def forward(self, img, **kwargs):
        """前向传播函数
        
        Args:
            img: 预提取的特征 [B, N, 1024] 或 [B, 1024] 
            **kwargs: 额外的参数（兼容性保留）
                
        Returns:
            dict: 包含预测结果的字典
                - logits: [B, N, num_genes] 基因表达预测
        """
        # 处理输入维度
        if img.dim() == 2:
            # 如果是训练模式的单个spot: [B, 1024] -> [B, 1, 1024]
            img = img.unsqueeze(1)
        elif img.dim() == 3:
            # 验证/测试模式的多个spots: [B, N, 1024]
            pass
        else:
            raise ValueError(f"输入特征维度不正确: {img.shape}")
        
        # 通过模型获取预测结果
        outputs = self.model(img)
        
        # 返回标准化的输出格式
        return {
            'logits': outputs['predictions']  # [B, N, num_genes]
        } 