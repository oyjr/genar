import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x, context):
        """交叉注意力机制
        Args:
            x: 特征向量 [B, N, D]
            context: 上下文特征 [B, 1, D]
        Returns:
            out: 增强后的特征 [B, N, D]
        """
        q = self.query(x)  # [B, N, D]
        k = self.key(context)  # [B, 1, D]
        v = self.value(context)  # [B, 1, D]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, N, 1]
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v)  # [B, N, D]
        return out

class GlobalFeatureAggregator(nn.Module):
    def __init__(self, feature_dim=1024):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, features):
        """聚合全局特征
        Args:
            features: [B, N, 1024] spot特征
        Returns:
            global_features: [B, 1, 1024] 全局特征
        """
        # 计算注意力权重
        attn_weights = self.attention(features)  # [B, N, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # 加权平均得到全局特征
        global_features = (features * attn_weights).sum(dim=1, keepdim=True)  # [B, 1, 1024]
        return global_features

class MFBPModule_no_density(nn.Module):
    def __init__(self, num_genes=50, feature_dim=1024):
        super().__init__()
        
        # 全局特征聚合
        self.global_aggregator = GlobalFeatureAggregator(feature_dim)
        
        # 只保留全局注意力
        self.global_attention = CrossAttention(feature_dim)
        
        # 特征融合（修改输入维度，因为少了密度特征）
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),  # 只有两个特征拼接
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 预测头
        self.predictor = nn.Linear(feature_dim // 2, num_genes)
        
    def forward(self, img):
        """
        Args:
            img: spot特征 [B, N, 1024]，虽然参数名是img，但实际是特征
        Returns:
            dict: 包含预测结果的字典
                - predictions: [B, N, num_genes] 每个spot的基因表达预测
                - density_pred: None (不进行密度预测)
        """
        # 1. 提取全局特征
        global_features = self.global_aggregator(img)  # [B, 1, 1024]
        
        # 2. 特征增强（只用全局特征）
        global_enhanced = self.global_attention(img, global_features)  # [B, N, 1024]
        
        # 3. 特征融合
        fused_features = torch.cat([
            img,  # 原始spot特征
            global_enhanced,  # 全局增强特征
        ], dim=-1)  # [B, N, 1024*2]
        
        fused_features = self.fusion(fused_features)  # [B, N, 512]
        
        # 4. 预测
        predictions = self.predictor(fused_features)  # [B, N, num_genes]
        
        return {
            'predictions': predictions,
            'density_pred': None  # 返回None表示没有密度预测
        } 