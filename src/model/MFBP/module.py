import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """保持原始的CrossAttention实现"""
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
            context: 上下文特征 [B, 1, D] 或 [B, N, D]
        Returns:
            out: 增强后的特征 [B, N, D]
        """
        q = self.query(x)  # [B, N, D]
        k = self.key(context)  # [B, 1, D] 或 [B, N, D]
        v = self.value(context)  # [B, 1, D] 或 [B, N, D]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, N, 1] 或 [B, N, N]
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v)  # [B, N, D]
        return out

class AdaptiveFusion(nn.Module):
    """改进的AdaptiveFusion，借鉴SURVPATH思想但保持原有结构"""
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 保持原始的三个独立CrossAttention
        self.local_global_attn = CrossAttention(feature_dim)  # P-to-P
        self.path_density_attn = CrossAttention(feature_dim)  # P-to-H
        self.density_path_attn = CrossAttention(feature_dim)  # H-to-P
        
        # 特征融合 - 保持原始设计
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )
        
        # 输出投影 - 保持原始设计
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, original_features, global_features, density_features):
        """保持原有结构，但角色对应SURVPATH
        Args:
            original_features: 原始特征 [B, N, D] - 对应"通路特征"
            global_features: 全局特征 [B, 1, D] - 扩展为[B, N, D]
            density_features: 密度特征 [B, N, D] - 对应"组织学特征"
        Returns:
            output: 融合后的特征 [B, N, D//2]
        """
        B, N, D = original_features.shape
        
        # 确保global_features维度匹配
        if global_features.dim() == 2:
            global_features = global_features.unsqueeze(1)
        if global_features.size(1) == 1:
            global_features = global_features.expand(-1, N, -1)
            
        # 关键点：保持原有的关注模式和独立注意力计算
        # 1. P-to-P: 原始特征关注全局特征 (而不是自注意力)
        local_global_attn = self.local_global_attn(original_features, global_features)
        
        # 2. P-to-H: 原始特征关注密度特征
        path_density_attn = self.path_density_attn(original_features, density_features)
        
        # 3. H-to-P: 密度特征关注原始特征
        density_path_attn = self.density_path_attn(density_features, original_features)
        
        # 按原始方式拼接特征 - 这是关键不同点
        fused_features = torch.cat([
            local_global_attn,
            path_density_attn,
            density_path_attn
        ], dim=-1)
        
        # 使用线性层融合 - 保持原始设计
        fused_features = self.fusion(fused_features)
        
        # 添加残差连接 - 保持原始设计
        fused_features = fused_features + original_features
        
        # 输出投影 - 保持原始设计
        output = self.output_proj(fused_features)
        
        return output

class DensityPredictor(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=256, output_dim=200):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()  # 只保留Softplus确保非负性
        )
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """优化初始权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, features):
        # features: [B, N, 1024] -> [B, N, 200]
        return self.predictor(features)  # 直接返回预测结果，不进行归一化

class DensityEncoder(nn.Module):
    def __init__(self, input_dim=200, hidden_dim=256, output_dim=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """优化初始权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # x: [B, N, 200] -> [B, N, 1024]
        return self.encoder(x)

class GlobalFeatureAggregator(nn.Module):
    def __init__(self, feature_dim=1024):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """优化初始权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
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

class MFBPModule(nn.Module):
    def __init__(self, num_genes=50, feature_dim=1024, num_density_genes=200):
        super().__init__()
        
        # 获取实际的密度基因数量
        try:
            # 尝试从batch中的density_genes获取维度
            self.num_density_genes = num_density_genes
        except:
            # 如果获取失败，使用默认值200
            self.num_density_genes = 200
            
        # 全局特征聚合
        self.global_aggregator = GlobalFeatureAggregator(feature_dim)
        
        # 密度预测和编码
        self.density_predictor = DensityPredictor(
            feature_dim=feature_dim, 
            output_dim=self.num_density_genes
        )
        self.density_encoder = DensityEncoder(
            input_dim=self.num_density_genes,
            output_dim=feature_dim
        )
        
        # 特征增强
        self.feature_enhancement = nn.ModuleDict({
            'original': nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU()
            ),
            'global': nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU()
            ),
            'density': nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.ReLU()
            )
        })
        
        # 使用基于SURVPATH的融合模块，但保持接口一致
        self.fusion = AdaptiveFusion(feature_dim)
        
        # 预测头，适度优化
        self.predictor = nn.Linear(feature_dim // 2, num_genes)
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """优化初始权重"""
        for name, module in self.feature_enhancement.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
        # 预测头使用小的初始权重
        nn.init.xavier_uniform_(self.predictor.weight, gain=0.01)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)
        
    def forward(self, img):
        """
        Args:
            img: spot特征 [B, N, 1024]，虽然参数名是img，但实际是特征
        Returns:
            dict: 包含预测结果的字典
                - logits: [B, N, num_genes] 每个spot的基因表达预测
                - density_pred: [B, N, num_density_genes] 每个spot的密度预测
        """
        # 1. 特征增强
        original_features = self.feature_enhancement['original'](img)
        
        # 2. 全局特征
        global_features = self.global_aggregator(img)  # [B, 1, 1024]
        global_features = self.feature_enhancement['global'](global_features)
        
        # 3. 密度预测和编码
        density_pred = self.density_predictor(img)  # [B, N, 200]
        density_features = self.density_encoder(density_pred)  # [B, N, 1024]
        density_features = self.feature_enhancement['density'](density_features)
        
        # 4. 自适应特征融合 - 使用优化版本
        fused_features = self.fusion(
            original_features,
            global_features,
            density_features
        )
        
        # 5. 预测每个spot的基因表达
        predictions = self.predictor(fused_features)  # [B, N, num_genes]
        
        return {
            'logits': predictions,
            'density_pred': density_pred
        }