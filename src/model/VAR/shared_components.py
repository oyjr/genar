"""
共享组件模块 - 两阶段VAR-ST的核心组件

🔧 主要特性：
1. 改进codebook初始化 (Xavier → std=0.02)
2. 增大commitment loss权重 (0.25 → 0.5)
3. 添加EMA更新支持
4. 添加codebook利用率监控
5. 编码器添加LayerNorm稳定训练

严格遵循VAR原始设计：
- 单一共享codebook，词汇表大小4096
- 所有尺度编码器输出统一128维
- 生物学多尺度：Global(1) → Pathway(8) → Module(32) → Individual(200)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, List


class SharedVectorQuantizer(nn.Module):
    """
    共享向量量化器 - 改进版，解决Codebook Collapse问题
    
    🔧 关键改进：
    - 更大的初始化范围，避免codebook向量过于相似
    - 增大commitment loss权重，强化编码器学习
    - 支持EMA更新，提高训练稳定性
    - 添加codebook利用率监控
    """
    
    def __init__(
        self,
        vocab_size: int = 4096,
        embed_dim: int = 128,
        beta: float = 0.5,  # 🔧 增大commitment loss权重
        use_ema: bool = True,  # 🆕 启用EMA更新
        ema_decay: float = 0.99,
        epsilon: float = 1e-5
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.epsilon = epsilon
        
        # 🔧 改进的codebook初始化
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 使用更大的初始化范围，确保codebook向量多样性
        std = 0.02  # 比原来的1/vocab_size=0.0002大100倍
        nn.init.normal_(self.embedding.weight, mean=0, std=std)
        
        # 🆕 EMA统计
        if use_ema:
            self.register_buffer('cluster_size', torch.zeros(vocab_size))
            self.register_buffer('embed_avg', self.embedding.weight.data.clone())
        
        # 🆕 利用率统计
        self.register_buffer('usage_count', torch.zeros(vocab_size))
        
        # Initialization complete
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """前向传播 - 改进版"""
        input_shape = x.shape
        
        # 处理不同输入形状
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        elif x.dim() == 3:
            squeeze_output = False
        else:
            raise ValueError(f"不支持的输入维度: {x.shape}")
        
        B, N, D = x.shape
        assert D == self.embed_dim, f"输入维度 {D} 不匹配嵌入维度 {self.embed_dim}"
        
        # 计算距离并获取最近的codebook entry
        flat_x = x.view(-1, D)  # [B*N, D]
        
        # 计算欧几里得距离
        distances = torch.cdist(flat_x, self.embedding.weight)  # [B*N, vocab_size]
        tokens_flat = torch.argmin(distances, dim=1)  # [B*N]
        tokens = tokens_flat.view(B, N)  # [B, N]
        
        # 获取量化特征
        quantized = self.embedding(tokens)  # [B, N, embed_dim]
        
        # 🆕 更新使用统计
        with torch.no_grad():
            token_counts = torch.bincount(tokens_flat, minlength=self.vocab_size).float()
            self.usage_count.add_(token_counts)
        
        # 🆕 EMA更新（只在训练时）
        if self.training and self.use_ema:
            self._ema_update(flat_x, tokens_flat)
        
        # 🔧 改进的VQ损失 - 更高的commitment weight
        commitment_loss = F.mse_loss(quantized.detach(), x)
        embedding_loss = F.mse_loss(quantized, x.detach())
        vq_loss = embedding_loss + self.beta * commitment_loss
        
        # 直通估计器
        quantized = x + (quantized - x).detach()
        
        # 根据输入形状调整输出
        if squeeze_output:
            tokens = tokens.squeeze(1)  # [B, 1] → [B]
            quantized = quantized.squeeze(1)  # [B, 1, embed_dim] → [B, embed_dim]
        
        return tokens, quantized, vq_loss
    
    def _ema_update(self, flat_x: torch.Tensor, tokens_flat: torch.Tensor):
        """🆕 EMA更新codebook - 修复内存泄漏"""
        with torch.no_grad():  # 🔧 确保整个EMA更新过程不保留计算图
            # 计算每个token的使用次数
            token_counts = torch.bincount(tokens_flat, minlength=self.vocab_size).float()
            
            # 更新cluster size
            self.cluster_size.mul_(self.ema_decay).add_(token_counts, alpha=1 - self.ema_decay)
            
            # 计算每个token对应的特征平均值
            embed_sum = torch.zeros_like(self.embed_avg)
            # 🔧 关键修复：确保flat_x不保留计算图
            flat_x_detached = flat_x.detach()
            embed_sum.index_add_(0, tokens_flat, flat_x_detached)
            
            # 更新embedding average
            self.embed_avg.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)
            
            # 更新embedding权重
            cluster_size = self.cluster_size + self.epsilon
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)
    
    def get_codebook_utilization(self) -> float:
        """🆕 获取codebook利用率"""
        used_codes = (self.usage_count > 0).sum().item()
        return used_codes / self.vocab_size
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """从tokens解码为特征"""
        return self.embedding(tokens)


# 其他组件保持不变，只添加改进的编码器（可选）
class GlobalEncoder(nn.Module):
    """Global层编码器: [B, 1] → [B, 1, 128]"""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 🔧 可选改进：添加LayerNorm
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),  # 🆕 稳定训练
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        encoded = self.encoder(x)  # [B, 1] → [B, 128]
        return encoded.view(B, 1, self.embed_dim)


class PathwayEncoder(nn.Module):
    """Pathway层编码器: [B, 8] → [B, 8, 128]"""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),  # 🆕 稳定训练
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        x_expanded = x.unsqueeze(-1)  # [B, 8, 1]
        encoded = self.encoder(x_expanded)  # [B, 8, 128]
        return encoded


class ModuleEncoder(nn.Module):
    """Module层编码器: [B, 32] → [B, 32, 128]"""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(1, 96),
            nn.LayerNorm(96),  # 🆕 稳定训练
            nn.ReLU(),
            nn.Linear(96, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        x_expanded = x.unsqueeze(-1)  # [B, 32, 1]
        encoded = self.encoder(x_expanded)  # [B, 32, 128]
        return encoded


class IndividualEncoder(nn.Module):
    """Individual层编码器: [B, 200] → [B, 200, 128]"""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.LayerNorm(256),  # 🆕 稳定训练
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        x_expanded = x.unsqueeze(-1)  # [B, 200, 1]
        encoded = self.encoder(x_expanded)  # [B, 200, 128]
        return encoded


# 解码器保持不变...
class GlobalDecoder(nn.Module):
    """Global层解码器: [B, 1, 128] → [B, 1]"""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_flat = x.view(B, -1)  # [B, 128]
        decoded = self.decoder(x_flat)  # [B, 1]
        return decoded


class PathwayDecoder(nn.Module):
    """Pathway层解码器: [B, 8, 128] → [B, 8]"""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(), 
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        decoded = self.decoder(x)  # [B, 8, 1]
        return decoded.squeeze(-1)  # [B, 8]


class ModuleDecoder(nn.Module):
    """Module层解码器: [B, 32, 128] → [B, 32]"""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 96),
            nn.ReLU(),
            nn.Linear(96, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        decoded = self.decoder(x)  # [B, 32, 1]
        return decoded.squeeze(-1)  # [B, 32]


class IndividualDecoder(nn.Module):
    """Individual层解码器: [B, 200, 128] → [B, 200]"""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        decoded = self.decoder(x)  # [B, 200, 1]
        return decoded.squeeze(-1)  # [B, 200]


class ResidualReconstructor(nn.Module):
    """残差重建器 - 不变"""
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self, 
        global_recon: torch.Tensor,
        pathway_recon: torch.Tensor,
        module_recon: torch.Tensor,
        individual_recon: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        B = global_recon.shape[0]
        
        # 1. Global层广播
        global_broadcast = global_recon.expand(B, 200)  # [B, 1] → [B, 200]
        
        # 2. Pathway层广播
        pathway_broadcast = pathway_recon.repeat_interleave(25, dim=1)  # [B, 8] → [B, 200]
        
        # 3. Module层广播
        module_expanded = F.interpolate(
            module_recon.unsqueeze(1),  # [B, 32] → [B, 1, 32]
            size=200,
            mode='linear',
            align_corners=False
        ).squeeze(1)  # [B, 1, 200] → [B, 200]
        
        # 4. 残差累积重建
        cumulative_recon = global_broadcast.clone()
        cumulative_recon = cumulative_recon + pathway_broadcast
        cumulative_recon = cumulative_recon + module_expanded
        final_recon = cumulative_recon + individual_recon
        
        return {
            'global_broadcast': global_broadcast,
            'pathway_broadcast': pathway_broadcast, 
            'module_broadcast': module_expanded,
            'individual_contribution': individual_recon,
            'cumulative_without_individual': cumulative_recon,
            'final_reconstruction': final_recon
        }


class MultiScaleDecomposer(nn.Module):
    """多尺度分解器 - 不变"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, gene_expression: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, num_genes = gene_expression.shape
        assert num_genes == 200, f"期望200个基因，得到{num_genes}"
        
        # Global层：整体平均
        global_features = gene_expression.mean(dim=1, keepdim=True)  # [B, 1]
        
        # Pathway层：8个生物学通路
        pathway_features = F.adaptive_avg_pool1d(
            gene_expression.unsqueeze(1), 8
        ).squeeze(1)  # [B, 8]
        
        # Module层：32个功能模块
        module_features = F.adaptive_avg_pool1d(
            gene_expression.unsqueeze(1), 32
        ).squeeze(1)  # [B, 32]
        
        # Individual层：保持原始分辨率
        individual_features = gene_expression.clone()  # [B, 200]
        
        return {
            'global': global_features,
            'pathway': pathway_features,
            'module': module_features,
            'individual': individual_features
        } 
