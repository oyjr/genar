"""
共享组件模块 - 两阶段VAR-ST的核心组件

包含：
1. SharedVectorQuantizer: 符合VAR原始设计的单一共享codebook
2. MultiScaleEncoder: 生物学多尺度编码器
3. MultiScaleDecoder: 对应的解码器  
4. ResidualReconstructor: 残差重建策略

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
    共享向量量化器 - 严格遵循VAR原始设计
    
    特性：
    - 单一共享codebook，所有尺度使用同一词汇表
    - 词汇表大小4096（与VAR一致）
    - 嵌入维度128
    - 支持不同形状的输入张量
    """
    
    def __init__(
        self,
        vocab_size: int = 4096,
        embed_dim: int = 128,
        beta: float = 0.25
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.beta = beta
        
        # 共享codebook - VAR核心设计
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1/vocab_size, 1/vocab_size)
        
        print(f"🔧 SharedVectorQuantizer初始化:")
        print(f"   词汇表大小: {vocab_size}")
        print(f"   嵌入维度: {embed_dim}")
        print(f"   β参数: {beta}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, N, embed_dim] 或 [B, embed_dim]
            
        Returns:
            tokens: 离散token索引 [B, N] 或 [B]
            quantized: 量化后的特征 [B, N, embed_dim] 或 [B, embed_dim]  
            vq_loss: VQ损失标量
        """
        input_shape = x.shape
        
        # 处理不同输入形状
        if x.dim() == 2:
            # [B, embed_dim] → [B, 1, embed_dim]
            x = x.unsqueeze(1)
            squeeze_output = True
        elif x.dim() == 3:
            # [B, N, embed_dim] 保持不变
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
        
        # 计算VQ损失
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
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        从tokens解码为特征
        
        Args:
            tokens: token索引 [B, N] 或 [B]
            
        Returns:
            特征 [B, N, embed_dim] 或 [B, embed_dim]
        """
        return self.embedding(tokens)


class GlobalEncoder(nn.Module):
    """Global层编码器: [B, 1] → [B, 1, 128]"""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1] - Global特征
        Returns:
            [B, 1, 128] - 编码后特征
        """
        B = x.shape[0]
        encoded = self.encoder(x)  # [B, 1] → [B, 128]
        return encoded.view(B, 1, self.embed_dim)  # [B, 1, 128]


class PathwayEncoder(nn.Module):
    """Pathway层编码器: [B, 8] → [B, 8, 128]"""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),  # 每个pathway独立编码
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 8] - Pathway特征
        Returns:
            [B, 8, 128] - 编码后特征
        """
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
            nn.Linear(1, 96),  # 每个module独立编码
            nn.ReLU(),
            nn.Linear(96, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 32] - Module特征
        Returns:
            [B, 32, 128] - 编码后特征
        """
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
            nn.Linear(1, 256),  # 每个基因独立编码
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 200] - Individual特征
        Returns:
            [B, 200, 128] - 编码后特征
        """
        B, N = x.shape
        x_expanded = x.unsqueeze(-1)  # [B, 200, 1]
        encoded = self.encoder(x_expanded)  # [B, 200, 128]
        return encoded


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
        """
        Args:
            x: [B, 1, 128] - 量化特征
        Returns:
            [B, 1] - 重建特征
        """
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
        """
        Args:
            x: [B, 8, 128] - 量化特征
        Returns:
            [B, 8] - 重建特征
        """
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
        """
        Args:
            x: [B, 32, 128] - 量化特征
        Returns:
            [B, 32] - 重建特征
        """
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
        """
        Args:
            x: [B, 200, 128] - 量化特征
        Returns:
            [B, 200] - 重建特征
        """
        B, N, D = x.shape
        decoded = self.decoder(x)  # [B, 200, 1]
        return decoded.squeeze(-1)  # [B, 200]


class ResidualReconstructor(nn.Module):
    """
    残差重建器 - 实现生物学多尺度的残差重建策略
    
    重建顺序：
    1. Global重建 → 广播到200维
    2. Pathway重建 → 广播到200维，加到Global基础上
    3. Module重建 → 广播到200维，加到前两层基础上  
    4. Individual重建 → 直接200维，最终细节
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self, 
        global_recon: torch.Tensor,      # [B, 1]
        pathway_recon: torch.Tensor,     # [B, 8] 
        module_recon: torch.Tensor,      # [B, 32]
        individual_recon: torch.Tensor   # [B, 200]
    ) -> Dict[str, torch.Tensor]:
        """
        残差重建
        
        Args:
            global_recon: Global层重建 [B, 1]
            pathway_recon: Pathway层重建 [B, 8]
            module_recon: Module层重建 [B, 32]
            individual_recon: Individual层重建 [B, 200]
            
        Returns:
            包含各层重建和最终重建的字典
        """
        B = global_recon.shape[0]
        
        # 1. Global层广播
        global_broadcast = global_recon.expand(B, 200)  # [B, 1] → [B, 200]
        
        # 2. Pathway层广播（8个通路，每个对应25个基因）
        pathway_broadcast = pathway_recon.repeat_interleave(25, dim=1)  # [B, 8] → [B, 200]
        
        # 3. Module层广播（32个模块，每个对应6.25个基因，需要处理非整数）
        # 使用线性插值的方式处理非整数倍的映射
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
    """
    多尺度分解器 - 将200维基因表达分解为生物学多尺度
    
    分解策略：
    - Global(1): 所有基因的平均值，代表整体转录活跃度
    - Pathway(8): 将200个基因分为8组，每组25个基因的平均值
    - Module(32): 将200个基因分为32组，每组6.25个基因的平均值  
    - Individual(200): 保持原始单基因分辨率
    """
    
    def __init__(self):
        super().__init__()
        
    def forward(self, gene_expression: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        多尺度分解
        
        Args:
            gene_expression: [B, 200] - 原始基因表达
            
        Returns:
            包含各尺度特征的字典
        """
        B, num_genes = gene_expression.shape
        assert num_genes == 200, f"期望200个基因，得到{num_genes}"
        
        # Global层：整体平均
        global_features = gene_expression.mean(dim=1, keepdim=True)  # [B, 1]
        
        # Pathway层：8个生物学通路
        pathway_features = F.adaptive_avg_pool1d(
            gene_expression.unsqueeze(1), 8
        ).squeeze(1)  # [B, 200] → [B, 1, 200] → [B, 1, 8] → [B, 8]
        
        # Module层：32个功能模块
        module_features = F.adaptive_avg_pool1d(
            gene_expression.unsqueeze(1), 32
        ).squeeze(1)  # [B, 200] → [B, 1, 200] → [B, 1, 32] → [B, 32]
        
        # Individual层：保持原始分辨率
        individual_features = gene_expression.clone()  # [B, 200]
        
        return {
            'global': global_features,      # [B, 1]
            'pathway': pathway_features,    # [B, 8]
            'module': module_features,      # [B, 32]
            'individual': individual_features  # [B, 200]
        } 