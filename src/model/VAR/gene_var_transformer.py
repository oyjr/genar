"""
Gene VAR Transformer Components

This module implements the core components for the multi-scale gene VAR model,
including AdaLN Self-Attention blocks, condition processors, and utility functions.

Based on the original VAR architecture with adaptations for gene expression prediction.

Key Components:
1. GeneAdaLNSelfAttn: Self-attention block with adaptive layer normalization
2. GeneAdaLNBeforeHead: AdaLN layer before output head
3. ConditionProcessor: Enhanced condition processing with positional encoding
4. DropPath: Stochastic depth for regularization

Author: Assistant
Date: 2024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample
    
    Implementation from timm library, used in original VAR
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output
    
    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob}'


class SelfAttention(nn.Module):
    """
    Self-Attention module with optional L2 normalization
    
    Based on original VAR's SelfAttention implementation
    """
    def __init__(
        self,
        block_idx: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_l2_norm: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.block_idx = block_idx
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_l2_norm = attn_l2_norm
        
        # L2 normalization setup (like original VAR)
        if self.attn_l2_norm:
            self.scale = 1.0
            self.scale_mul_1H11 = nn.Parameter(
                torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), 
                requires_grad=True
            )
            self.max_scale_mul = torch.log(torch.tensor(100.0)).item()
        else:
            self.scale = 0.25 / math.sqrt(self.head_dim)
        
        # QKV projection
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()
        self.attn_drop = attn_drop
        
        # KV caching for inference
        self.caching = False
        self.cached_k = None
        self.cached_v = None
    
    def kv_caching(self, enable: bool):
        """Enable/disable KV caching"""
        self.caching = enable
        if not enable:
            self.cached_k = None
            self.cached_v = None
    
    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        
        # QKV projection with bias
        qkv = F.linear(
            input=x, 
            weight=self.mat_qkv.weight, 
            bias=torch.cat([self.q_bias, self.zero_k_bias, self.v_bias])
        ).view(B, L, 3, self.num_heads, self.head_dim)
        
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B, H, L, C//H]
        
        # L2 normalization
        if self.attn_l2_norm:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
        
        # KV caching
        if self.caching:
            if self.cached_k is None:
                self.cached_k, self.cached_v = k, v
            else:
                k = self.cached_k = torch.cat([self.cached_k, k], dim=2)
                v = self.cached_v = torch.cat([self.cached_v, v], dim=2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        if self.training and self.attn_drop > 0:
            attn_probs = F.dropout(attn_probs, p=self.attn_drop)
        
        out = torch.matmul(attn_probs, v)  # [B, H, L, C//H]
        out = out.transpose(1, 2).reshape(B, L, C)
        
        return self.proj_drop(self.proj(out))
    
    def extra_repr(self) -> str:
        return f'attn_l2_norm={self.attn_l2_norm}, caching={self.caching}'


class FFN(nn.Module):
    """
    Feed-Forward Network with GELU activation
    
    Based on original VAR's FFN implementation
    """
    def __init__(
        self, 
        in_features: int, 
        hidden_features: Optional[int] = None, 
        out_features: Optional[int] = None, 
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GeneAdaLNSelfAttn(nn.Module):
    """
    Gene-specific AdaLN Self-Attention Block
    
    Based on original VAR's AdaLNSelfAttn with adaptations for gene expression prediction.
    Uses Adaptive Layer Normalization to condition on histology and spatial features.
    """
    def __init__(
        self,
        block_idx: int,
        embed_dim: int = 768,
        condition_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_eps: float = 1e-6,
        shared_aln: bool = False,
        attn_l2_norm: bool = True,
    ):
        super().__init__()
        
        self.block_idx = block_idx
        self.embed_dim = embed_dim
        self.condition_dim = condition_dim
        self.shared_aln = shared_aln
        
        # Self-Attention
        self.attn = SelfAttention(
            block_idx=block_idx,
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            attn_l2_norm=attn_l2_norm,
        )
        
        # Feed-Forward Network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.ffn = FFN(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            drop=drop_rate
        )
        
        # LayerNorm without learnable parameters
        self.ln_wo_grad = nn.LayerNorm(embed_dim, eps=norm_eps, elementwise_affine=False)
        
        # Adaptive LayerNorm parameters
        if shared_aln:
            # Shared AdaLN parameters (saves parameters)
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            # Independent AdaLN parameters
            self.ada_lin = nn.Sequential(
                nn.SiLU(inplace=False),
                nn.Linear(condition_dim, 6 * embed_dim)
            )
        
        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
    
    def enable_kv_cache(self, enable: bool = True):
        """Enable/disable KV caching for inference"""
        self.attn.kv_caching(enable)
    
    def forward(
        self, 
        x: torch.Tensor,                    # [B, L, C]
        condition_embed: torch.Tensor,      # [B, C] or [B, 1, 6, C]
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with adaptive layer normalization
        
        Args:
            x: Input token embeddings [B, L, C]
            condition_embed: Condition embeddings [B, C]
            attn_mask: Attention mask [L, L] or [B, H, L, L]
            
        Returns:
            Output embeddings [B, L, C]
        """
        B, L, C = x.shape
        
        # Get AdaLN parameters
        if self.shared_aln:
            if condition_embed.dim() == 2:
                condition_embed = condition_embed.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, C]
            gamma1, gamma2, scale1, scale2, shift1, shift2 = (
                self.ada_gss + condition_embed
            ).unbind(2)  # 6 tensors of [B, 1, C]
        else:
            ada_params = self.ada_lin(condition_embed)  # [B, 6*C]
            gamma1, gamma2, scale1, scale2, shift1, shift2 = ada_params.view(
                B, 1, 6, C
            ).unbind(2)  # 6 tensors of [B, 1, C]
        
        # First AdaLN + Self-Attention
        x_norm1 = self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1)
        attn_output = self.attn(x_norm1, attn_mask)
        x = x + self.drop_path(attn_output.mul_(gamma1))
        
        # Second AdaLN + FFN
        x_norm2 = self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2)
        ffn_output = self.ffn(x_norm2)
        x = x + self.drop_path(ffn_output.mul(gamma2))
        
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, block_idx={self.block_idx}'


class GeneAdaLNBeforeHead(nn.Module):
    """
    Adaptive LayerNorm before output head
    
    Based on original VAR's AdaLNBeforeHead
    """
    def __init__(self, embed_dim: int, condition_dim: int, norm_eps: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.condition_dim = condition_dim
        
        self.ln_wo_grad = nn.LayerNorm(embed_dim, eps=norm_eps, elementwise_affine=False)
        self.ada_lin = nn.Sequential(
            nn.SiLU(inplace=False),
            nn.Linear(condition_dim, 2 * embed_dim)
        )
    
    def forward(self, x: torch.Tensor, condition_embed: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive layer normalization before output head
        
        Args:
            x: Input embeddings [B, L, C]
            condition_embed: Condition embeddings [B, C]
            
        Returns:
            Normalized embeddings [B, L, C]
        """
        scale, shift = self.ada_lin(condition_embed).view(-1, 1, 2, self.embed_dim).unbind(2)
        return self.ln_wo_grad(x).mul(scale.add(1)).add_(shift)


class ConditionProcessor(nn.Module):
    """
    Enhanced Condition Processor for histology features and spatial coordinates
    
    Improvements over original:
    - Better positional encoding for spatial coordinates
    - More robust feature processing
    - Proper normalization and dropout
    """
    def __init__(
        self,
        histology_dim: int = 1024,
        spatial_dim: int = 2,
        condition_embed_dim: int = 768,
        histology_hidden_dim: int = 512,
        spatial_hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.histology_dim = histology_dim
        self.spatial_dim = spatial_dim
        self.condition_embed_dim = condition_embed_dim
        
        # Histology feature processor
        self.histology_processor = nn.Sequential(
            nn.LayerNorm(histology_dim),
            nn.Linear(histology_dim, histology_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(histology_hidden_dim, histology_hidden_dim),
            nn.LayerNorm(histology_hidden_dim)
        )
        
        # Spatial coordinate processor
        self.spatial_processor = nn.Sequential(
            nn.Linear(spatial_dim, spatial_hidden_dim // 2),
            nn.GELU(),
            nn.Linear(spatial_hidden_dim // 2, spatial_hidden_dim),
            nn.LayerNorm(spatial_hidden_dim)
        )
        
        # Sinusoidal positional encoding for spatial coordinates
        self.pos_encoding_dim = spatial_hidden_dim // 2
        div_term = torch.exp(torch.arange(0, self.pos_encoding_dim, 2).float() * 
                           (-math.log(10000.0) / self.pos_encoding_dim))
        self.register_buffer('div_term', div_term)
        
        # Final projection to condition embedding dimension
        total_dim = histology_hidden_dim + spatial_hidden_dim
        self.final_projection = nn.Sequential(
            nn.Linear(total_dim, condition_embed_dim),
            nn.LayerNorm(condition_embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        histology_features: torch.Tensor,  # [B, histology_dim]
        spatial_coords: torch.Tensor       # [B, spatial_dim]
    ) -> torch.Tensor:                     # [B, condition_embed_dim]
        """
        Process histology features and spatial coordinates into condition embeddings
        
        Args:
            histology_features: Histology features [B, 1024]
            spatial_coords: Spatial coordinates [B, 2]
            
        Returns:
            Condition embeddings [B, condition_embed_dim]
        """
        
        # Process histology features
        histology_embed = self.histology_processor(histology_features)  # [B, histology_hidden_dim]
        
        # Process spatial coordinates
        spatial_embed = self.spatial_processor(spatial_coords)  # [B, spatial_hidden_dim]
        
        # Add sinusoidal positional encoding to spatial coordinates
        B = spatial_coords.shape[0]
        x_coords = spatial_coords[:, 0:1]  # [B, 1]
        y_coords = spatial_coords[:, 1:2]  # [B, 1]
        
        # Create positional encodings
        x_pe = torch.zeros(B, self.pos_encoding_dim, device=spatial_coords.device)
        y_pe = torch.zeros(B, self.pos_encoding_dim, device=spatial_coords.device)
        
        # Apply sinusoidal encoding
        x_pe[:, 0::2] = torch.sin(x_coords * self.div_term[None, :])  # Even dimensions
        x_pe[:, 1::2] = torch.cos(x_coords * self.div_term[None, :])  # Odd dimensions
        y_pe[:, 0::2] = torch.sin(y_coords * self.div_term[None, :])
        y_pe[:, 1::2] = torch.cos(y_coords * self.div_term[None, :])
        
        # Combine x and y positional encodings
        pos_encoding = torch.cat([x_pe, y_pe], dim=1)  # [B, spatial_hidden_dim]
        
        # Add positional encoding to spatial embeddings
        spatial_embed = spatial_embed + pos_encoding
        
        # Fuse histology and spatial features
        condition_features = torch.cat([histology_embed, spatial_embed], dim=1)  # [B, total_dim]
        
        # Final projection to condition embedding space
        condition_embed = self.final_projection(condition_features)  # [B, condition_embed_dim]
        
        return condition_embed
    
    def extra_repr(self) -> str:
        return (f'histology_dim={self.histology_dim}, spatial_dim={self.spatial_dim}, '
                f'condition_embed_dim={self.condition_embed_dim}')


# Legacy components for backward compatibility
class PositionalEncoding(nn.Module):
    """Legacy positional encoding (kept for compatibility)"""
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class GeneVARTransformer(nn.Module):
    """
    Legacy Gene VAR Transformer (kept for backward compatibility)
    
    Note: This is the old implementation. New code should use MultiScaleGeneVAR instead.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # This is kept for backward compatibility but should not be used
        raise NotImplementedError(
            "GeneVARTransformer is deprecated. Use MultiScaleGeneVAR instead."
        )