"""
VAR Transformer for Spatial Transcriptomics

This module implements the autoregressive transformer component of VAR,
adapted for spatial transcriptomics gene expression prediction.

Key Features:
1. Multi-scale autoregressive generation
2. Adaptive layer normalization with conditioning
3. Next-scale prediction for spatial gene expression
4. Complete preservation of VAR transformer architecture

Author: VAR-ST Team
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from typing import Optional, Tuple, Union, List, Dict, Any

from .var_basic_components import (
    AdaLNSelfAttn, AdaLNBeforeHead, SharedAdaLin, 
    sample_with_top_k_top_p_, gumbel_softmax_with_rng
)


class VAR_ST(nn.Module):
    """
    Vector-quantized Autoregressive transformer for Spatial Transcriptomics
    
    This implements the core VAR transformer that performs autoregressive
    generation of spatial gene expression patterns at multiple scales.
    
    Architecture:
    1. Input: discrete tokens from VQVAE + conditioning (histology features)
    2. Multi-scale autoregressive modeling with adaptive layer norm
    3. Output: next-scale gene expression token predictions
    
    Completely preserves the original VAR transformer design for zero performance loss.
    """
    
    def __init__(
        self,
        vae_embed_dim: int = 256,          # VQVAE embedding dimension
        num_classes: int = 1000,           # Number of condition classes (tissue types, etc.)
        depth: int = 16,                   # Number of transformer blocks
        embed_dim: int = 1024,             # Transformer embedding dimension
        num_heads: int = 16,               # Number of attention heads
        mlp_ratio: float = 4.0,            # MLP expansion ratio
        drop_rate: float = 0.0,            # Dropout rate
        attn_drop_rate: float = 0.0,       # Attention dropout rate
        drop_path_rate: float = 0.0,       # Stochastic depth rate
        norm_eps: float = 1e-6,            # Layer norm epsilon
        shared_aln: bool = False,          # Share adaptive layer norm parameters
        cond_drop_rate: float = 0.1,       # Conditioning dropout rate
        attn_l2_norm: bool = False,        # L2 normalize attention
        patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # Multi-scale patch numbers
        flash_if_available: bool = True,   # Use flash attention if available
        fused_if_available: bool = True,   # Use fused operations if available
    ):
        """
        Initialize VAR transformer for spatial transcriptomics
        
        Args:
            vae_embed_dim: Dimension of VQVAE embeddings
            num_classes: Number of conditioning classes
            depth: Number of transformer layers
            embed_dim: Transformer hidden dimension
            num_heads: Number of attention heads per layer
            mlp_ratio: Expansion ratio for MLP layers
            drop_rate: General dropout probability
            attn_drop_rate: Attention dropout probability
            drop_path_rate: Stochastic depth probability
            norm_eps: LayerNorm epsilon value
            shared_aln: Whether to share adaptive layer norm parameters
            cond_drop_rate: Conditioning dropout rate for robust training
            attn_l2_norm: Whether to L2 normalize attention weights
            patch_nums: Sequence of patch numbers for multi-scale generation
            flash_if_available: Use flash attention optimization if available
            fused_if_available: Use fused MLP optimization if available
        """
        super().__init__()
        
        # Store architecture parameters
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.patch_nums = patch_nums
        self.cond_drop_rate = cond_drop_rate
        
        # Multi-scale generation setup
        # Each scale corresponds to a different spatial resolution of gene expression
        self.first_l = patch_nums[0]  # Starting scale
        self.last_l = patch_nums[-1]  # Final scale (highest resolution)
        
        print(f"🚀 初始化 VAR-ST Transformer:")
        print(f"  - 层数: {depth}")
        print(f"  - 嵌入维度: {embed_dim}")
        print(f"  - 注意力头数: {num_heads}")
        print(f"  - 多尺度序列: {patch_nums}")
        print(f"  - 条件类别数: {num_classes}")
        
        # Layer normalization
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        
        # Class embedding for conditioning (e.g., tissue type, sample metadata)
        self.class_emb = nn.Embedding(num_classes, embed_dim)
        self.pos_start = nn.Parameter(torch.empty(embed_dim))
        
        # Token embeddings for each scale
        # Maps discrete tokens from VQVAE to transformer input embeddings
        self.token_embed = nn.ModuleList([
            nn.Embedding(vae_embed_dim, embed_dim) for _ in patch_nums
        ])
        
        # Positional embeddings for each scale
        # Provides spatial awareness for the transformer at different resolutions
        self.pos_embed = nn.ParameterList([
            nn.Parameter(torch.empty(1, pn * pn, embed_dim)) for pn in patch_nums
        ])
        
        # Adaptive layer norm conditioning
        # This is key to VAR's performance - allows dynamic adaptation based on conditioning
        if shared_aln:
            self.shared_ada_lin = SharedAdaLin(embed_dim, 6 * embed_dim)
        else:
            self.shared_ada_lin = nn.Identity()
        
        # Transformer blocks with adaptive layer normalization
        # Each block can adapt its processing based on the conditioning signal
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                block_idx=block_idx,
                last_drop_p=drop_path_rates[-1],
                embed_dim=embed_dim,
                cond_dim=embed_dim,
                shared_aln=shared_aln,
                norm_layer=norm_layer,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rates[block_idx],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available,
                fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        # Output heads for each scale
        # Predicts the next-scale tokens in the autoregressive sequence
        self.head_nm = AdaLNBeforeHead(embed_dim, embed_dim, norm_layer)
        self.head = nn.ModuleList([
            nn.Linear(embed_dim, vae_embed_dim, bias=False) for _ in patch_nums
        ])
        
        # Initialize parameters
        self.init_weights()
        
        # For inference: cached representations and attention states
        self.prog_si = -1  # Current progress in generation sequence
    
    def init_weights(self):
        """Initialize model parameters following VAR's initialization scheme"""
        # Initialize positional embeddings
        for pos_emb in self.pos_embed:
            nn.init.trunc_normal_(pos_emb, std=0.02)
        nn.init.trunc_normal_(self.pos_start, std=0.02)
        
        # Initialize class embedding
        nn.init.trunc_normal_(self.class_emb.weight, std=0.02)
        
        # Initialize token embeddings
        for token_emb in self.token_embed:
            nn.init.trunc_normal_(token_emb.weight, std=0.02)
        
        # Initialize output heads
        for head in self.head:
            nn.init.trunc_normal_(head.weight, std=0.02)
    
    def get_conditioning(self, B: int, class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate conditioning signal for adaptive layer normalization
        
        This conditioning allows the transformer to adapt its processing
        based on sample-specific information (e.g., tissue type, patient metadata).
        
        Args:
            B: Batch size
            class_labels: [B] - class labels for conditioning (optional)
        
        Returns:
            conditioning: [B, embed_dim] - conditioning vector
        """
        if class_labels is None:
            # Default conditioning (e.g., for unconditional generation)
            class_labels = torch.zeros(B, dtype=torch.long, device=self.class_emb.weight.device)
        
        # Apply conditioning dropout during training for robustness
        if self.training and self.cond_drop_rate > 0:
            mask = torch.rand(B, device=class_labels.device) >= self.cond_drop_rate
            class_labels = class_labels * mask.long()
        
        # Get class embeddings
        conditioning = self.class_emb(class_labels)  # [B, embed_dim]
        
        return conditioning
    
    def forward_for_loss(
        self, 
        gt_indices: List[torch.Tensor],
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training loss computation
        
        Implements teacher-forcing training where the model learns to predict
        the next scale given all previous scales.
        
        Args:
            gt_indices: List of ground truth token indices for each scale
                       Each element: [B, Hi, Wi] where Hi, Wi are spatial dims for scale i
            class_labels: [B] - conditioning class labels
        
        Returns:
            loss: scalar - autoregressive prediction loss
        """
        B = gt_indices[0].shape[0]
        
        # Get conditioning signal
        conditioning = self.get_conditioning(B, class_labels)  # [B, embed_dim]
        
        # Apply shared adaptive linear layer if enabled
        if hasattr(self.shared_ada_lin, 'weight'):
            conditioning = self.shared_ada_lin(conditioning)
        
        # Initialize sequence with start token
        sos = self.pos_start.unsqueeze(0).expand(B, 1, -1)  # [B, 1, embed_dim]
        x = sos
        
        total_loss = 0.0
        
        # Process each scale in the multi-scale sequence
        for si, pn in enumerate(self.patch_nums[:-1]):  # Exclude last scale
            # Current scale tokens
            cur_indices = gt_indices[si].view(B, -1)  # [B, pn*pn]
            cur_tokens = self.token_embed[si](cur_indices)  # [B, pn*pn, embed_dim]
            cur_pos = self.pos_embed[si].expand(B, -1, -1)  # [B, pn*pn, embed_dim]
            cur_tokens = cur_tokens + cur_pos
            
            # Concatenate with previous sequence
            x = torch.cat([x, cur_tokens], dim=1)  # [B, seq_len, embed_dim]
            
            # Apply transformer blocks with adaptive conditioning
            for block in self.blocks:
                x = block(x, conditioning, attn_bias=None)
            
            # Predict next scale
            next_pn = self.patch_nums[si + 1]
            next_len = next_pn * next_pn
            
            # Extract representations for next scale prediction
            pred_repr = x[:, -next_len:]  # [B, next_len, embed_dim]
            pred_repr = self.head_nm(pred_repr, conditioning)
            logits = self.head[si + 1](pred_repr)  # [B, next_len, vocab_size]
            
            # Compute cross-entropy loss with ground truth
            gt_next = gt_indices[si + 1].view(B, -1)  # [B, next_len]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), gt_next.view(-1))
            total_loss += loss
        
        return total_loss / (len(self.patch_nums) - 1)
    
    def autoregressive_infer_cfg(
        self,
        B: int,
        class_labels: Optional[torch.Tensor] = None,
        cfg: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        more_smooth: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        """
        Autoregressive inference with classifier-free guidance
        
        Generates spatial gene expression patterns by predicting tokens
        at progressively higher resolutions.
        
        Args:
            B: Batch size
            class_labels: [B] - conditioning class labels
            cfg: Classifier-free guidance scale
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            more_smooth: Whether to use temperature scaling for smoother generation
            rng: Random number generator for reproducible sampling
        
        Returns:
            generated_indices: List of generated token indices for each scale
        """
        # Setup conditioning for classifier-free guidance
        if cfg != 1.0:
            # Duplicate batch: conditional + unconditional
            if class_labels is not None:
                class_labels_cfg = torch.cat([class_labels, torch.zeros_like(class_labels)], dim=0)
            else:
                class_labels_cfg = torch.cat([
                    torch.zeros(B, dtype=torch.long, device=self.class_emb.weight.device),
                    torch.zeros(B, dtype=torch.long, device=self.class_emb.weight.device)
                ], dim=0)
            B_cfg = B * 2
        else:
            class_labels_cfg = class_labels
            B_cfg = B
        
        # Get conditioning
        conditioning = self.get_conditioning(B_cfg, class_labels_cfg)
        if hasattr(self.shared_ada_lin, 'weight'):
            conditioning = self.shared_ada_lin(conditioning)
        
        # Initialize sequence
        sos = self.pos_start.unsqueeze(0).expand(B_cfg, 1, -1)
        x = sos
        
        generated_indices = []
        
        # Generate each scale autoregressively
        for si, pn in enumerate(self.patch_nums):
            cur_len = pn * pn
            
            if si == 0:
                # First scale: predict from start token
                pred_repr = x
            else:
                # Subsequent scales: use accumulated sequence
                pred_repr = x[:, -cur_len:]
            
            # Apply transformer
            for block in self.blocks:
                x_input = x if si == 0 else torch.cat([x[:, :-cur_len], pred_repr], dim=1)
                pred_repr = block(x_input, conditioning, attn_bias=None)[:, -cur_len:]
            
            # Generate tokens for current scale
            pred_repr = self.head_nm(pred_repr, conditioning)
            logits = self.head[si](pred_repr)  # [B_cfg, cur_len, vocab_size]
            
            # Apply classifier-free guidance
            if cfg != 1.0:
                logits_cond, logits_uncond = logits.chunk(2, dim=0)
                logits = logits_uncond + cfg * (logits_cond - logits_uncond)
            
            # Temperature scaling for smoother generation
            if more_smooth:
                logits = logits / 1.5
            
            # Sample tokens
            if top_k > 0 or top_p > 0:
                indices = sample_with_top_k_top_p_(logits, top_k=top_k, top_p=top_p, rng=rng)
                indices = indices.squeeze(-1)  # [B, cur_len]
            else:
                indices = torch.multinomial(F.softmax(logits, dim=-1).view(-1, logits.size(-1)), 
                                          num_samples=1, generator=rng).view(logits.shape[0], -1)
            
            generated_indices.append(indices.view(B, pn, pn))
            
            # Add generated tokens to sequence for next scale
            if si < len(self.patch_nums) - 1:
                gen_tokens = self.token_embed[si](indices)  # [B, cur_len, embed_dim]
                gen_pos = self.pos_embed[si].expand(B, -1, -1)
                gen_tokens = gen_tokens + gen_pos
                x = torch.cat([x, gen_tokens], dim=1)
        
        return generated_indices
    
    def forward(self, indices: List[torch.Tensor], class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard forward pass (delegates to forward_for_loss for training)"""
        return self.forward_for_loss(indices, class_labels)


class VAR(nn.Module):
    """
    简化的VAR模型 - 专门用于单spot基因表达vectors
    
    与VAR_ST不同，这个版本针对基因表达向量的多尺度建模，
    不需要复杂的空间位置编码。
    """
    
    def __init__(
        self,
        vocab_size: int = 8192,            # VQVAE codebook size
        depth: int = 16,                   # Number of transformer blocks
        embed_dim: int = 1024,             # Transformer embedding dimension
        num_heads: int = 16,               # Number of attention heads
        patch_nums: Tuple[int, ...] = (1, 4, 16, 64, 256),  # Gene scales
        rope_theta: float = 10000.0,       # RoPE theta parameter
        dropout: float = 0.0,              # Dropout rate
        drop_path_rate: float = 0.1,       # Drop path rate
        **kwargs
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.patch_nums = patch_nums
        
        print(f"🚀 初始化简化VAR模型:")
        print(f"  - 词汇表大小: {vocab_size}")
        print(f"  - 嵌入维度: {embed_dim}")
        print(f"  - 基因尺度: {patch_nums}")
        
        # Token embeddings for different gene scales
        self.token_embed = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for _ in patch_nums
        ])
        
        # Positional embeddings for different scales
        self.pos_embed = nn.ModuleList([
            nn.Embedding(scale, embed_dim) for scale in patch_nums
        ])
        
        # Start token
        self.pos_start = nn.Parameter(torch.randn(embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(depth)
        ])
        
        # Output heads for each scale
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, vocab_size) for _ in patch_nums
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize parameters"""
        nn.init.normal_(self.pos_start, std=0.02)
        
        for embed in self.token_embed:
            nn.init.normal_(embed.weight, std=0.02)
        
        for embed in self.pos_embed:
            nn.init.normal_(embed.weight, std=0.02)
        
        for head in self.heads:
            nn.init.normal_(head.weight, std=0.02)
    
    def forward_training(
        self,
        tokens: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        cfg: float = 1.0,
        cond_drop_prob: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        训练前向传播
        
        Args:
            tokens: [B, total_tokens] - 连接的所有尺度tokens
            class_labels: [B] - 类别标签（暂时不用）
            cfg: CFG scale
            cond_drop_prob: 条件dropout概率
        
        Returns:
            Dict包含训练loss
        """
        B = tokens.shape[0]
        device = tokens.device
        
        # 分解tokens到不同尺度
        scale_tokens = []
        start_idx = 0
        for scale in self.patch_nums:
            end_idx = start_idx + scale
            if end_idx <= tokens.shape[1]:
                scale_tokens.append(tokens[:, start_idx:end_idx])
            else:
                # 填充不足的tokens
                remaining = end_idx - tokens.shape[1]
                if start_idx < tokens.shape[1]:
                    partial_tokens = tokens[:, start_idx:]
                    pad_tokens = torch.zeros(B, remaining, dtype=torch.long, device=device)
                    scale_tokens.append(torch.cat([partial_tokens, pad_tokens], dim=1))
                else:
                    scale_tokens.append(torch.zeros(B, scale, dtype=torch.long, device=device))
            start_idx = end_idx
        
        # 初始化序列
        start_token = self.pos_start.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)  # [B, 1, embed_dim]
        sequence = start_token
        
        total_loss = 0.0
        num_predictions = 0
        
        # 对每个尺度进行自回归训练
        for i in range(len(scale_tokens) - 1):
            # 当前尺度的tokens
            current_tokens = scale_tokens[i]  # [B, scale_i]
            
            # Token embedding
            token_emb = self.token_embed[i](current_tokens)  # [B, scale_i, embed_dim]
            
            # Position embedding
            pos_indices = torch.arange(self.patch_nums[i], device=device).unsqueeze(0).expand(B, -1)
            pos_emb = self.pos_embed[i](pos_indices)  # [B, scale_i, embed_dim]
            
            # 组合embedding
            current_repr = token_emb + pos_emb  # [B, scale_i, embed_dim]
            
            # 添加到序列
            sequence = torch.cat([sequence, current_repr], dim=1)  # [B, seq_len, embed_dim]
            
            # Transformer处理
            x = sequence
            for block in self.blocks:
                x = block(x)
            
            # 预测下一个尺度
            next_scale = self.patch_nums[i + 1]
            
            # 修复：使用序列的最后hidden state来预测所有下一尺度的tokens
            # 获取最后一个hidden state并扩展到预测所有next_scale个tokens
            last_hidden = x[:, -1:, :]  # [B, 1, embed_dim] - 取最后一个位置
            pred_input = last_hidden.expand(-1, next_scale, -1)  # [B, next_scale, embed_dim]
            pred_input = self.norm(pred_input)
            
            # 输出预测
            logits = self.heads[i + 1](pred_input)  # [B, next_scale, vocab_size]
            
            # 计算loss
            target_tokens = scale_tokens[i + 1]  # [B, next_scale]
            
            # 确保维度匹配
            logits_flat = logits.view(-1, self.vocab_size)  # [B*next_scale, vocab_size]
            target_flat = target_tokens.view(-1)  # [B*next_scale]
            
            print(f"🔍 Loss计算 - 尺度{i}->{i+1}:")
            print(f"   - logits: {logits.shape} -> {logits_flat.shape}")
            print(f"   - targets: {target_tokens.shape} -> {target_flat.shape}")
            
            loss = F.cross_entropy(logits_flat, target_flat)
            total_loss += loss
            num_predictions += 1
        
        # 平均loss
        avg_loss = total_loss / max(1, num_predictions)
        
        return {
            'loss': avg_loss,
            'num_predictions': num_predictions
        }
    
    def autoregressive_infer_cfg(
        self,
        B: int,
        class_labels: Optional[torch.Tensor] = None,
        cfg: float = 1.5,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0,
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        自回归推理生成
        
        Args:
            B: batch size
            class_labels: 类别标签（暂时不用）
            cfg: CFG scale
            top_k: top-k采样
            top_p: top-p采样
            temperature: 采样温度
            generator: 随机数生成器
        
        Returns:
            生成的tokens [B, total_tokens]
        """
        device = next(self.parameters()).device
        
        # 初始化序列
        start_token = self.pos_start.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        sequence = start_token
        
        all_generated = []
        
        # 为每个尺度生成tokens
        for i, scale in enumerate(self.patch_nums):
            if i == 0:
                # 第一个尺度：从start token预测
                x = sequence
                for block in self.blocks:
                    x = block(x)
                
                pred_input = x[:, -1:].expand(-1, scale, -1)  # [B, scale, embed_dim]
            else:
                # 后续尺度：基于之前的序列预测
                x = sequence
                for block in self.blocks:
                    x = block(x)
                
                pred_input = x[:, -scale:]  # [B, scale, embed_dim]
            
            pred_input = self.norm(pred_input)
            logits = self.heads[i](pred_input)  # [B, scale, vocab_size]
            
            # 应用温度
            if temperature != 1.0:
                logits = logits / temperature
            
            # 采样
            if top_k > 0 or top_p > 0:
                # Top-k/top-p采样
                if top_k > 0:
                    top_k_actual = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k_actual)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                if top_p > 0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
            
            # 采样tokens
            probs = F.softmax(logits, dim=-1)
            tokens = torch.multinomial(probs.view(-1, self.vocab_size), 1, generator=generator)
            tokens = tokens.view(B, scale)  # [B, scale]
            
            all_generated.append(tokens)
            
            # 更新序列（除了最后一个尺度）
            if i < len(self.patch_nums) - 1:
                token_emb = self.token_embed[i](tokens)
                pos_indices = torch.arange(scale, device=device).unsqueeze(0).expand(B, -1)
                pos_emb = self.pos_embed[i](pos_indices)
                current_repr = token_emb + pos_emb
                sequence = torch.cat([sequence, current_repr], dim=1)
        
        # 连接所有生成的tokens
        return torch.cat(all_generated, dim=1)  # [B, total_tokens] 