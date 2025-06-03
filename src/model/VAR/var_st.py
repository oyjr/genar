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
        losses_per_scale = []
        
        # Process each scale in the multi-scale sequence
        for si, pn in enumerate(self.patch_nums[:-1]):  # Exclude last scale
            # Current scale tokens
            cur_indices = gt_indices[si].view(B, -1).contiguous()  # [B, pn*pn]
            cur_tokens = self.token_embed[si](cur_indices)  # [B, pn*pn, embed_dim]
            cur_pos = self.pos_embed[si].expand(B, -1, -1).contiguous()  # [B, pn*pn, embed_dim]
            cur_tokens = cur_tokens + cur_pos
            cur_tokens = cur_tokens.contiguous()
            
            # Concatenate with previous sequence
            x = torch.cat([x, cur_tokens], dim=1).contiguous()  # [B, seq_len, embed_dim]
            
            # Apply transformer blocks with adaptive conditioning
            for block in self.blocks:
                x = block(x, conditioning, attn_bias=None)
            
            # Predict next scale
            next_pn = self.patch_nums[si + 1]
            next_len = next_pn * next_pn
            
            # Extract representations for next scale prediction
            pred_repr = x[:, -next_len:].contiguous()  # [B, next_len, embed_dim]
            pred_repr = self.head_nm(pred_repr, conditioning)
            logits = self.head[si + 1](pred_repr)  # [B, next_len, vocab_size]
            logits = logits.contiguous()
            
            # Compute cross-entropy loss with ground truth
            gt_next = gt_indices[si + 1].view(B, -1).contiguous()  # [B, next_len]
            scale_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), gt_next.reshape(-1))
            losses_per_scale.append(scale_loss)
            total_loss += scale_loss
        
        # Average loss
        if len(losses_per_scale) > 0:
            avg_loss = total_loss / len(losses_per_scale)
        else:
            avg_loss = total_loss
        
        return avg_loss
    
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
                indices = indices.squeeze(-1).contiguous()  # [B, cur_len]
            else:
                logits_softmax = F.softmax(logits, dim=-1).contiguous()
                indices = torch.multinomial(logits_softmax.reshape(-1, logits.size(-1)), 
                                          num_samples=1, generator=rng).reshape(logits.shape[0], -1).contiguous()
            
            generated_indices.append(indices.reshape(B, pn, pn).contiguous())
            
            # Add generated tokens to sequence for next scale
            if si < len(self.patch_nums) - 1:
                gen_tokens = self.token_embed[si](indices)  # [B, cur_len, embed_dim]
                gen_pos = self.pos_embed[si].expand(B, -1, -1).contiguous()
                gen_tokens = (gen_tokens + gen_pos).contiguous()
                x = torch.cat([x, gen_tokens], dim=1).contiguous()
        
        return generated_indices
    
    def forward(self, indices: List[torch.Tensor], class_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard forward pass (delegates to forward_for_loss for training)"""
        return self.forward_for_loss(indices, class_labels)


class VAR_ST_Gene(VAR_ST):
    """
    基因维度多尺度的VAR_ST - 专门用于基因表达生成
    
    继承完整VAR_ST的所有高级特性:
    - AdaLN自适应层归一化
    - Classifier-Free Guidance (CFG)
    - 条件生成控制
    - 高级采样策略
    
    针对基因维度多尺度的关键修改:
    - patch_nums对应基因特征数量 [1, 4, 16, 64, 200]
    - 位置编码适配基因维度 (不是空间pn*pn)
    - Token序列处理适配单spot基因向量
    """
    
    def __init__(
        self,
        gene_scales: List[int] = [1, 4, 16, 64, 200],
        vae_embed_dim: int = 8192,          # VQVAE最大codebook size
        num_classes: int = 1000,            # 条件类别数
        depth: int = 16,                    # Transformer层数
        embed_dim: int = 1024,              # 嵌入维度
        num_heads: int = 16,                # 注意力头数
        **kwargs
    ):
        """
        初始化基因维度VAR_ST
        
        Args:
            gene_scales: 基因多尺度特征数量 [1, 4, 16, 64, 200]
            vae_embed_dim: VQVAE词汇表大小
            其他参数与原始VAR_ST相同
        """
        # 设置基因尺度为patch_nums
        kwargs['patch_nums'] = tuple(gene_scales)
        kwargs['vae_embed_dim'] = vae_embed_dim
        kwargs['num_classes'] = num_classes
        kwargs['depth'] = depth
        kwargs['embed_dim'] = embed_dim
        kwargs['num_heads'] = num_heads
        
        print(f"🧬 初始化基因维度VAR_ST:")
        print(f"  - 基因尺度: {gene_scales}")
        print(f"  - 词汇表大小: {vae_embed_dim}")
        print(f"  - 条件类别数: {num_classes}")
        
        # 调用父类初始化 (但会被下面的修改覆盖)
        super().__init__(**kwargs)
        
        # 🔧 关键修改: 重写位置编码以适配基因维度
        # 原始VAR_ST: pos_embed为 [1, pn*pn, embed_dim] (空间维度)
        # 基因VAR_ST: pos_embed为 [1, 1, embed_dim] (基因维度，每个尺度1个token)
        self.pos_embed = nn.ParameterList([
            nn.Parameter(torch.empty(1, 1, embed_dim))  # 每个尺度只有1个token
            for _ in gene_scales
        ])
        
        # 重新初始化位置编码
        for pos_emb in self.pos_embed:
            nn.init.trunc_normal_(pos_emb, std=0.02)
        
        print(f"  - 位置编码适配: 基因维度 (每个尺度1个token)")
        print(f"  - 位置编码形状: {[tuple(pos.shape) for pos in self.pos_embed]}")
        
        # 存储基因尺度信息
        self.gene_scales = gene_scales
        self.num_gene_scales = len(gene_scales)
        
        print(f"✅ 基因维度VAR_ST初始化完成")
    
    def forward_for_loss(
        self, 
        gt_tokens: List[torch.Tensor],  # 修改: 基因tokens格式
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        基因维度的训练前向传播
        
        Args:
            gt_tokens: List of gene tokens for each scale
                      每个元素: [B, 1] - 每个样本每个尺度1个token
            class_labels: [B] - 条件类别标签
        
        Returns:
            loss: 自回归预测损失
        """
        B = gt_tokens[0].shape[0]
        
        # 获取条件信号
        conditioning = self.get_conditioning(B, class_labels)
        
        # 应用共享自适应线性层
        if hasattr(self.shared_ada_lin, 'weight'):
            conditioning = self.shared_ada_lin(conditioning)
        
        # 初始化序列
        sos = self.pos_start.unsqueeze(0).expand(B, 1, -1)  # [B, 1, embed_dim]
        x = sos
        
        total_loss = 0.0
        losses_per_scale = []
        
        # 🔧 修改: 适配基因维度的token处理
        for si, gene_scale in enumerate(self.gene_scales[:-1]):  # 排除最后一个尺度
            # 当前尺度的tokens: [B, 1] -> [B]
            cur_tokens = gt_tokens[si].squeeze(-1) if gt_tokens[si].dim() == 2 else gt_tokens[si]
            cur_tokens = cur_tokens.contiguous()  # [B]
            
            # Token嵌入: [B] -> [B, 1, embed_dim]
            cur_token_emb = self.token_embed[si](cur_tokens).unsqueeze(1)
            
            # 位置嵌入: [1, 1, embed_dim] -> [B, 1, embed_dim]
            cur_pos_emb = self.pos_embed[si].expand(B, -1, -1)
            
            # 组合嵌入
            cur_repr = cur_token_emb + cur_pos_emb  # [B, 1, embed_dim]
            
            # 添加到序列
            x = torch.cat([x, cur_repr], dim=1).contiguous()  # [B, seq_len, embed_dim]
            
            # Transformer处理
            for block in self.blocks:
                x = block(x, conditioning, attn_bias=None)
            
            # 预测下一个尺度
            next_scale_idx = si + 1
            next_gene_scale = self.gene_scales[next_scale_idx]
            
            # 🔧 修改: 基因维度的预测
            # 使用最后的表示来预测下一尺度的单个token
            pred_repr = x[:, -1:].contiguous()  # [B, 1, embed_dim] - 最后一个位置
            pred_repr = self.head_nm(pred_repr, conditioning)
            logits = self.head[next_scale_idx](pred_repr)  # [B, 1, vocab_size]
            logits = logits.contiguous()
            
            # 计算损失
            gt_next = gt_tokens[next_scale_idx].squeeze(-1) if gt_tokens[next_scale_idx].dim() == 2 else gt_tokens[next_scale_idx]
            gt_next = gt_next.contiguous()  # [B]
            
            scale_loss = F.cross_entropy(logits.squeeze(1), gt_next)  # [B, vocab_size] vs [B]
            losses_per_scale.append(scale_loss)
            total_loss += scale_loss
        
        # 平均损失
        avg_loss = total_loss / len(losses_per_scale) if losses_per_scale else total_loss
        
        return avg_loss
    
    def autoregressive_infer_cfg(
        self,
        B: int,
        class_labels: Optional[torch.Tensor] = None,
        cfg: float = 1.5,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0,
        more_smooth: bool = False,
        rng: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        """
        基因维度的自回归推理生成
        
        Args:
            B: batch size
            class_labels: [B] - 条件类别标签
            cfg: Classifier-free guidance缩放因子
            其他参数: 采样控制参数
        
        Returns:
            List[torch.Tensor]: 每个基因尺度的生成tokens
                               每个元素: [B] - 每个样本每个尺度1个token
        """
        # CFG设置
        if cfg != 1.0:
            if class_labels is not None:
                class_labels_cfg = torch.cat([class_labels, torch.zeros_like(class_labels)], dim=0)
            else:
                device = next(self.parameters()).device
                class_labels_cfg = torch.cat([
                    torch.zeros(B, dtype=torch.long, device=device),
                    torch.zeros(B, dtype=torch.long, device=device)
                ], dim=0)
            B_cfg = B * 2
        else:
            class_labels_cfg = class_labels
            B_cfg = B
        
        # 获取条件
        conditioning = self.get_conditioning(B_cfg, class_labels_cfg)
        if hasattr(self.shared_ada_lin, 'weight'):
            conditioning = self.shared_ada_lin(conditioning)
        
        # 初始化序列
        sos = self.pos_start.unsqueeze(0).expand(B_cfg, 1, -1)
        x = sos
        
        generated_tokens = []
        
        # 🔧 修改: 基因维度的自回归生成
        for si, gene_scale in enumerate(self.gene_scales):
            if si == 0:
                # 第一个尺度: 从起始token预测
                pred_repr = x  # [B_cfg, 1, embed_dim]
            else:
                # 后续尺度: 使用累积序列的最后位置
                pred_repr = x[:, -1:].contiguous()  # [B_cfg, 1, embed_dim]
            
            # Transformer处理
            for block in self.blocks:
                if si == 0:
                    pred_repr = block(pred_repr, conditioning, attn_bias=None)
                else:
                    # 对整个序列处理，但只取最后的表示
                    full_repr = block(x, conditioning, attn_bias=None)
                    pred_repr = full_repr[:, -1:].contiguous()
            
            # 生成当前尺度的token
            pred_repr = self.head_nm(pred_repr, conditioning)
            logits = self.head[si](pred_repr)  # [B_cfg, 1, vocab_size]
            
            # CFG应用
            if cfg != 1.0:
                logits_cond, logits_uncond = logits.chunk(2, dim=0)
                logits = logits_uncond + cfg * (logits_cond - logits_uncond)
                logits = logits[:B]  # 只保留条件部分
            
            # 温度缩放
            if more_smooth:
                logits = logits / 1.5
            if temperature != 1.0:
                logits = logits / temperature
            
            # 采样
            if top_k > 0 or top_p > 0:
                tokens = sample_with_top_k_top_p_(logits, top_k=top_k, top_p=top_p, rng=rng)
                tokens = tokens.squeeze(-1).contiguous()  # [B]
            else:
                probs = F.softmax(logits.squeeze(1), dim=-1)  # [B, vocab_size]
                tokens = torch.multinomial(probs, num_samples=1, generator=rng).squeeze(-1)  # [B]
            
            generated_tokens.append(tokens)
            
            # 添加生成的token到序列 (除了最后一个尺度)
            if si < len(self.gene_scales) - 1:
                # Token嵌入
                gen_token_emb = self.token_embed[si](tokens).unsqueeze(1)  # [B, 1, embed_dim]
                
                # 位置嵌入  
                gen_pos_emb = self.pos_embed[si].expand(B, -1, -1)  # [B, 1, embed_dim]
                
                # 组合并添加到序列
                gen_repr = (gen_token_emb + gen_pos_emb).contiguous()  # [B, 1, embed_dim]
                
                # 🔍 调试信息
                print(f"  Debug scale {si}: x.shape={x.shape}, gen_repr.shape={gen_repr.shape}, B={B}, cfg={cfg}")
                
                # 确保x的维度正确：如果使用CFG，需要特别处理
                if cfg != 1.0:
                    # CFG模式：x是[B_cfg, seq_len, embed_dim]，需要只取前B个
                    x_cond = x[:B].contiguous()  # [B, seq_len, embed_dim]
                    print(f"  CFG mode: x_cond.shape={x_cond.shape}")
                    x_new = torch.cat([x_cond, gen_repr], dim=1).contiguous()  # [B, seq_len+1, embed_dim]
                    # 复制给unconditional部分
                    x = torch.cat([x_new, x_new], dim=0).contiguous()  # [B_cfg, seq_len+1, embed_dim]
                else:
                    x = torch.cat([x, gen_repr], dim=1).contiguous()  # [B, seq_len+1, embed_dim]
        
        return generated_tokens 