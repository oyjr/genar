"""
Multi-Scale Gene VAR for Spatial Transcriptomics

This module implements a multi-scale VAR model for spatial transcriptomics
based on the original VAR architecture. The model uses cumulative multi-scale
generation to predict gene expressions from histology features and spatial coordinates.

Key Features:
- Multi-scale cumulative generation (like original VAR)
- AdaLN conditioning for deep feature fusion
- Residual accumulation across scales
- KV caching for efficient inference
- FiLM-based dynamic gene identity modulation

Author: Assistant
Date: 2024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from functools import partial

# 软标签类型定义
SoftTarget = Dict[str, torch.Tensor]  # 软标签字典类型
HierarchicalTargets = List[Union[torch.Tensor, SoftTarget]]  # 层次化目标类型

from .gene_var_transformer import (
    GeneAdaLNSelfAttn, 
    GeneAdaLNBeforeHead, 
    ConditionProcessor,
    DropPath
)
from .film_layer import FiLMLayer

logger = logging.getLogger(__name__)


class GeneGroupUpsampling(nn.Module):
    """
    基于基因分组关系的上采样模块
    考虑到每个token代表的基因组的包含关系
    """
    
    def __init__(self, embed_dim: int, scale_dims: Tuple[int, ...], num_genes: int = 200):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale_dims = scale_dims
        self.num_genes = num_genes
        
        # 预计算每个尺度的基因分组映射
        self.group_mappings = self._compute_group_mappings()
        
        # 为每个尺度转换创建学习式上采样层
        self.upsample_transforms = nn.ModuleDict()
        for i in range(len(scale_dims) - 1):
            self.upsample_transforms[f'scale_{i}_to_{i+1}'] = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.1)
            )
        
        logger.info(f"🧬 Gene Group Upsampling initialized:")
        logger.info(f"   Scale dims: {scale_dims}")
        logger.info(f"   Number of upsampling transforms: {len(self.upsample_transforms)}")
        for key in self.group_mappings:
            logger.info(f"   {key}: {len(self.group_mappings[key])} mappings")
    
    def _compute_group_mappings(self):
        """
        计算每个尺度之间的基因分组映射关系
        """
        mappings = {}
        
        for i in range(len(self.scale_dims) - 1):
            source_dim = self.scale_dims[i]
            target_dim = self.scale_dims[i + 1]
            
            # 计算从source_dim到target_dim的映射
            # 每个source token对应多少个target tokens
            genes_per_source = self.num_genes // source_dim
            genes_per_target = self.num_genes // target_dim
            targets_per_source = genes_per_source // genes_per_target
            
            mapping = []
            for source_idx in range(source_dim):
                # source_idx对应的target indices
                start_target = source_idx * targets_per_source
                end_target = start_target + targets_per_source
                target_indices = list(range(start_target, min(end_target, target_dim)))
                mapping.append(target_indices)
            
            mappings[f'scale_{i}_to_{i+1}'] = mapping
            
        return mappings
    
    def forward(self, source_embeddings: torch.Tensor, source_scale_idx: int, target_scale_idx: int):
        """
        基于分组关系进行上采样
        
        Args:
            source_embeddings: [B, source_dim, embed_dim]
            source_scale_idx: 源尺度索引
            target_scale_idx: 目标尺度索引
            
        Returns:
            upsampled_embeddings: [B, target_dim, embed_dim]
        """
        if source_embeddings is None:
            target_dim = self.scale_dims[target_scale_idx]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return torch.zeros(1, target_dim, self.embed_dim, device=device)
        
        B, source_dim, embed_dim = source_embeddings.shape
        target_dim = self.scale_dims[target_scale_idx]
        
        # 获取映射关系
        mapping_key = f'scale_{source_scale_idx}_to_{target_scale_idx}'
        if mapping_key not in self.group_mappings:
            # 跨尺度映射，使用插值
            return self._interpolate_upsample(source_embeddings, target_dim)
        
        mapping = self.group_mappings[mapping_key]
        
        # 基于分组关系进行上采样
        upsampled = torch.zeros(B, target_dim, embed_dim, device=source_embeddings.device)
        
        for source_idx, target_indices in enumerate(mapping):
            if source_idx < source_dim:
                source_emb = source_embeddings[:, source_idx, :]  # [B, embed_dim]
                
                # 学习式变换
                if mapping_key in self.upsample_transforms:
                    transformed_emb = self.upsample_transforms[mapping_key](source_emb)
                else:
                    transformed_emb = source_emb
                
                # 分配到对应的target positions
                for target_idx in target_indices:
                    if target_idx < target_dim:
                        upsampled[:, target_idx, :] = transformed_emb
        
        return upsampled
    
    def _interpolate_upsample(self, source_embeddings, target_dim):
        """插值上采样作为fallback"""
        _, source_dim, _ = source_embeddings.shape
        
        if source_dim == 1:
            return source_embeddings.expand(-1, target_dim, -1)
        
        # 使用线性插值
        source_embeddings_t = source_embeddings.transpose(1, 2)  # [B, embed_dim, source_dim]
        upsampled = F.interpolate(source_embeddings_t, size=target_dim, mode='linear', align_corners=False)
        return upsampled.transpose(1, 2)  # [B, target_dim, embed_dim]


class MultiScaleGeneVAR(nn.Module):
    """
    Hierarchical Gene VAR for Spatial Transcriptomics with Semantic-Aware Embeddings
    
    This model implements a hierarchical generation process with improved position embeddings
    that address semantic mismatches between different scales. Key improvements include:
    
    Architecture:
    - Condition Processor: Encodes histology and spatial features
    - Hierarchical Position Embeddings: Scale-specific embeddings that preserve semantic meaning
    - Hierarchical Generation: Sequentially refines predictions across scales (e.g., 1 → 4 → 8 → 40 → 100 → 200)
    - AdaLN Transformer: Core computation block with deep conditioning
    - Teacher Forcing: Uses ground truth averages at each scale to guide the next
    
    Key Features:
    - Multi-scale cumulative generation (like original VAR)
    - Semantic-aware position embeddings for each scale
    - AdaLN conditioning for deep feature fusion
    - Residual accumulation across scales
    - KV caching for efficient inference
    - Soft label training for information preservation
    
    Embedding Innovation:
    - Each scale has dedicated position embeddings with appropriate semantic meaning
    - Intermediate scales use pool position embeddings (representing gene groups)
    - Final scale uses gene identity embeddings (representing individual genes)
    - This eliminates semantic confusion where same position represents different biology
    """
    
    def __init__(
        self,
        # Gene-related parameters
        vocab_size: int,
        num_genes: int = 200,
        scale_dims: Tuple[int, ...] = (1, 4, 8, 40, 100, 200),
        
        # Model architecture parameters
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        
        # Dropout parameters
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        
        # Condition-related parameters
        histology_feature_dim: int = 1024,
        spatial_coord_dim: int = 2,
        condition_embed_dim: int = 768,
        cond_drop_rate: float = 0.1,
        
        # Other parameters
        norm_eps: float = 1e-6,
        shared_aln: bool = False,
        attn_l2_norm: bool = True,
        device: str = 'cuda',
        adaptive_sigma_alpha: float = 0.1,  # Proportional factor for adaptive sigma
        adaptive_sigma_beta: float = 1.0    # Base value for adaptive sigma
    ):
        super().__init__()
        
        # Store key parameters
        self.num_genes = num_genes
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cond_drop_rate = cond_drop_rate
        self.device = device
        self.adaptive_sigma_alpha = adaptive_sigma_alpha  # Store adaptive sigma parameters
        self.adaptive_sigma_beta = adaptive_sigma_beta
        
        # Hierarchical scale configuration
        self.scale_dims = scale_dims
        self.num_scales = len(scale_dims)
        
        # Store other config parameters for checkpointing
        self.histology_feature_dim = histology_feature_dim
        self.spatial_coord_dim = spatial_coord_dim
        self.condition_embed_dim = condition_embed_dim
        
        # Log the new hierarchical configuration
        logger.info(f"🧬 Hierarchical scale dimensions: {self.scale_dims}")
        logger.info(f"🔢 Number of scales: {self.num_scales}")
        
        # Condition processor
        self.condition_processor = ConditionProcessor(
            histology_dim=histology_feature_dim,
            spatial_dim=spatial_coord_dim,
            condition_embed_dim=condition_embed_dim
        )
        
        # Gene token embedding (for expression counts)
        self.gene_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # NEW: Unified gene identity embedding as modulation condition
        self.gene_identity_embedding = nn.Embedding(num_genes, embed_dim)
        
        # NEW: FiLM layer for dynamic gene-specific modulation
        self.film_layer = FiLMLayer(
            condition_dim=embed_dim,
            feature_dim=embed_dim,
            hidden_dim=embed_dim // 2
        )
        
        # NEW: Gene group upsampling module for intelligent target position initialization
        self.gene_upsampling = GeneGroupUpsampling(
            embed_dim=embed_dim,
            scale_dims=scale_dims,
            num_genes=num_genes
        )
        
        # Hierarchical position embedding - separate embedding for each scale
        # Updated to support cumulative input from all previous scales
        self.hierarchical_pos_embedding = nn.ModuleDict()
        for i, dim in enumerate(self.scale_dims):
            # Calculate maximum sequence length for this scale:
            # start_token + all previous scales + current scale
            if dim == self.num_genes:
                # For the final scale, add extra positions for all target genes
                # This follows VAR's approach: cumulative_context + new_target_positions
                max_cumulative_length = 1 + sum(self.scale_dims[:i]) + self.num_genes
            else:
                # For intermediate scales, use normal cumulative length
                max_cumulative_length = 1 + sum(self.scale_dims[:i+1])
            self.hierarchical_pos_embedding[f'scale_{i}'] = nn.Embedding(max_cumulative_length, embed_dim)
        
        logger.info(f"🔧 Created hierarchical position embeddings for VAR-style input:")
        for i, dim in enumerate(self.scale_dims):
            if dim == self.num_genes:
                max_length = 1 + sum(self.scale_dims[:i]) + self.num_genes
                logger.info(f"   Scale {i} (dim={dim}): max {max_length} positions (VAR-style: context + targets)")
            else:
                max_length = 1 + sum(self.scale_dims[:i+1])
                logger.info(f"   Scale {i} (dim={dim}): max {max_length} positions (cumulative)")
        
        logger.info(f"🚀 VAR-ST Improvements Applied:")
        logger.info(f"   ✅ Gene Group Upsampling: Intelligent target position initialization")
        logger.info(f"   ✅ Scale Embedding Storage: For progressive information transfer")
        logger.info(f"   ✅ Weighted Identity Fusion: Final scale combines upsampling + gene identity")
        
        # Scale embedding to distinguish different scales
        self.scale_embedding = nn.Embedding(self.num_scales, embed_dim)
        
        # Single start token for initiating the generation process
        self.start_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer backbone with AdaLN
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.transformer_blocks = nn.ModuleList([
            GeneAdaLNSelfAttn(
                block_idx=i,
                embed_dim=embed_dim,
                condition_dim=condition_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                norm_eps=norm_eps,
                shared_aln=shared_aln,
                attn_l2_norm=attn_l2_norm,
            )
            for i in range(num_layers)
        ])
        
        # Output head with AdaLN
        self.head_norm = GeneAdaLNBeforeHead(embed_dim, condition_embed_dim, norm_eps)
        self.output_head = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
        
        # Log detailed parameter information
        total_params = self._count_parameters()
        identity_params = self.gene_identity_embedding.num_embeddings * self.gene_identity_embedding.embedding_dim
        film_params = sum(p.numel() for p in self.film_layer.parameters())
        
        logger.info(f"✅ Hierarchical Gene VAR initialized successfully")
        logger.info(f"🎬 FiLM layer for dynamic gene identity modulation:")
        logger.info(f"   - Gene identity embedding: [{num_genes}, {embed_dim}]")
        logger.info(f"   - FiLM hidden dimension: {embed_dim // 2}")
        logger.info(f"📊 Parameter breakdown:")
        logger.info(f"   - Total parameters: ~{total_params/1e6:.1f}M")
        logger.info(f"   - Gene identity embedding: {identity_params:,} ({identity_params/1e3:.1f}K)")
        logger.info(f"   - FiLM layer: {film_params:,} ({film_params/1e3:.1f}K)")
        logger.info(f"   - New parameters ratio: {(identity_params + film_params)/total_params*100:.2f}%")
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _get_hierarchical_position_embedding(self, scale_idx: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        获取指定尺度的位置嵌入
        
        这个方法为不同尺度提供语义正确的位置嵌入：
        - 中间尺度：pool位置嵌入，表示基因组的聚合位置
        - 最终尺度：基因身份嵌入，表示具体基因的身份特征
        
        Args:
            scale_idx: 当前尺度索引 (0 to num_scales-1)
            seq_len: 序列长度，包含start token
            device: 张量设备
            
        Returns:
            位置嵌入张量 [1, seq_len, embed_dim]
        """
        # 选择当前尺度的专用位置嵌入层
        embedding_layer = self.hierarchical_pos_embedding[f'scale_{scale_idx}']
        
        # 生成位置索引：[0, 1, 2, ..., seq_len-1]
        pos_indices = torch.arange(seq_len, device=device)
        
        # 获取位置嵌入 [seq_len, embed_dim]
        pos_embed = embedding_layer(pos_indices)
        
        # 扩展批次维度 [1, seq_len, embed_dim]
        return pos_embed.unsqueeze(0)
    
    def init_weights(self, init_std: float = 0.02):
        """Initialize model weights following VAR initialization"""
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=init_std)
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
        
        self.apply(_init_weights)
        logger.info("🎯 Model weights initialized with hierarchical position embeddings")

    def _create_hierarchical_targets(self, target_genes: torch.Tensor) -> HierarchicalTargets:
        """
        Creates hierarchical ground truth targets using soft labels for intermediate scales.
        
        This method solves the information loss problem in the original implementation by:
        1. Using soft labels (floor/ceil + weights) for intermediate scales instead of hard rounding
        2. Preserving the complete information from floating-point pooled values
        3. Only using hard labels for the final, full-resolution scale

        Args:
            target_genes (torch.Tensor): The ground truth gene expressions, shape [B, 200].

        Returns:
            HierarchicalTargets: A list containing either hard targets (torch.Tensor) for the final scale
                               or soft targets (Dict[str, torch.Tensor]) for intermediate scales.
                               Soft targets contain:
                               - 'floor_targets': Lower bound token IDs
                               - 'ceil_targets': Upper bound token IDs  
                               - 'weights': Interpolation weights (0.0 to 1.0)
        """
        hierarchical_targets = []

        # Ensure target_genes is float for pooling operations
        target_genes_float = target_genes.float().unsqueeze(1) # -> [B, 1, 200]

        for _, dim in enumerate(self.scale_dims):
            if dim == self.num_genes:
                # The final scale uses hard labels (original behavior)
                hard_targets = target_genes.long()
                hard_targets = torch.clamp(hard_targets, 0, self.vocab_size - 1)
                hierarchical_targets.append(hard_targets)
            else:
                # Intermediate scales use soft labels to preserve information
                pooled_targets = F.adaptive_avg_pool1d(target_genes_float, output_size=dim)
                pooled_targets = pooled_targets.squeeze(1) # -> [B, dim]
                
                # Generate soft labels: floor + ceil + interpolation weight
                floor_targets = torch.floor(pooled_targets).long()
                ceil_targets = torch.ceil(pooled_targets).long()
                weights = pooled_targets - floor_targets.float()  # Interpolation weights [0.0, 1.0]
                
                # Handle boundary conditions to ensure valid vocab indices
                floor_targets = torch.clamp(floor_targets, 0, self.vocab_size - 1)
                ceil_targets = torch.clamp(ceil_targets, 0, self.vocab_size - 1)
                
                # Special case: when ceil would exceed vocab boundary, merge weight into floor
                boundary_mask = ceil_targets >= self.vocab_size
                if boundary_mask.any():
                    weights = torch.where(boundary_mask, torch.zeros_like(weights), weights)
                    ceil_targets = torch.where(boundary_mask, floor_targets, ceil_targets)
                
                soft_target = {
                    'floor_targets': floor_targets,
                    'ceil_targets': ceil_targets,
                    'weights': weights
                }
                
                hierarchical_targets.append(soft_target)
            
        return hierarchical_targets

    def _create_gaussian_target_distribution(self, target: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Create adaptive Gaussian target distribution for final scale loss computation.
        
        This method constructs a Gaussian probability distribution centered at the true gene 
        expression values with adaptive sigma that scales with expression level.
        High expression genes get larger sigma (more tolerance), low expression genes get 
        smaller sigma (stricter requirements).
        
        Args:
            target (torch.Tensor): Hard target labels, shape [B, seq_len]
            device (torch.device): Device to create tensors on
            
        Returns:
            torch.Tensor: Gaussian target distribution, shape [B, seq_len, vocab_size]
        """
        vocab_size = self.vocab_size
        
        # Create vocabulary indices tensor [vocab_size]
        vocab_indices = torch.arange(vocab_size, device=device, dtype=torch.float32)
        
        # Expand target to [B, seq_len, 1] for broadcasting
        mu = target.float().unsqueeze(-1)  # [B, seq_len, 1]
        
        # --- ADAPTIVE SIGMA COMPUTATION ---
        # sigma = alpha * mu + beta
        # This allows high expression genes to have larger tolerance
        sigma = self.adaptive_sigma_alpha * mu + self.adaptive_sigma_beta
        
        # Ensure numerical stability by clamping sigma to reasonable bounds
        sigma = torch.clamp(sigma, min=1e-6, max=100.0)  # Prevent extreme values
        # --- END ADAPTIVE SIGMA ---
        
        # Expand vocab_indices to [1, 1, vocab_size] for broadcasting
        x = vocab_indices.view(1, 1, -1)  # [1, 1, vocab_size]
        
        # Compute Gaussian probabilities: exp(-(x - mu)^2 / (2 * sigma^2))
        # Broadcasting: [B, seq_len, 1] and [1, 1, vocab_size] -> [B, seq_len, vocab_size]
        # Note: sigma now has shape [B, seq_len, 1], enabling per-gene adaptive scaling
        squared_diff = (x - mu) ** 2
        gaussian_unnormalized = torch.exp(-squared_diff / (2 * sigma ** 2))
        
        # Handle potential numerical issues
        gaussian_unnormalized = torch.clamp(gaussian_unnormalized, min=1e-10, max=1.0)
        
        # Normalize to create valid probability distribution
        # Sum over vocab dimension and ensure non-zero normalization
        normalization_factor = gaussian_unnormalized.sum(dim=-1, keepdim=True)
        normalization_factor = torch.clamp(normalization_factor, min=1e-10)
        
        target_dist = gaussian_unnormalized / normalization_factor
        
        # Final verification and fallback
        if torch.isnan(target_dist).any() or torch.isinf(target_dist).any():
            logger.warning("NaN or Inf detected in adaptive Gaussian target distribution, using uniform fallback")
            # Fallback to uniform distribution around the target
            target_dist = torch.zeros_like(gaussian_unnormalized)
            target_indices = target.long().unsqueeze(-1)  # [B, seq_len, 1]
            target_dist.scatter_(-1, target_indices, 1.0)
        
        return target_dist

    def _compute_soft_label_loss(self, logits: torch.Tensor, target: Union[torch.Tensor, SoftTarget]) -> torch.Tensor:
        """
        Compute loss for either hard labels (cross-entropy) or soft labels (KL divergence).
        
        This method enables information-preserving training by:
        1. Using standard cross-entropy for hard labels (final scale)
        2. Using KL divergence for soft labels (intermediate scales)
        3. Constructing target probability distributions that preserve floating-point precision
        
        Args:
            logits (torch.Tensor): Model predictions, shape [B, seq_len, vocab_size]
            target (Union[torch.Tensor, SoftTarget]): Either hard targets or soft target dict
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Handle hard labels (final scale) - Use Gaussian KL divergence loss
        if isinstance(target, torch.Tensor):
            # Create Gaussian target distribution centered at true values
            target_dist = self._create_gaussian_target_distribution(target, logits.device)
            
            # Compute log probabilities for KL divergence
            log_probs = F.log_softmax(logits, dim=-1)  # [B, seq_len, vocab_size]
            
            # Additional numerical stability checks
            if torch.isinf(log_probs).any():
                logger.warning("Inf detected in log probabilities for final scale, clipping values")
                log_probs = torch.clamp(log_probs, min=-50.0, max=50.0)
            
            # Compute KL divergence: KL(target_dist || predicted_dist)
            kl_loss = F.kl_div(log_probs, target_dist, reduction='batchmean', log_target=False)
            
            # Final sanity check and fallback
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                logger.warning("Invalid KL loss detected for final scale, falling back to cross-entropy")
                return F.cross_entropy(logits.reshape(-1, self.vocab_size), target.reshape(-1))
            
            return kl_loss
        
        # Handle soft labels (intermediate scales)
        if not isinstance(target, dict) or 'floor_targets' not in target:
            raise ValueError("Soft target must be a dict containing 'floor_targets', 'ceil_targets', 'weights'")
        
        floor_targets = target['floor_targets']  # [B, seq_len]
        ceil_targets = target['ceil_targets']    # [B, seq_len]
        weights = target['weights']              # [B, seq_len], interpolation weights
        
        B, seq_len, _ = logits.shape
        target_seq_len = floor_targets.shape[1]  # Get the actual target sequence length
        
        # Ensure logits and targets have matching sequence dimensions
        if seq_len != target_seq_len:
            raise ValueError(f"Logits seq_len ({seq_len}) doesn't match target seq_len ({target_seq_len})")
        
        # Compute log probabilities for KL divergence
        log_probs = F.log_softmax(logits, dim=-1)  # [B, seq_len, vocab_size]
        
        # Construct target probability distribution
        # P(floor) = 1 - weight, P(ceil) = weight, P(others) = 0
        target_dist = torch.zeros_like(log_probs)  # [B, seq_len, vocab_size]
        
        # Create indices for scatter operations using the actual target dimensions
        batch_indices = torch.arange(B, device=logits.device).unsqueeze(1).expand(-1, target_seq_len)
        seq_indices = torch.arange(target_seq_len, device=logits.device).unsqueeze(0).expand(B, -1)
        
        # Set probabilities for floor targets: P(floor) = 1 - weight
        floor_probs = 1.0 - weights  # [B, target_seq_len]
        target_dist[batch_indices, seq_indices, floor_targets] = floor_probs
        
        # Set probabilities for ceil targets: P(ceil) = weight
        # Only when ceil != floor (avoid double-counting when pooled value is integer)
        ceil_mask = (ceil_targets != floor_targets)
        if ceil_mask.any():
            ceil_probs = weights * ceil_mask.float()  # [B, target_seq_len]
            target_dist[batch_indices, seq_indices, ceil_targets] = ceil_probs
        
        # Add small epsilon for numerical stability and ensure valid probability distribution
        eps = 1e-8
        target_dist = target_dist + eps
        target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True)  # Renormalize
        
        # Additional numerical stability checks
        if torch.isnan(target_dist).any():
            logger.warning("NaN detected in target distribution, falling back to hard labels")
            # Fallback to hard cross-entropy with floor targets
            return F.cross_entropy(logits.reshape(-1, self.vocab_size), floor_targets.reshape(-1))
        
        if torch.isinf(log_probs).any():
            logger.warning("Inf detected in log probabilities, clipping values")
            log_probs = torch.clamp(log_probs, min=-50.0, max=50.0)
        
        # Compute KL divergence: KL(target_dist || predicted_dist)
        # Note: PyTorch's kl_div expects log_probs as first argument and target as second
        kl_loss = F.kl_div(log_probs, target_dist, reduction='batchmean', log_target=False)
        
        # Final sanity check on loss value
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            logger.warning("Invalid KL loss detected, falling back to cross-entropy")
            return F.cross_entropy(logits.reshape(-1, self.vocab_size), floor_targets.reshape(-1))
        
        return kl_loss

    def forward(
        self,
        histology_features: torch.Tensor,   # [B, 1024]
        spatial_coords: torch.Tensor,       # [B, 2]
        target_genes: Optional[torch.Tensor] = None,  # [B, 200] for training
        top_k: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass for the model.
        Dispatches to either training or inference pass based on the model's training state.
        """
        # Condition embedding
        condition_embed = self.condition_processor(histology_features, spatial_coords)
        
        # Dispatch based on the model's training state and presence of target_genes
        if self.training:
            # Training mode: use teacher forcing, target_genes must be provided
            if target_genes is None:
                raise ValueError("target_genes must be provided during training.")
            return self.forward_training(condition_embed, target_genes)
        else:
            # Evaluation/Inference mode
            if target_genes is not None:
                # If targets are provided (e.g., for loss calculation in validation)
                # still use forward_training which involves teacher forcing
                return self.forward_training(condition_embed, target_genes)
            else:
                # Pure inference without targets, possibly with sampling
                return self.forward_inference(condition_embed, top_k=top_k)
    
    def forward_training(
        self,
        condition_embed: torch.Tensor,      # [B, condition_embed_dim]
        target_genes: torch.Tensor          # [B, num_genes]
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical training pass with teacher forcing using soft labels.
        
        This method implements an information-preserving training strategy:
        1. At each intermediate scale, soft labels preserve floating-point precision from pooling
        2. Soft labels use KL divergence loss instead of hard cross-entropy
        3. The final scale uses hard labels for discrete token prediction
        4. Each scale receives ground truth from the previous scale (teacher forcing)
        
        The soft label mechanism solves the critical information loss problem where
        pooling [100, 0] → 50.0 would previously be rounded to hard target 50,
        but now preserves the distribution information via floor/ceil targets.
        """
        B = condition_embed.shape[0]
        device = condition_embed.device
        
        # 1. Create all hierarchical targets with soft labels for intermediate scales
        # This preserves floating-point precision instead of lossy rounding
        hierarchical_targets = self._create_hierarchical_targets(target_genes)
        
        # Initialize storage for scale processing
        scale_embeddings = []  # NEW: Store embeddings from each scale for upsampling
        total_loss = 0.0
        final_predictions = None
        final_loss = torch.tensor(0.0, device=device) # Initialize final_loss
        
        # 2. Iteratively train each scale
        for scale_idx, (scale_dim, scale_target) in enumerate(zip(self.scale_dims, hierarchical_targets)):
            # 🔧 NEW: Build cumulative input from ALL previous scales (not just the last one)
            if scale_idx == 0:
                # For the first scale, input is just the start token
                x = self.start_token.expand(B, -1, -1) # [B, 1, D]
            else:
                # For subsequent scales, use ALL previous scale tokens (cumulative input)
                cumulative_input_tokens = []
                
                # Collect tokens from all previous scales
                for prev_idx in range(scale_idx):
                    prev_target = hierarchical_targets[prev_idx]
                    
                    # Extract hard tokens for embedding from soft or hard targets
                    if isinstance(prev_target, dict):
                        # Previous scale used soft labels - extract hard tokens for teacher forcing
                        floor_targets = prev_target['floor_targets']
                        ceil_targets = prev_target['ceil_targets']
                        weights = prev_target['weights']
                        
                        # Sample based on weights: if weight > 0.5, use ceil, otherwise floor
                        prev_scale_tokens = torch.where(weights > 0.5, ceil_targets, floor_targets)
                    else:
                        # Previous scale used hard labels
                        prev_scale_tokens = prev_target
                    
                    cumulative_input_tokens.append(prev_scale_tokens)
                
                # Concatenate all previous scale tokens
                all_prev_tokens = torch.cat(cumulative_input_tokens, dim=1)  # [B, cumulative_length]
                
                # Embed all previous tokens
                input_embed = self.gene_embedding(all_prev_tokens) # [B, cumulative_length, D]
                
                # Prepend start token
                start_token_expanded = self.start_token.expand(B, -1, -1) # [B, 1, D]
                x = torch.cat([start_token_expanded, input_embed], dim=1) # [B, 1 + cumulative_length, D]

            # VAR-style sequence extension for all scales
            # Extend sequence with target positions for current scale
            # NEW: Use intelligent upsampling instead of zero placeholders
            if scale_idx == 0:
                # First scale: still use zeros as we have no previous information
                target_positions = torch.zeros(B, scale_dim, self.embed_dim, device=device)
            elif scale_dim == self.num_genes:
                # Final scale: combine upsampled information with gene identity embeddings
                prev_embeddings = scale_embeddings[-1]  # Get most recent scale embeddings
                upsampled_positions = self.gene_upsampling(
                    prev_embeddings, 
                    source_scale_idx=scale_idx-1, 
                    target_scale_idx=scale_idx
                )
                identity_embeddings = self.gene_identity_embedding.weight.unsqueeze(0).expand(B, -1, -1)
                # Weighted combination: 70% upsampled + 30% identity
                target_positions = 0.7 * upsampled_positions + 0.3 * identity_embeddings
            else:
                # Intermediate scales: use upsampling from previous scale
                prev_embeddings = scale_embeddings[-1]  # Get most recent scale embeddings
                target_positions = self.gene_upsampling(
                    prev_embeddings,
                    source_scale_idx=scale_idx-1,
                    target_scale_idx=scale_idx
                )
            
            x = torch.cat([x, target_positions], dim=1)  # [B, cumulative_length + scale_dim, D]
            
            # Add scale and position embeddings
            current_seq_len = x.shape[1]
            scale_embed = self.scale_embedding(torch.tensor([scale_idx], device=device)).view(1, 1, -1)
            pos_embed = self._get_hierarchical_position_embedding(scale_idx, current_seq_len, device)
            x = x + pos_embed + scale_embed
            
            # Create a causal mask for the current sequence length
            causal_mask = torch.triu(torch.ones(current_seq_len, current_seq_len, device=device) * float('-inf'), diagonal=1)

            # Pass through transformer blocks
            for block in self.transformer_blocks:
                x = block(x, condition_embed, causal_mask)

            # Extract predictions from the appropriate positions
            # For all scales: use the last scale_dim positions (VAR-style autoregressive prediction)
            x_for_prediction = x[:, -scale_dim:, :]  # [B, scale_dim, D]
            
            # Get logits for the current scale's prediction
            x_for_prediction = self.head_norm(x_for_prediction, condition_embed)
            
            # NEW: Apply dynamic gene identity modulation for final scale only
            if scale_dim == self.num_genes:
                # Get gene identity embeddings as modulation conditions
                # Shape: [num_genes, embed_dim] -> [1, num_genes, embed_dim] -> [B, num_genes, embed_dim]
                identity_conditions = self.gene_identity_embedding.weight.unsqueeze(0).expand(B, -1, -1)
                
                # Apply FiLM modulation: each gene dynamically adjusts its contextual response
                x_for_prediction = self.film_layer(x_for_prediction, identity_conditions)
                
                logger.debug(f"🎬 Applied FiLM modulation for {self.num_genes} genes (final scale)")
            
            logits = self.output_head(x_for_prediction) # Shape: [B, scale_dim, vocab_size]

            # The logits now have the correct sequence length to match the target
            logits_for_loss = logits
            
            # Calculate loss for the current scale using soft label loss computation
            loss = self._compute_soft_label_loss(logits_for_loss, scale_target)
            total_loss += loss
            
            # NEW: Store current scale embeddings for next scale's upsampling
            # Use predicted tokens to get embeddings for consistency
            predicted_tokens = torch.argmax(logits_for_loss, dim=-1)  # [B, scale_dim]
            current_scale_embeddings = self.gene_embedding(predicted_tokens)  # [B, scale_dim, embed_dim]
            scale_embeddings.append(current_scale_embeddings)
            
            # Store logits and loss for the final, full-resolution scale
            if scale_dim == self.num_genes:
                # From the final scale, we extract the predictions.
                # The prediction for the Nth gene comes from the Nth token's output logit.
                final_predictions = predicted_tokens.float()  # Use the tokens we just computed
                final_loss = loss
        
        return {
            'loss': total_loss / self.num_scales,
            'loss_final': final_loss,
            'predictions': final_predictions.float(),
            'targets': target_genes.float()
        }

    def forward_inference(
        self,
        condition_embed: torch.Tensor,      # [B, condition_embed_dim]
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical inference pass.
        Autoregressively generates tokens for each scale, where the output of a coarser
        scale becomes the input for the next, finer scale.
        
        Note: top_p sampling is not currently implemented but kept for future compatibility.
        """
        B = condition_embed.shape[0]
        device = condition_embed.device

        # 🔧 NEW: Store all generated tokens from previous scales for cumulative input
        all_generated_scale_tokens = []
        scale_embeddings = []  # NEW: Store embeddings from each scale for upsampling

        for scale_idx, scale_dim in enumerate(self.scale_dims):
            # 🔧 NEW: Build cumulative input from all previously generated scales
            if scale_idx == 0:
                # First scale, just use start token
                x = self.start_token.expand(B, -1, -1) # [B, 1, D]
            else:
                # For subsequent scales, use ALL previously generated tokens (cumulative input)
                # Concatenate all previous scale tokens
                all_prev_tokens = torch.cat(all_generated_scale_tokens, dim=1)  # [B, cumulative_length]
                
                # Embed all previous tokens
                input_embed = self.gene_embedding(all_prev_tokens) # [B, cumulative_length, D]
                
                # Prepend start token
                start_token_expanded = self.start_token.expand(B, -1, -1)
                x = torch.cat([start_token_expanded, input_embed], dim=1) # [B, 1 + cumulative_length, D]
            
            # VAR-style sequence extension for all scales
            # NEW: Use intelligent upsampling (same logic as training)
            if scale_idx == 0:
                # First scale: still use zeros as we have no previous information
                target_positions = torch.zeros(B, scale_dim, self.embed_dim, device=device)
            elif scale_dim == self.num_genes:
                # Final scale: combine upsampled information with gene identity embeddings
                prev_embeddings = scale_embeddings[-1]  # Get most recent scale embeddings
                upsampled_positions = self.gene_upsampling(
                    prev_embeddings, 
                    source_scale_idx=scale_idx-1, 
                    target_scale_idx=scale_idx
                )
                identity_embeddings = self.gene_identity_embedding.weight.unsqueeze(0).expand(B, -1, -1)
                # Weighted combination: 70% upsampled + 30% identity
                target_positions = 0.7 * upsampled_positions + 0.3 * identity_embeddings
            else:
                # Intermediate scales: use upsampling from previous scale
                prev_embeddings = scale_embeddings[-1]  # Get most recent scale embeddings
                target_positions = self.gene_upsampling(
                    prev_embeddings,
                    source_scale_idx=scale_idx-1,
                    target_scale_idx=scale_idx
                )
            
            x = torch.cat([x, target_positions], dim=1)  # [B, cumulative_length + scale_dim, D]
            
            # Add scale and position embeddings
            current_seq_len = x.shape[1]
            scale_embed = self.scale_embedding(torch.tensor([scale_idx], device=device)).view(1, 1, -1)
            pos_embed = self._get_hierarchical_position_embedding(scale_idx, current_seq_len, device)
            x = x + pos_embed + scale_embed

            # Create causal mask
            causal_mask = torch.triu(torch.ones(current_seq_len, current_seq_len, device=device) * float('-inf'), diagonal=1)

            # Pass through transformer blocks
            for block in self.transformer_blocks:
                x = block(x, condition_embed, causal_mask)
            
            # Extract predictions from the appropriate positions (same as training)
            # For all scales: use the last scale_dim positions (VAR-style autoregressive prediction)
            x_for_prediction = x[:, -scale_dim:, :]  # [B, scale_dim, D]
            
            # Get logits for the current scale
            x_for_prediction = self.head_norm(x_for_prediction, condition_embed)
            
            # NEW: Apply dynamic gene identity modulation for final scale only (same as training)
            if scale_dim == self.num_genes:
                # Get gene identity embeddings as modulation conditions
                # Shape: [num_genes, embed_dim] -> [1, num_genes, embed_dim] -> [B, num_genes, embed_dim]
                identity_conditions = self.gene_identity_embedding.weight.unsqueeze(0).expand(B, -1, -1)
                
                # Apply FiLM modulation: each gene dynamically adjusts its contextual response
                x_for_prediction = self.film_layer(x_for_prediction, identity_conditions)
            
            logits = self.output_head(x_for_prediction) # Shape: [B, scale_dim, vocab_size]
            
            # --- START: Top-k Sampling Logic ---
            # Apply temperature scaling before filtering and sampling
            logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None and top_k > 0:
                # Get the values of the top k logits for each position in the sequence
                top_k_values, _ = torch.topk(logits, top_k, dim=-1)
                
                # The k-th value is the last one in the sorted top-k values
                kth_value = top_k_values[:, :, -1].unsqueeze(-1)
                
                # Create a mask for all logits smaller than the k-th value
                mask = logits < kth_value
                
                # Set the logits of tokens below the top-k threshold to -infinity
                logits[mask] = float('-inf')

            # Convert filtered logits to probabilities
            probabilities = F.softmax(logits, dim=-1)

            # Seed for reproducibility if provided
            if seed is not None:
                torch.manual_seed(seed)
            
            # Sample from the filtered distribution using multinomial sampling
            # Reshape for multinomial which expects a 2D tensor
            probabilities_flat = probabilities.view(-1, self.vocab_size)
            sampled_tokens = torch.multinomial(probabilities_flat, num_samples=1)
            
            # Reshape the sampled tokens back to the original sequence shape
            current_scale_dim = logits.shape[1]
            sampled_tokens = sampled_tokens.view(B, current_scale_dim)
            # --- END: Top-k Sampling Logic ---
            
            # 🔧 NEW: Store current scale tokens for future cumulative input
            all_generated_scale_tokens.append(sampled_tokens)
            
            # NEW: Store current scale embeddings for next scale's upsampling
            current_scale_embeddings = self.gene_embedding(sampled_tokens)  # [B, scale_dim, embed_dim]
            scale_embeddings.append(current_scale_embeddings)
            
            # Keep the last generated tokens for final output
            generated_tokens = sampled_tokens

        return {
            'generated_sequence': generated_tokens.float()
        }

    def inference(
        self,
        histology_features: torch.Tensor,
        spatial_coords: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode for gene expression prediction
        
        Args:
            histology_features: Histology features [B, 1024]
            spatial_coords: Spatial coordinates [B, 2]
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            
        Returns:
            Dictionary containing predicted gene expressions
        """
        
        self.eval()
        with torch.no_grad():
            # Process conditions
            condition_embed = self.condition_processor(histology_features, spatial_coords)
            
            # Generate predictions
            return self.forward_inference(
                condition_embed, 
                temperature, 
                top_k, 
                top_p,
                seed
            )
    
    def save_checkpoint(self, save_path: str, epoch: Optional[int] = None):
        """
        Save model checkpoint
        
        Args:
            save_path: Path to save checkpoint
            epoch: Current epoch (optional)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_genes': self.num_genes,
            'scale_dims': self.scale_dims,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'histology_feature_dim': self.histology_feature_dim,
            'spatial_coord_dim': self.spatial_coord_dim,
            'condition_embed_dim': self.condition_embed_dim,
            'adaptive_sigma_alpha': self.adaptive_sigma_alpha,
            'adaptive_sigma_beta': self.adaptive_sigma_beta,
            'epoch': epoch
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"💾 Checkpoint saved to: {save_path}")
    
    @classmethod
    def load_checkpoint(cls, ckpt_path: str, device: str = 'cuda') -> 'MultiScaleGeneVAR':
        """
        Load model from checkpoint
        
        Args:
            ckpt_path: Path to checkpoint
            device: Device to load model on
            
        Returns:
            Loaded MultiScaleGeneVAR model
        """
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Create model with saved configuration
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            num_genes=checkpoint['num_genes'],
            scale_dims=checkpoint['scale_dims'],
            embed_dim=checkpoint['embed_dim'],
            num_heads=checkpoint['num_heads'],
            num_layers=checkpoint['num_layers'],
            histology_feature_dim=checkpoint['histology_feature_dim'],
            spatial_coord_dim=checkpoint['spatial_coord_dim'],
            condition_embed_dim=checkpoint['condition_embed_dim'],
            adaptive_sigma_alpha=checkpoint.get('adaptive_sigma_alpha', 0.1),  # Default for backward compatibility
            adaptive_sigma_beta=checkpoint.get('adaptive_sigma_beta', 1.0),   # Default for backward compatibility
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"📂 Model loaded from: {ckpt_path}")
        if 'epoch' in checkpoint:
            logger.info(f"📊 Loaded model from epoch: {checkpoint['epoch']}")
        
        return model
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        condition_params = sum(p.numel() for p in self.condition_processor.parameters())
        transformer_params = sum(p.numel() for p in self.transformer_blocks.parameters())
        embedding_params = self.gene_embedding.weight.numel() + sum(p.numel() for p in self.hierarchical_pos_embedding.parameters()) + self.scale_embedding.weight.numel()
        output_params = sum(p.numel() for p in self.head_norm.parameters()) + sum(p.numel() for p in self.output_head.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'condition_processor_parameters': condition_params,
            'transformer_parameters': transformer_params,
            'embedding_parameters': embedding_params,
            'output_parameters': output_params,
            'num_genes': self.num_genes,
            'scale_dims': self.scale_dims,
            'num_scales': self.num_scales,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'vocab_size': self.vocab_size,
            'total_sequence_length': self.num_genes + 1
        }

    def enable_kv_cache(self):
        """Enable KV caching for all transformer blocks during inference"""
        for block in self.transformer_blocks:
            block.enable_kv_cache(True)
    
    def disable_kv_cache(self):
        """Disable KV caching for all transformer blocks during training"""
        for block in self.transformer_blocks:
            block.enable_kv_cache(False)

    def _compute_weighted_cross_entropy_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        加权交叉熵损失，考虑token之间的距离关系
        距离越近的错误预测，惩罚越小
        
        Args:
            logits: [total_predictions, vocab_size]
            targets: [total_predictions]
        """
        vocab_size = logits.shape[-1]
        
        # 计算距离权重矩阵
        token_ids = torch.arange(vocab_size, device=logits.device, dtype=torch.float32)  # [vocab_size]
        target_values = targets.float().unsqueeze(1)  # [total_predictions, 1]
        
        # 计算每个预测token与真实token的距离
        distances = torch.abs(token_ids.unsqueeze(0) - target_values)  # [total_predictions, vocab_size]
        
        # 距离加权：使用高斯权重，距离越近权重越大
        sigma = vocab_size * 0.1  # 可调参数
        weights = torch.exp(-distances ** 2 / (2 * sigma ** 2))  # 高斯权重 [total_predictions, vocab_size]
        
        # 计算log概率
        log_probs = F.log_softmax(logits, dim=-1)  # [total_predictions, vocab_size]
        
        # 应用距离权重
        weighted_log_probs = log_probs * weights  # [total_predictions, vocab_size]
        
        # 选择目标位置的加权log概率
        target_log_probs = weighted_log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # [total_predictions]
        
        # 计算加权交叉熵损失
        loss = -target_log_probs.mean()
        
        return loss


# Backward compatibility alias
VARST = MultiScaleGeneVAR