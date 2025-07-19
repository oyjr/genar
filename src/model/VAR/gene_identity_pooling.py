"""
Gene Identity Pooling Module for Multi-Scale Gene Identity Modulation

This module implements a conservative approach to extend gene identity information
to all scales by pooling from the complete gene identity embeddings.

Author: Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class GeneIdentityPooling(nn.Module):
    """
    Gene Identity Pooling for Multi-Scale Modulation
    
    This module generates scale-appropriate gene identity representations by
    pooling from the complete gene identity embeddings. It's a conservative
    approach that maintains the original architecture while extending gene
    identity modulation to all scales.
    
    Key Features:
    - Preserves original gene identity embeddings
    - Generates pooled representations for intermediate scales
    - Minimal modification to existing architecture
    - Optional activation (can be disabled with a flag)
    """
    
    def __init__(
        self, 
        num_genes: int = 200, 
        scale_dims: Tuple[int, ...] = (1, 4, 8, 40, 100, 200), 
        embed_dim: int = 512,
        enable_pooling: bool = True  # Safety flag to enable/disable
    ):
        super().__init__()
        
        self.num_genes = num_genes
        self.scale_dims = scale_dims
        self.embed_dim = embed_dim
        self.enable_pooling = enable_pooling
        
        if not enable_pooling:
            logger.info("🔒 Gene Identity Pooling is DISABLED - using original behavior")
            return
        
        # Create pooling layers for intermediate scales only
        # Skip the last scale (final gene scale) as it uses original gene identities
        self.scale_poolers = nn.ModuleDict()
        
        for i, dim in enumerate(scale_dims[:-1]):  # Exclude final scale
            if dim == 1:
                # Scale 0: Global genome identity - use simple average
                self.scale_poolers[f'scale_{i}'] = nn.Sequential(
                    nn.AdaptiveAvgPool1d(1),
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.Dropout(0.1)
                )
            else:
                # Intermediate scales: adaptive pooling + projection
                self.scale_poolers[f'scale_{i}'] = nn.Sequential(
                    nn.AdaptiveAvgPool1d(dim),
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.Dropout(0.1)
                )
        
        # Initialize pooling layers with small weights for stability
        self._init_pooling_weights()
        
        logger.info(f"🧬 Gene Identity Pooling initialized:")
        logger.info(f"   - Enable pooling: {enable_pooling}")
        logger.info(f"   - Number of pooling layers: {len(self.scale_poolers)}")
        for i, dim in enumerate(scale_dims[:-1]):
            logger.info(f"   - Scale {i} (dim={dim}): pooling layer created")
    
    def _init_pooling_weights(self):
        """Initialize pooling layers with small weights for stability"""
        for pooler in self.scale_poolers.values():
            for module in pooler:
                if isinstance(module, nn.Linear):
                    # Small initialization to start close to identity
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def get_scale_identity(self, scale_idx: int, gene_identity_embedding: nn.Embedding) -> torch.Tensor:
        """
        Generate gene identity representation for the specified scale
        
        Args:
            scale_idx: Scale index (0 to len(scale_dims)-1)
            gene_identity_embedding: The original gene identity embedding layer
            
        Returns:
            torch.Tensor: Gene identity representation for the scale
                         Shape: [scale_dim, embed_dim]
        """
        scale_dim = self.scale_dims[scale_idx]
        
        # Safety check: if pooling is disabled, only return for final scale
        if not self.enable_pooling:
            if scale_dim == self.num_genes:
                return gene_identity_embedding.weight  # [200, 512]
            else:
                return None  # Signal to skip modulation
        
        # Final scale: use original gene identities
        if scale_dim == self.num_genes:
            return gene_identity_embedding.weight  # [200, 512]
        
        # Intermediate scales: generate pooled representations
        pooler_key = f'scale_{scale_idx}'
        if pooler_key not in self.scale_poolers:
            logger.warning(f"No pooler found for scale {scale_idx}, skipping modulation")
            return None
        
        # Get full gene identities and pool them
        full_identities = gene_identity_embedding.weight  # [200, 512]
        
        # Transpose for adaptive pooling: [512, 200]
        full_identities_t = full_identities.transpose(0, 1).unsqueeze(0)  # [1, 512, 200]
        
        # Apply adaptive pooling first
        pooler = self.scale_poolers[pooler_key]
        adaptive_pool = pooler[0]  # AdaptiveAvgPool1d
        linear_proj = pooler[1]    # Linear layer
        layer_norm = pooler[2]     # LayerNorm
        dropout = pooler[3]        # Dropout
        
        # Pool: [1, 512, 200] -> [1, 512, scale_dim]
        pooled_features = adaptive_pool(full_identities_t)  # [1, 512, scale_dim]
        
        # Transpose for linear layer: [1, 512, scale_dim] -> [1, scale_dim, 512]
        pooled_features = pooled_features.transpose(1, 2)  # [1, scale_dim, 512]
        
        # Apply linear projection: [1, scale_dim, 512] -> [1, scale_dim, 512]
        projected = linear_proj(pooled_features)  # [1, scale_dim, 512]
        
        # Apply layer norm and dropout
        normalized = layer_norm(projected)  # [1, scale_dim, 512]
        final_output = dropout(normalized)  # [1, scale_dim, 512]
        
        # Remove batch dimension: [scale_dim, 512]
        scale_identities = final_output.squeeze(0)  # [scale_dim, 512]
        
        return scale_identities
    
    def get_scale_conditions(
        self, 
        scale_idx: int, 
        batch_size: int, 
        gene_identity_embedding: nn.Embedding,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate scale-appropriate condition embeddings for FiLM modulation
        
        Args:
            scale_idx: Scale index
            batch_size: Batch size
            gene_identity_embedding: Original gene identity embedding
            device: Target device
            
        Returns:
            torch.Tensor: Condition embeddings [B, scale_dim, embed_dim] or None
        """
        scale_identities = self.get_scale_identity(scale_idx, gene_identity_embedding)
        
        if scale_identities is None:
            return None
        
        scale_dim = self.scale_dims[scale_idx]
        
        # Expand to batch dimension
        scale_conditions = scale_identities.unsqueeze(0).expand(batch_size, scale_dim, self.embed_dim)
        scale_conditions = scale_conditions.to(device)
        
        return scale_conditions
    
    def enable(self):
        """Enable gene identity pooling"""
        self.enable_pooling = True
        logger.info("🔓 Gene Identity Pooling ENABLED")
    
    def disable(self):
        """Disable gene identity pooling (fallback to original behavior)"""
        self.enable_pooling = False
        logger.info("🔒 Gene Identity Pooling DISABLED")