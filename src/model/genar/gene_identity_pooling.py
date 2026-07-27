"""Pool gene-identity embeddings to each GenAR scale."""

import torch
import torch.nn as nn
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class GeneIdentityPooling(nn.Module):
    """Build scale-sized FiLM conditions from per-gene identities."""
    
    def __init__(
        self, 
        num_genes: int = 200, 
        scale_dims: Tuple[int, ...] = (1, 4, 8, 40, 100, 200), 
        embed_dim: int = 512,
        enable_pooling: bool = True,
    ):
        super().__init__()
        
        self.num_genes = num_genes
        self.scale_dims = scale_dims
        self.embed_dim = embed_dim
        self.enable_pooling = enable_pooling
        
        if not enable_pooling:
            raise ValueError("Gene Identity Pooling cannot be disabled in strict mode")
        
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
        
        logger.info(
            "Gene identity pooling initialised with %d intermediate scales",
            len(self.scale_poolers),
        )
    
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
        """Return identity embeddings shaped for one scale."""
        scale_dim = self.scale_dims[scale_idx]
        
        # Final scale: use original gene identities
        if scale_dim == self.num_genes:
            return gene_identity_embedding.weight  # [200, 512]
        
        # Intermediate scales: generate pooled representations
        pooler_key = f'scale_{scale_idx}'
        if pooler_key not in self.scale_poolers:
            raise ValueError(f"No pooler found for scale {scale_idx}")
        
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
        """Return FiLM conditions shaped ``[batch, scale, embedding]``."""
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
        raise RuntimeError("Gene Identity Pooling cannot be enabled/disabled at runtime in strict mode")
    
    def disable(self):
        """Disable gene identity pooling (unsupported in strict mode)"""
        raise RuntimeError("Gene Identity Pooling cannot be enabled/disabled at runtime in strict mode")
