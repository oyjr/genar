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
            raise ValueError("Gene identity pooling must be enabled")
        
        # The final scale uses individual gene embeddings directly.
        self.scale_poolers = nn.ModuleDict()
        
        for i, dim in enumerate(scale_dims[:-1]):  # Exclude final scale
            if dim == 1:
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
        
        self._init_pooling_weights()
        
        logger.info(
            "Gene identity pooling initialised with %d intermediate scales",
            len(self.scale_poolers),
        )
    
    def _init_pooling_weights(self):
        """Initialize the intermediate-scale projections."""
        for pooler in self.scale_poolers.values():
            for module in pooler:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
    
    def get_scale_identity(self, scale_idx: int, gene_identity_embedding: nn.Embedding) -> torch.Tensor:
        """Return identity embeddings shaped for one scale."""
        scale_dim = self.scale_dims[scale_idx]
        
        if scale_dim == self.num_genes:
            return gene_identity_embedding.weight
        
        # Intermediate scales: generate pooled representations
        pooler_key = f'scale_{scale_idx}'
        if pooler_key not in self.scale_poolers:
            raise ValueError(f"No pooler found for scale {scale_idx}")
        
        full_identities = gene_identity_embedding.weight
        full_identities_t = full_identities.transpose(0, 1).unsqueeze(0)

        pooler = self.scale_poolers[pooler_key]
        pooled = pooler[0](full_identities_t).transpose(1, 2)
        projected = pooler[1](pooled)
        normalized = pooler[2](projected)
        return pooler[3](normalized).squeeze(0)
    
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
        
        scale_conditions = scale_identities.unsqueeze(0).expand(batch_size, scale_dim, self.embed_dim)
        scale_conditions = scale_conditions.to(device)
        
        return scale_conditions
    
    def enable(self):
        """Reject runtime changes to the model structure."""
        raise RuntimeError("Gene identity pooling cannot be changed at runtime")
    
    def disable(self):
        """Reject runtime changes to the model structure."""
        raise RuntimeError("Gene identity pooling cannot be changed at runtime")
