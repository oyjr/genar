import torch
import torch.nn as nn
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class FiLMLayer(nn.Module):
    """Apply gene-conditioned feature-wise affine modulation."""
    
    def __init__(
        self,
        condition_dim: int,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim or feature_dim // 2
        
        self.condition_projector = nn.Sequential(
            nn.Linear(condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, feature_dim * 2)
        )
        
        self.feature_dim = feature_dim
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the layer as an identity transformation."""
        with torch.no_grad():
            final_layer = self.condition_projector[-1]
            final_layer.weight.zero_()
            final_layer.bias.zero_()
            
            half = self.feature_dim
            final_layer.bias[:half] = 1.0
            final_layer.bias[half:] = 0.0
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Modulate ``x`` with the corresponding gene identities."""
        modulation_params = self.condition_projector(condition)

        gamma, beta = torch.split(modulation_params, self.feature_dim, dim=-1)
        return gamma * x + beta

    def extra_repr(self) -> str:
        return f'condition_dim={self.condition_projector[0].in_features}, ' \
               f'feature_dim={self.feature_dim}, hidden_dim={self.hidden_dim}'
