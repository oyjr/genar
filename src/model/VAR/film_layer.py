import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer for Gene Identity Modulation.
    
    This layer takes contextual hidden states and gene identity embeddings,
    then generates gene-specific scaling (gamma) and shifting (beta) parameters
    to dynamically modulate the hidden states.
    
    Biological Intuition:
    - gamma: Controls how much each gene amplifies/suppresses contextual signals
    - beta: Adds gene-specific baseline expression patterns
    """
    
    def __init__(self, condition_dim: int, feature_dim: int, hidden_dim: int = None):
        super().__init__()
        
        # Use a small hidden layer for more expressive transformation
        self.hidden_dim = hidden_dim or feature_dim // 2
        
        # MLP to generate gamma and beta from gene identity
        self.condition_projector = nn.Sequential(
            nn.Linear(condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, feature_dim * 2)  # Output: gamma + beta
        )
        
        self.feature_dim = feature_dim
        
        # Initialize to near-identity transformation initially
        self._init_weights()
        
    def _init_weights(self):
        """Initialize to produce gamma≈1, beta≈0 initially for stable training"""
        with torch.no_grad():
            # Initialize the last layer to output [1, 0, 1, 0, ...] pattern
            final_layer = self.condition_projector[-1]
            final_layer.weight.zero_()
            final_layer.bias.zero_()
            
            # Set bias to produce gamma=1, beta=0
            # The output is [gamma_0, gamma_1, ..., gamma_n, beta_0, beta_1, ..., beta_n]
            half = self.feature_dim
            final_layer.bias[:half] = 1.0  # gamma components = 1
            final_layer.bias[half:] = 0.0  # beta components = 0
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Dynamically modulate features using gene identity.
        
        Args:
            x (torch.Tensor): Contextual hidden states [B, N, feature_dim]
            condition (torch.Tensor): Gene identity embeddings [B, N, condition_dim]
        
        Returns:
            torch.Tensor: Modulated features [B, N, feature_dim]
        """
        # Generate modulation parameters from gene identity
        modulation_params = self.condition_projector(condition)  # [B, N, feature_dim * 2]
        
        # Split into scaling (gamma) and shifting (beta) parameters
        gamma, beta = torch.split(modulation_params, self.feature_dim, dim=-1)
        
        # Apply modulation: gamma * x + beta
        # This allows each gene to selectively amplify/suppress contextual signals
        modulated = gamma * x + beta
        
        return modulated
    
    def extra_repr(self) -> str:
        return f'condition_dim={self.condition_projector[0].in_features}, ' \
               f'feature_dim={self.feature_dim}, hidden_dim={self.hidden_dim}' 