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

logger = logging.getLogger(__name__)


class MultiScaleGeneVAR(nn.Module):
    """
    Hierarchical Gene VAR for Spatial Transcriptomics
    
    This model implements a hierarchical generation process, moving from a coarse,
    global representation of gene expression to a fine-grained, per-gene prediction.
    
    Architecture:
    - Condition Processor: Encodes histology and spatial features.
    - Hierarchical Generation: Sequentially refines predictions across scales (e.g., 1 -> 4 -> 16 -> 64 -> 200).
    - AdaLN Transformer: Core computation block with deep conditioning.
    - Teacher Forcing: Uses ground truth averages at each scale to guide the next.
    """
    
    def __init__(
        self,
        # Gene-related parameters
        vocab_size: int,
        num_genes: int = 200,
        scale_dims: Tuple[int, ...] = (1, 4, 16, 64, 200),
        
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
        device: str = 'cuda'
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
        
        # Gene token embedding
        self.gene_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Unified position embedding for the maximum sequence length (200 genes + 1 start token)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_genes + 1, embed_dim) * 0.02)
        
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
        
        logger.info(f"✅ Hierarchical Gene VAR initialized successfully")
        logger.info(f"📈 Model parameters: ~{self._count_parameters()/1e6:.1f}M")
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
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
        logger.info("🎯 Model weights initialized")

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
        B = target_genes.shape[0]
        hierarchical_targets = []

        # Ensure target_genes is float for pooling operations
        target_genes_float = target_genes.float().unsqueeze(1) # -> [B, 1, 200]

        for i, dim in enumerate(self.scale_dims):
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
        # Handle hard labels (final scale)
        if isinstance(target, torch.Tensor):
            return F.cross_entropy(logits.reshape(-1, self.vocab_size), target.reshape(-1))
        
        # Handle soft labels (intermediate scales)
        if not isinstance(target, dict) or 'floor_targets' not in target:
            raise ValueError("Soft target must be a dict containing 'floor_targets', 'ceil_targets', 'weights'")
        
        floor_targets = target['floor_targets']  # [B, seq_len]
        ceil_targets = target['ceil_targets']    # [B, seq_len]
        weights = target['weights']              # [B, seq_len], interpolation weights
        
        B, seq_len, vocab_size = logits.shape
        
        # Compute log probabilities for KL divergence
        log_probs = F.log_softmax(logits, dim=-1)  # [B, seq_len, vocab_size]
        
        # Construct target probability distribution
        # P(floor) = 1 - weight, P(ceil) = weight, P(others) = 0
        target_dist = torch.zeros_like(log_probs)  # [B, seq_len, vocab_size]
        
        # Create indices for scatter operations
        batch_indices = torch.arange(B, device=logits.device).unsqueeze(1).expand(-1, seq_len)
        seq_indices = torch.arange(seq_len, device=logits.device).unsqueeze(0).expand(B, -1)
        
        # Set probabilities for floor targets: P(floor) = 1 - weight
        floor_probs = 1.0 - weights  # [B, seq_len]
        target_dist[batch_indices, seq_indices, floor_targets] = floor_probs
        
        # Set probabilities for ceil targets: P(ceil) = weight
        # Only when ceil != floor (avoid double-counting when pooled value is integer)
        ceil_mask = (ceil_targets != floor_targets)
        if ceil_mask.any():
            ceil_probs = weights * ceil_mask.float()  # [B, seq_len]
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
        
        # Initialize lists to store outputs from each scale
        all_logits = []
        total_loss = 0.0
        final_predictions = None
        final_loss = torch.tensor(0.0, device=device) # Initialize final_loss
        
        # 2. Iteratively train each scale
        for scale_idx, (scale_dim, scale_target) in enumerate(zip(self.scale_dims, hierarchical_targets)):
            # Determine the input for the current scale
            if scale_idx == 0:
                # For the first scale, input is just the start token
                scale_input_tokens = None # Will be handled by start_token inside the loop
            else:
                # For subsequent scales, input is the ground truth from the previous scale (Teacher Forcing)
                prev_target = hierarchical_targets[scale_idx - 1]
                
                # Extract hard tokens for embedding from soft or hard targets
                if isinstance(prev_target, dict):
                    # Previous scale used soft labels - extract hard tokens for teacher forcing
                    # Use weighted sampling between floor and ceil based on weights
                    floor_targets = prev_target['floor_targets']
                    ceil_targets = prev_target['ceil_targets']
                    weights = prev_target['weights']
                    
                    # Sample based on weights: if weight > 0.5, use ceil, otherwise floor
                    # This maintains the probabilistic nature while providing discrete tokens
                    scale_input_tokens = torch.where(weights > 0.5, ceil_targets, floor_targets)
                else:
                    # Previous scale used hard labels
                    scale_input_tokens = prev_target
            
            # Embed the input tokens
            if scale_input_tokens is not None:
                # Input from previous scale: [B, prev_dim]
                input_embed = self.gene_embedding(scale_input_tokens) # [B, prev_dim, D]
                # Prepend start token
                start_token_expanded = self.start_token.expand(B, -1, -1) # [B, 1, D]
                x = torch.cat([start_token_expanded, input_embed], dim=1) # [B, 1 + prev_dim, D]
            else:
                # First scale, just use start token
                x = self.start_token.expand(B, -1, -1) # [B, 1, D]

            # Add scale and position embeddings
            current_seq_len = x.shape[1]
            scale_embed = self.scale_embedding(torch.tensor([scale_idx], device=device)).view(1, 1, -1)
            pos_embed = self.pos_embedding[:, :current_seq_len, :]
            x = x + pos_embed + scale_embed
            
            # Create a causal mask for the current sequence length
            causal_mask = torch.triu(torch.ones(current_seq_len, current_seq_len, device=device) * float('-inf'), diagonal=1)

            # Pass through transformer blocks
            for block in self.transformer_blocks:
                x = block(x, condition_embed, causal_mask)

            # --- FIX: Upsample hidden states to match target dimension ---
            # Permute from [B, L, D] to [B, D, L] for interpolation
            x = x.permute(0, 2, 1)
            # Linearly interpolate the sequence length to the current scale's target dimension
            x = F.interpolate(x, size=scale_dim, mode='linear', align_corners=False)
            # Permute back to [B, L, D]
            x = x.permute(0, 2, 1)
            # --- END FIX ---

            # Get logits for the current scale's prediction
            x = self.head_norm(x, condition_embed)
            logits = self.output_head(x) # Shape: [B, scale_dim, vocab_size]

            # The logits now have the correct sequence length to match the target
            logits_for_loss = logits
            
            # Calculate loss for the current scale using soft label loss computation
            loss = self._compute_soft_label_loss(logits_for_loss, scale_target)
            total_loss += loss
            
            # Store logits and loss for the final, full-resolution scale
            if scale_dim == self.num_genes:
                # From the final scale, we extract the predictions.
                # The prediction for the Nth gene comes from the Nth token's output logit.
                final_predictions = torch.argmax(logits_for_loss, dim=-1) # [B, 200]
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
        """
        B = condition_embed.shape[0]
        device = condition_embed.device

        generated_tokens = None # This will hold the output of the previous scale

        for scale_idx, scale_dim in enumerate(self.scale_dims):
            # Embed the input tokens for the current scale
            if generated_tokens is not None:
                # Input from previous scale's generation
                input_embed = self.gene_embedding(generated_tokens) # [B, prev_dim, D]
                start_token_expanded = self.start_token.expand(B, -1, -1)
                x = torch.cat([start_token_expanded, input_embed], dim=1) # [B, 1 + prev_dim, D]
            else:
                # First scale, just use start token
                x = self.start_token.expand(B, -1, -1) # [B, 1, D]
            
            # Add scale and position embeddings
            current_seq_len = x.shape[1]
            scale_embed = self.scale_embedding(torch.tensor([scale_idx], device=device)).view(1, 1, -1)
            pos_embed = self.pos_embedding[:, :current_seq_len, :]
            x = x + pos_embed + scale_embed

            # Create causal mask
            causal_mask = torch.triu(torch.ones(current_seq_len, current_seq_len, device=device) * float('-inf'), diagonal=1)

            # Pass through transformer blocks
            for block in self.transformer_blocks:
                x = block(x, condition_embed, causal_mask)
            
            # --- FIX: Upsample hidden states to match target dimension ---
            x = x.permute(0, 2, 1)
            x = F.interpolate(x, size=scale_dim, mode='linear', align_corners=False)
            x = x.permute(0, 2, 1)
            # --- END FIX ---
            
            # Get logits for the current scale
            x = self.head_norm(x, condition_embed)
            logits = self.output_head(x) # Shape: [B, scale_dim, vocab_size]
            
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
            
            # The output of this scale becomes the input for the next
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
        embedding_params = self.gene_embedding.weight.numel() + self.pos_embedding.numel() + self.scale_embedding.weight.numel()
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