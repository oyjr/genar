"""Multi-scale autoregressive model used by GenAR."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

from .gene_genar_transformer import (
    GeneAdaLNSelfAttn,
    GeneAdaLNBeforeHead,
    ConditionProcessor,
)
from .film_layer import FiLMLayer
from .gene_identity_pooling import GeneIdentityPooling
from safe_io import safe_torch_load

# Soft-target type annotations
SoftTarget = Dict[str, torch.Tensor]
HierarchicalTargets = List[Union[torch.Tensor, SoftTarget]]

logger = logging.getLogger(__name__)


class GeneGroupUpsampling(nn.Module):
    """Group-aware upsampling module respecting gene hierarchy."""
    
    def __init__(self, embed_dim: int, scale_dims: Tuple[int, ...], num_genes: int = 200):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale_dims = scale_dims
        self.num_genes = num_genes
        
        # Pre-compute group mappings for each scale transition
        self.group_mappings = self._compute_group_mappings()

        # Learnable upsampling transforms between adjacent scales
        self.upsample_transforms = nn.ModuleDict()
        for i in range(len(scale_dims) - 1):
            self.upsample_transforms[f'scale_{i}_to_{i+1}'] = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Dropout(0.1)
            )
        
        logger.info("Gene Group Upsampling initialised")
        logger.info(f"   Scale dims: {scale_dims}")
        logger.info(f"   Number of upsampling transforms: {len(self.upsample_transforms)}")
        for key in self.group_mappings:
            logger.info(f"   {key}: {len(self.group_mappings[key])} mappings")
    
    def _compute_group_mappings(self):
        """Compute mapping tables between source and target scales."""
        mappings = {}
        
        for i in range(len(self.scale_dims) - 1):
            source_dim = self.scale_dims[i]
            target_dim = self.scale_dims[i + 1]
            
            # Assign every finer position to the proportional coarse group.
            # This covers non-integer transitions in the paper hierarchy such
            # as 40 -> 100 (alternating two/three targets per source).
            mapping = [[] for _ in range(source_dim)]
            for target_idx in range(target_dim):
                source_idx = min(
                    source_dim - 1,
                    (target_idx * source_dim) // target_dim,
                )
                mapping[source_idx].append(target_idx)
            
            mappings[f'scale_{i}_to_{i+1}'] = mapping
            
        return mappings
    
    def forward(self, source_embeddings: torch.Tensor, source_scale_idx: int, target_scale_idx: int):
        """Group-aware upsampling between scales."""
        if source_embeddings is None:
            raise ValueError("source_embeddings must not be None for upsampling")
        
        B, source_dim, embed_dim = source_embeddings.shape
        target_dim = self.scale_dims[target_scale_idx]
        
        # Lookup the precomputed mapping
        mapping_key = f'scale_{source_scale_idx}_to_{target_scale_idx}'
        if mapping_key not in self.group_mappings:
            raise ValueError(f"Missing group mapping for {mapping_key}")
        
        mapping = self.group_mappings[mapping_key]
        
        # Group-aware upsampling
        upsampled = torch.zeros(B, target_dim, embed_dim, device=source_embeddings.device)
        
        for source_idx, target_indices in enumerate(mapping):
            if source_idx < source_dim:
                source_emb = source_embeddings[:, source_idx, :]

                # Optional learned transform
                if mapping_key in self.upsample_transforms:
                    transformed_emb = self.upsample_transforms[mapping_key](source_emb)
                else:
                    transformed_emb = source_emb

                # Copy into target indices
                for target_idx in target_indices:
                    if target_idx < target_dim:
                        upsampled[:, target_idx, :] = transformed_emb

        return upsampled
    
    def _interpolate_upsample(self, source_embeddings, target_dim):
        """Interpolation-based upsampling for non-adjacent scales."""
        _, source_dim, _ = source_embeddings.shape
        
        if source_dim == 1:
            return source_embeddings.expand(-1, target_dim, -1)
        
        # Linear interpolation across the sequence dimension
        source_embeddings_t = source_embeddings.transpose(1, 2)  # [B, embed_dim, source_dim]
        upsampled = F.interpolate(source_embeddings_t, size=target_dim, mode='linear', align_corners=False)
        return upsampled.transpose(1, 2)  # [B, target_dim, embed_dim]


class MultiScaleGenAR(nn.Module):
    """Generate gene-count predictions from coarse groups to individual genes."""

    DEFAULT_USE_GENE_IDENTITY = True
    
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
        adaptive_sigma_beta: float = 1.0,   # Base value for adaptive sigma
        prediction_mode: str = 'discrete',
        continuous_loss: str = 'mse',
        continuous_loss_alpha: float = 0.01,
        continuous_loss_beta: float = 0.1,
        library_scale: float = 10000.0,
        scale_loss_weights: Optional[List[float]] = None,
        final_loss_mode: str = 'gaussian_kl',
        ablation_protocol: str = 'normalized',
        use_gene_identity: bool = True,
    ):
        super().__init__()
        
        # Store key parameters
        self.num_genes = num_genes
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_ratio = float(mlp_ratio)
        self.drop_rate = float(drop_rate)
        self.attn_drop_rate = float(attn_drop_rate)
        self.drop_path_rate = float(drop_path_rate)
        self.cond_drop_rate = cond_drop_rate
        self.norm_eps = float(norm_eps)
        self.shared_aln = bool(shared_aln)
        self.attn_l2_norm = bool(attn_l2_norm)
        self.device = device
        self.adaptive_sigma_alpha = adaptive_sigma_alpha  # Store adaptive sigma parameters
        self.adaptive_sigma_beta = adaptive_sigma_beta
        self.prediction_mode = str(prediction_mode).lower()
        self.continuous_loss = str(continuous_loss).lower()
        self.continuous_loss_alpha = float(continuous_loss_alpha)
        self.continuous_loss_beta = float(continuous_loss_beta)
        self.library_scale = float(library_scale)
        self.final_loss_mode = str(final_loss_mode).lower()
        self.ablation_protocol = str(ablation_protocol).lower()
        self.use_gene_identity = bool(use_gene_identity)

        if self.prediction_mode not in {'discrete', 'continuous'}:
            raise ValueError("prediction_mode must be 'discrete' or 'continuous'")
        if self.continuous_loss not in {'mse', 'gaussian_nll'}:
            raise ValueError("continuous_loss must be 'mse' or 'gaussian_nll'")
        if self.final_loss_mode == 'gaussian_nll':
            warnings.warn(
                "final_loss_mode='gaussian_nll' is a legacy name; the implemented "
                "objective is Gaussian soft-token KL. Use 'gaussian_kl'.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.final_loss_mode = 'gaussian_kl'
        if self.final_loss_mode not in {'gaussian_kl', 'cross_entropy'}:
            raise ValueError(
                "final_loss_mode must be 'gaussian_kl' or 'cross_entropy'"
            )
        if self.ablation_protocol not in {'normalized', 'published'}:
            raise ValueError(
                "ablation_protocol must be 'normalized' or 'published'"
            )
        if self.adaptive_sigma_alpha < 0 or self.adaptive_sigma_beta <= 0:
            raise ValueError("adaptive sigma requires alpha >= 0 and beta > 0")
        if self.library_scale <= 0:
            raise ValueError("library_scale must be positive")
        
        # Hierarchical scale configuration
        self.scale_dims = tuple(int(dim) for dim in scale_dims)
        if not self.scale_dims:
            raise ValueError("scale_dims must contain at least one scale")
        if self.scale_dims[-1] != self.num_genes:
            raise ValueError(
                f"The final scale must equal num_genes ({self.num_genes}), "
                f"got {self.scale_dims[-1]}"
            )
        if any(dim <= 0 for dim in self.scale_dims):
            raise ValueError("All scale dimensions must be positive")
        if any(left >= right for left, right in zip(self.scale_dims, self.scale_dims[1:])):
            raise ValueError("scale_dims must be strictly increasing")
        if any(self.num_genes % dim != 0 for dim in self.scale_dims):
            raise ValueError(
                f"Every scale dimension must divide num_genes={self.num_genes}"
            )
        self.num_scales = len(scale_dims)
        self.scale_loss_weights = self._prepare_scale_loss_weights(scale_loss_weights)
        
        # Store other config parameters for checkpointing
        self.histology_feature_dim = histology_feature_dim
        self.spatial_coord_dim = spatial_coord_dim
        self.condition_embed_dim = condition_embed_dim
        
        # Log the new hierarchical configuration
        logger.info(f"Hierarchical scale dimensions: {self.scale_dims}")
        logger.info(f"Number of scales: {self.num_scales}")
        
        # Condition processor
        self.condition_processor = ConditionProcessor(
            histology_dim=histology_feature_dim,
            spatial_dim=spatial_coord_dim,
            condition_embed_dim=condition_embed_dim,
            dropout=cond_drop_rate,
        )
        
        # Gene token embedding (for expression counts)
        self.gene_embedding = nn.Embedding(vocab_size, embed_dim)
        
        if self.use_gene_identity:
            self.gene_identity_embedding = nn.Embedding(num_genes, embed_dim)
            self.film_layer = FiLMLayer(
                condition_dim=embed_dim,
                feature_dim=embed_dim,
                hidden_dim=embed_dim // 2,
            )
            self.gene_identity_pooling = GeneIdentityPooling(
                num_genes=num_genes,
                scale_dims=self.scale_dims,
                embed_dim=embed_dim,
                enable_pooling=True,
            )
        else:
            self.gene_identity_embedding = None
            self.film_layer = None
            self.gene_identity_pooling = None
        
        # Carries coarse-scale information into the next target positions.
        self.gene_upsampling = GeneGroupUpsampling(
            embed_dim=embed_dim,
            scale_dims=self.scale_dims,
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
                # This follows GenAR's approach: cumulative_context + new_target_positions
                max_cumulative_length = 1 + sum(self.scale_dims[:i]) + self.num_genes
            else:
                # For intermediate scales, use normal cumulative length
                max_cumulative_length = 1 + sum(self.scale_dims[:i+1])
            self.hierarchical_pos_embedding[f'scale_{i}'] = nn.Embedding(max_cumulative_length, embed_dim)
        
        logger.info("Created hierarchical position embeddings for GenAR-style input:")
        for i, dim in enumerate(self.scale_dims):
            if dim == self.num_genes:
                max_length = 1 + sum(self.scale_dims[:i]) + self.num_genes
                logger.info(f"   Scale {i} (dim={dim}): max {max_length} positions (GenAR-style: context + targets)")
            else:
                max_length = 1 + sum(self.scale_dims[:i+1])
                logger.info(f"   Scale {i} (dim={dim}): max {max_length} positions (cumulative)")
        
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
        # The output matrix is shared by the discrete model and the
        # architecture-matched regression control, keeping parameter counts
        # exact. The normalized regression protocol aggregates every channel.
        self.output_head = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
        
        # Log detailed parameter information
        total_params = self._count_parameters()
        logger.info(
            "GenAR initialised: %.1fM parameters, gene identity %s",
            total_params / 1e6,
            "enabled" if self.use_gene_identity else "disabled",
        )
    
    def _count_parameters(self) -> int:
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _get_hierarchical_position_embedding(self, scale_idx: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return semantic position embeddings for a given scale."""
        # Select the scale-specific embedding table
        embedding_layer = self.hierarchical_pos_embedding[f'scale_{scale_idx}']

        # Generate indices [0, 1, ..., seq_len-1]
        pos_indices = torch.arange(seq_len, device=device)

        # Lookup position embeddings
        pos_embed = embedding_layer(pos_indices)

        # Add batch dimension
        return pos_embed.unsqueeze(0)

    def _make_target_positions(
        self,
        scale_idx: int,
        batch_size: int,
        device: torch.device,
        scale_embeddings: List[torch.Tensor],
    ) -> torch.Tensor:
        """Initialize the positions predicted at one scale."""
        scale_dim = self.scale_dims[scale_idx]
        if scale_idx == 0:
            return torch.zeros(
                batch_size,
                scale_dim,
                self.embed_dim,
                device=device,
            )

        positions = self.gene_upsampling(
            scale_embeddings[-1],
            source_scale_idx=scale_idx - 1,
            target_scale_idx=scale_idx,
        )
        if self.use_gene_identity and scale_dim == self.num_genes:
            if self.gene_identity_embedding is None:
                raise RuntimeError("Gene identity embedding is unavailable")
            identities = (
                self.gene_identity_embedding.weight
                .unsqueeze(0)
                .expand(batch_size, -1, -1)
            )
            positions = 0.7 * positions + 0.3 * identities
        return positions

    def _apply_gene_modulation(
        self,
        hidden: torch.Tensor,
        scale_idx: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Apply gene-identity FiLM, or return the no-FiLM ablation unchanged."""
        if not self.use_gene_identity:
            return hidden
        if (
            self.gene_identity_embedding is None
            or self.gene_identity_pooling is None
            or self.film_layer is None
        ):
            raise RuntimeError("Gene identity modules are unavailable")
        conditions = self.gene_identity_pooling.get_scale_conditions(
            scale_idx=scale_idx,
            batch_size=batch_size,
            gene_identity_embedding=self.gene_identity_embedding,
            device=device,
        )
        if conditions is None:
            raise ValueError(
                f"Missing gene identity conditions for scale {scale_idx}"
            )
        return self.film_layer(hidden, conditions)
    
    def init_weights(self, init_std: float = 0.02):
        """Initialize model weights following GenAR initialization"""
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
        logger.info("Model weights initialised with hierarchical position embeddings")

    def _create_hierarchical_targets(self, target_genes: torch.Tensor) -> HierarchicalTargets:
        """Pool targets at each scale, retaining fractional intermediate values."""
        hierarchical_targets = []

        if torch.any(target_genes < 0) or torch.any(target_genes >= self.vocab_size):
            raise ValueError("Target genes are out of vocabulary range")
        target_genes_float = target_genes.float().unsqueeze(1) # -> [B, 1, 200]

        for _, dim in enumerate(self.scale_dims):
            if dim == self.num_genes:
                hard_targets = target_genes.long()
                hard_targets = torch.clamp(hard_targets, 0, self.vocab_size - 1)
                hierarchical_targets.append(hard_targets)
            else:
                pooled_targets = F.adaptive_avg_pool1d(target_genes_float, output_size=dim)
                pooled_targets = pooled_targets.squeeze(1) # -> [B, dim]
                
                if pooled_targets.min() < 0 or pooled_targets.max() > (self.vocab_size - 1):
                    raise ValueError("Pooled targets out of vocabulary range")

                floor_targets = torch.floor(pooled_targets).long()
                ceil_targets = torch.ceil(pooled_targets).long()
                weights = pooled_targets - floor_targets.float()  # Interpolation weights [0.0, 1.0]
                
                soft_target = {
                    'floor_targets': floor_targets,
                    'ceil_targets': ceil_targets,
                    'weights': weights
                }
                
                hierarchical_targets.append(soft_target)
            
        return hierarchical_targets

    def _prepare_scale_loss_weights(
        self,
        weights: Optional[List[float]],
    ) -> List[float]:
        """Validate per-scale supervision weights."""
        if weights is None:
            return [1.0] * self.num_scales

        processed = [float(weight) for weight in weights]
        if len(processed) != self.num_scales:
            raise ValueError(
                "scale_loss_weights must have one value per scale: "
                f"expected {self.num_scales}, got {len(processed)}"
            )
        if any(weight < 0 for weight in processed):
            raise ValueError("scale_loss_weights cannot contain negative values")
        if not any(weight > 0 for weight in processed):
            raise ValueError("At least one scale loss weight must be positive")
        return processed

    def _create_continuous_targets(
        self,
        target_genes: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Create the architecture-matched regression targets at every scale."""
        targets = target_genes.float().unsqueeze(1)
        return [
            F.adaptive_avg_pool1d(targets, output_size=dim).squeeze(1)
            for dim in self.scale_dims
        ]

    def _embed_continuous_sequence(self, values: torch.Tensor) -> torch.Tensor:
        """Interpolate the shared count-token embedding for continuous values."""
        clipped = values.float().clamp(0, self.vocab_size - 1)
        lower = torch.floor(clipped).long()
        upper = torch.ceil(clipped).long()
        weight = (clipped - lower.float()).unsqueeze(-1)
        return (
            self.gene_embedding(lower) * (1.0 - weight)
            + self.gene_embedding(upper) * weight
        )

    def _continuous_head(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return a scalar from the parameter-matched regression head."""
        outputs = self.output_head(hidden)
        if self.ablation_protocol == 'published':
            return outputs[..., 0]
        return F.softplus(outputs).mean(dim=-1)

    def _reduce_ablation_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduce a tokenwise ablation loss under the selected protocol."""
        if self.ablation_protocol == 'published':
            return loss.mean()
        return loss.reshape(loss.shape[0], -1).sum(dim=1).mean()

    def _compute_continuous_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the controlled continuous-regression objective."""
        if self.continuous_loss == 'mse':
            tokenwise = F.mse_loss(predictions, targets, reduction='none')
            return self._reduce_ablation_loss(tokenwise)

        variance = torch.clamp(
            self.continuous_loss_alpha * torch.exp(predictions)
            + self.continuous_loss_beta,
            min=1.0e-4,
        )
        loss = (
            0.5 * torch.log(variance)
            + 0.5 * (targets - predictions).square() / variance
        )
        return self._reduce_ablation_loss(loss)

    def _combine_scale_losses(
        self,
        scale_losses: List[torch.Tensor],
    ) -> torch.Tensor:
        """Return a normalized weighted mean over supervised scales."""
        if len(scale_losses) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} scale losses, got {len(scale_losses)}"
            )
        weight_sum = sum(self.scale_loss_weights)
        return sum(
            weight * loss
            for weight, loss in zip(self.scale_loss_weights, scale_losses)
        ) / weight_sum

    def _compute_final_scale_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the final-scale objective reported by the paper or its CE ablation."""
        if self.final_loss_mode == 'cross_entropy':
            tokenwise = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                target.reshape(-1),
                reduction='none',
            ).reshape(target.shape)
            return self._reduce_ablation_loss(
                tokenwise,
            )
        return self._gaussian_kl_loss(logits, target)

    def _gaussian_kl_loss(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """KL(target Gaussian count tokens || predicted token distribution)."""
        target_dist = self._create_gaussian_target_distribution(
            target,
            logits.device,
        )
        log_probs = F.log_softmax(logits, dim=-1)
        if not torch.isfinite(log_probs).all():
            raise ValueError("Non-finite log probabilities in final-scale loss")
        loss = F.kl_div(
            log_probs,
            target_dist,
            reduction='batchmean',
            log_target=False,
        )
        if not torch.isfinite(loss):
            raise ValueError("Non-finite Gaussian soft-token KL loss")
        return loss

    def _create_gaussian_target_distribution(self, target: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Create the adaptive Gaussian token target used at the final scale."""
        vocab_size = self.vocab_size
        if torch.any(target < 0) or torch.any(target >= vocab_size):
            raise ValueError("Target values are out of vocabulary range")
        
        # Create vocabulary indices tensor [vocab_size]
        vocab_indices = torch.arange(vocab_size, device=device, dtype=torch.float32)
        
        # Expand target to [B, seq_len, 1] for broadcasting
        mu = target.float().unsqueeze(-1)  # [B, seq_len, 1]
        
        # Higher counts receive a wider target distribution.
        sigma = self.adaptive_sigma_alpha * mu + self.adaptive_sigma_beta
        
        if torch.any(sigma <= 0):
            raise ValueError("Adaptive sigma must be positive")
        # Expand vocab_indices to [1, 1, vocab_size] for broadcasting
        x = vocab_indices.view(1, 1, -1)  # [1, 1, vocab_size]
        
        # Compute Gaussian probabilities: exp(-(x - mu)^2 / (2 * sigma^2))
        # Broadcasting: [B, seq_len, 1] and [1, 1, vocab_size] -> [B, seq_len, vocab_size]
        # Note: sigma now has shape [B, seq_len, 1], enabling per-gene adaptive scaling
        squared_diff = (x - mu) ** 2
        gaussian_unnormalized = torch.exp(-squared_diff / (2 * sigma ** 2))
        
        # Normalize to create valid probability distribution
        # Sum over vocab dimension and ensure non-zero normalization
        normalization_factor = gaussian_unnormalized.sum(dim=-1, keepdim=True)
        if torch.any(normalization_factor == 0):
            raise ValueError("Normalization factor is zero in Gaussian target distribution")
        
        target_dist = gaussian_unnormalized / normalization_factor
        
        if torch.isnan(target_dist).any() or torch.isinf(target_dist).any():
            raise ValueError("NaN or Inf detected in adaptive Gaussian target distribution")
        
        return target_dist

    def _compute_soft_label_loss(self, logits: torch.Tensor, target: Union[torch.Tensor, SoftTarget]) -> torch.Tensor:
        """Compute the final Gaussian KL or an intermediate soft-label KL."""
        if isinstance(target, torch.Tensor):
            return self._gaussian_kl_loss(logits, target)
        
        if not isinstance(target, dict) or 'floor_targets' not in target:
            raise ValueError("Soft target must be a dict containing 'floor_targets', 'ceil_targets', 'weights'")
        
        floor_targets = target['floor_targets']  # [B, seq_len]
        ceil_targets = target['ceil_targets']    # [B, seq_len]
        weights = target['weights']              # [B, seq_len], interpolation weights
        
        _, seq_len, _ = logits.shape
        target_seq_len = floor_targets.shape[1]  # Get the actual target sequence length
        
        # Ensure logits and targets have matching sequence dimensions
        if seq_len != target_seq_len:
            raise ValueError(f"Logits seq_len ({seq_len}) doesn't match target seq_len ({target_seq_len})")
        
        # Compute log probabilities for KL divergence
        log_probs = F.log_softmax(logits, dim=-1)  # [B, seq_len, vocab_size]
        
        # Construct target probability distribution
        # P(floor) = 1 - weight, P(ceil) = weight, P(others) = 0
        target_dist = torch.zeros_like(log_probs)  # [B, seq_len, vocab_size]
        
        # ``scatter_add_`` is essential here. For an integer pooled target,
        # floor == ceil and the two contributions must add to one. Assignment
        # would overwrite the floor mass with a zero ceil mass.
        target_dist.scatter_add_(
            dim=-1,
            index=floor_targets.unsqueeze(-1),
            src=(1.0 - weights).unsqueeze(-1),
        )
        target_dist.scatter_add_(
            dim=-1,
            index=ceil_targets.unsqueeze(-1),
            src=weights.unsqueeze(-1),
        )
        
        dist_sum = target_dist.sum(dim=-1, keepdim=True)
        if torch.any(dist_sum == 0):
            raise ValueError("Target distribution sums to zero")
        if not torch.allclose(dist_sum, torch.ones_like(dist_sum)):
            raise ValueError("Target distribution does not sum to 1")
        if torch.isnan(target_dist).any():
            raise ValueError("NaN detected in target distribution")

        if torch.isinf(log_probs).any():
            raise ValueError("Inf detected in log probabilities")
        
        # Compute KL divergence: KL(target_dist || predicted_dist)
        # Note: PyTorch's kl_div expects log_probs as first argument and target as second
        kl_loss = F.kl_div(log_probs, target_dist, reduction='batchmean', log_target=False)
        
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            raise ValueError("Invalid KL loss detected")
        
        return kl_loss

    def forward(
        self,
        histology_features: torch.Tensor,   # [B, 1024]
        spatial_coords: torch.Tensor,       # [B, 2]
        target_genes: Optional[torch.Tensor] = None,  # [B, 200] for training
        top_k: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Run teacher-forced training or autoregressive inference."""
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
        """Dispatch to the discrete paper model or matched regression ablation."""
        if target_genes.ndim != 2 or target_genes.shape[1] != self.num_genes:
            raise ValueError(
                "target_genes must have shape "
                f"[batch, {self.num_genes}], got {tuple(target_genes.shape)}"
            )
        if not torch.isfinite(target_genes).all():
            raise ValueError("target_genes contains NaN or infinite values")
        if torch.any(target_genes < 0):
            raise ValueError("target_genes contains negative values")
        if self.prediction_mode == 'discrete':
            if torch.is_floating_point(target_genes) and not torch.equal(
                target_genes,
                torch.round(target_genes),
            ):
                raise ValueError(
                    "Discrete target_genes must contain integer count tokens"
                )
            if torch.any(target_genes >= self.vocab_size):
                raise ValueError("Target genes are out of vocabulary range")
        if self.prediction_mode == 'continuous':
            return self._forward_training_continuous(
                condition_embed,
                target_genes,
            )
        return self._forward_training_discrete(condition_embed, target_genes)

    def _forward_training_discrete(
        self,
        condition_embed: torch.Tensor,
        target_genes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Train all scales with teacher forcing and interpolated soft targets."""
        B = condition_embed.shape[0]
        device = condition_embed.device
        
        # 1. Create all hierarchical targets with soft labels for intermediate scales
        # This preserves floating-point precision instead of lossy rounding
        hierarchical_targets = self._create_hierarchical_targets(target_genes)
        
        # Initialize storage for scale processing
        scale_embeddings = []  # Store embeddings from each scale for upsampling
        scale_losses: List[torch.Tensor] = []
        final_predictions = None
        final_loss = torch.tensor(0.0, device=device) # Initialize final_loss
        
        # 2. Iteratively train each scale
        for scale_idx, (scale_dim, scale_target) in enumerate(zip(self.scale_dims, hierarchical_targets)):
            # Build cumulative input from all previous scales (not just the last)
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

            target_positions = self._make_target_positions(
                scale_idx,
                B,
                device,
                scale_embeddings,
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
            # For all scales: use the last scale_dim positions (GenAR-style autoregressive prediction)
            x_for_prediction = x[:, -scale_dim:, :]  # [B, scale_dim, D]
            
            # Get logits for the current scale's prediction
            x_for_prediction = self.head_norm(x_for_prediction, condition_embed)
            
            x_for_prediction = self._apply_gene_modulation(
                x_for_prediction,
                scale_idx,
                B,
                device,
            )
            
            logits = self.output_head(x_for_prediction) # Shape: [B, scale_dim, vocab_size]

            # The logits now have the correct sequence length to match the target
            logits_for_loss = logits
            
            # Intermediate scales use interpolated soft-label KL; the final
            # scale uses Gaussian soft-token KL (or hard CE for the ablation).
            if scale_dim == self.num_genes:
                loss = self._compute_final_scale_loss(
                    logits_for_loss,
                    scale_target,
                )
            else:
                loss = self._compute_soft_label_loss(
                    logits_for_loss,
                    scale_target,
                )
            scale_losses.append(loss)
            
            # Predicted-token embeddings initialize the next scale.
            predicted_tokens = torch.argmax(logits_for_loss, dim=-1)  # [B, scale_dim]
            current_scale_embeddings = self.gene_embedding(predicted_tokens)  # [B, scale_dim, embed_dim]
            scale_embeddings.append(current_scale_embeddings)
            
            # Store logits and loss for the final, full-resolution scale
            if scale_dim == self.num_genes:
                # From the final scale, we extract the predictions.
                # The prediction for the Nth gene comes from the Nth token's output logit.
                final_predictions = predicted_tokens.float()  # Use the tokens we just computed
                final_loss = loss
        
        if final_predictions is None:
            raise RuntimeError("The hierarchy did not produce a final-scale prediction")

        return {
            'loss': self._combine_scale_losses(scale_losses),
            'loss_final': final_loss,
            'predictions': final_predictions.float(),
            'targets': target_genes.float()
        }

    def _forward_training_continuous(
        self,
        condition_embed: torch.Tensor,
        target_genes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Architecture-matched continuous multi-scale regression ablation."""
        batch_size = condition_embed.shape[0]
        device = condition_embed.device
        scale_targets = self._create_continuous_targets(target_genes)
        scale_embeddings: List[torch.Tensor] = []
        scale_losses: List[torch.Tensor] = []
        final_predictions: Optional[torch.Tensor] = None
        final_loss = torch.zeros((), device=device)

        for scale_idx, (scale_dim, scale_target) in enumerate(
            zip(self.scale_dims, scale_targets)
        ):
            if scale_idx == 0:
                x = self.start_token.expand(batch_size, -1, -1)
            else:
                previous_values = torch.cat(scale_targets[:scale_idx], dim=1)
                previous_embeddings = self._embed_continuous_sequence(
                    previous_values
                )
                x = torch.cat(
                    [
                        self.start_token.expand(batch_size, -1, -1),
                        previous_embeddings,
                    ],
                    dim=1,
                )

            target_positions = self._make_target_positions(
                scale_idx,
                batch_size,
                device,
                scale_embeddings,
            )

            x = torch.cat([x, target_positions], dim=1)
            sequence_length = x.shape[1]
            scale_embed = self.scale_embedding(
                torch.tensor([scale_idx], device=device)
            ).view(1, 1, -1)
            x = (
                x
                + self._get_hierarchical_position_embedding(
                    scale_idx,
                    sequence_length,
                    device,
                )
                + scale_embed
            )
            causal_mask = torch.triu(
                torch.full(
                    (sequence_length, sequence_length),
                    float('-inf'),
                    device=device,
                ),
                diagonal=1,
            )
            for block in self.transformer_blocks:
                x = block(x, condition_embed, causal_mask)

            hidden = self.head_norm(
                x[:, -scale_dim:, :],
                condition_embed,
            )
            hidden = self._apply_gene_modulation(
                hidden,
                scale_idx,
                batch_size,
                device,
            )
            predictions = self._continuous_head(hidden)
            loss = self._compute_continuous_loss(predictions, scale_target)
            scale_losses.append(loss)
            scale_embeddings.append(
                self._embed_continuous_sequence(predictions.detach())
            )

            if scale_dim == self.num_genes:
                final_predictions = predictions
                final_loss = loss

        if final_predictions is None:
            raise RuntimeError("The hierarchy did not produce final predictions")

        return {
            'loss': self._combine_scale_losses(scale_losses),
            'loss_final': final_loss,
            'predictions': final_predictions,
            'targets': target_genes.float(),
        }

    def forward_inference(
        self,
        condition_embed: torch.Tensor,      # [B, condition_embed_dim]
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Dispatch inference to discrete generation or continuous regression."""
        if self.prediction_mode == 'continuous':
            return self._forward_inference_continuous(condition_embed)
        return self._forward_inference_discrete(
            condition_embed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )

    def _forward_inference_discrete(
        self,
        condition_embed: torch.Tensor,
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
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if top_p is not None:
            raise NotImplementedError("top_p sampling is not implemented")
        if top_k is not None and (top_k < 1 or top_k > self.vocab_size):
            raise ValueError(
                f"top_k must be in [1, {self.vocab_size}], got {top_k}"
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))

        # Store generated tokens from previous scales for cumulative input
        all_generated_scale_tokens = []
        scale_embeddings = []  # Store embeddings from each scale for upsampling

        for scale_idx, scale_dim in enumerate(self.scale_dims):
            # Build cumulative input from all previously generated scales
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
            
            target_positions = self._make_target_positions(
                scale_idx,
                B,
                device,
                scale_embeddings,
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
            # For all scales: use the last scale_dim positions (GenAR-style autoregressive prediction)
            x_for_prediction = x[:, -scale_dim:, :]  # [B, scale_dim, D]
            
            # Get logits for the current scale
            x_for_prediction = self.head_norm(x_for_prediction, condition_embed)
            
            x_for_prediction = self._apply_gene_modulation(
                x_for_prediction,
                scale_idx,
                B,
                device,
            )
            
            logits = self.output_head(x_for_prediction) # Shape: [B, scale_dim, vocab_size]
            
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

            if top_k == 1:
                # Strict top-1 decoding must remain deterministic even when
                # multiple logits tie at initialization.
                sampled_tokens = torch.argmax(logits, dim=-1)
            else:
                probabilities = F.softmax(logits, dim=-1)
                probabilities_flat = probabilities.reshape(
                    -1,
                    self.vocab_size,
                )
                sampled_tokens = torch.multinomial(
                    probabilities_flat,
                    num_samples=1,
                    generator=generator,
                )
                current_scale_dim = logits.shape[1]
                sampled_tokens = sampled_tokens.view(B, current_scale_dim)
            # Store current scale tokens for future cumulative input
            all_generated_scale_tokens.append(sampled_tokens)
            
            current_scale_embeddings = self.gene_embedding(sampled_tokens)  # [B, scale_dim, embed_dim]
            scale_embeddings.append(current_scale_embeddings)
            
            # Keep the last generated tokens for final output
            generated_tokens = sampled_tokens

        return {
            'generated_sequence': generated_tokens.float()
        }

    def _forward_inference_continuous(
        self,
        condition_embed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Generate continuous values progressively without teacher forcing."""
        batch_size = condition_embed.shape[0]
        device = condition_embed.device
        scale_embeddings: List[torch.Tensor] = []
        generated_values: List[torch.Tensor] = []

        for scale_idx, scale_dim in enumerate(self.scale_dims):
            if scale_idx == 0:
                x = self.start_token.expand(batch_size, -1, -1)
            else:
                previous_values = torch.cat(generated_values, dim=1)
                x = torch.cat(
                    [
                        self.start_token.expand(batch_size, -1, -1),
                        self._embed_continuous_sequence(previous_values),
                    ],
                    dim=1,
                )

            target_positions = self._make_target_positions(
                scale_idx,
                batch_size,
                device,
                scale_embeddings,
            )

            x = torch.cat([x, target_positions], dim=1)
            sequence_length = x.shape[1]
            scale_embed = self.scale_embedding(
                torch.tensor([scale_idx], device=device)
            ).view(1, 1, -1)
            x = (
                x
                + self._get_hierarchical_position_embedding(
                    scale_idx,
                    sequence_length,
                    device,
                )
                + scale_embed
            )
            causal_mask = torch.triu(
                torch.full(
                    (sequence_length, sequence_length),
                    float('-inf'),
                    device=device,
                ),
                diagonal=1,
            )
            for block in self.transformer_blocks:
                x = block(x, condition_embed, causal_mask)

            hidden = self.head_norm(
                x[:, -scale_dim:, :],
                condition_embed,
            )
            hidden = self._apply_gene_modulation(
                hidden,
                scale_idx,
                batch_size,
                device,
            )
            predictions = self._continuous_head(hidden)
            generated_values.append(predictions)
            scale_embeddings.append(
                self._embed_continuous_sequence(predictions)
            )

        return {'generated_sequence': generated_values[-1]}

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
        save_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'num_genes': self.num_genes,
            'scale_dims': self.scale_dims,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'mlp_ratio': self.mlp_ratio,
            'drop_rate': self.drop_rate,
            'attn_drop_rate': self.attn_drop_rate,
            'drop_path_rate': self.drop_path_rate,
            'histology_feature_dim': self.histology_feature_dim,
            'spatial_coord_dim': self.spatial_coord_dim,
            'condition_embed_dim': self.condition_embed_dim,
            'cond_drop_rate': self.cond_drop_rate,
            'norm_eps': self.norm_eps,
            'shared_aln': self.shared_aln,
            'attn_l2_norm': self.attn_l2_norm,
            'adaptive_sigma_alpha': self.adaptive_sigma_alpha,
            'adaptive_sigma_beta': self.adaptive_sigma_beta,
            'prediction_mode': self.prediction_mode,
            'continuous_loss': self.continuous_loss,
            'continuous_loss_alpha': self.continuous_loss_alpha,
            'continuous_loss_beta': self.continuous_loss_beta,
            'library_scale': self.library_scale,
            'scale_loss_weights': self.scale_loss_weights,
            'final_loss_mode': self.final_loss_mode,
            'ablation_protocol': self.ablation_protocol,
            'use_gene_identity': self.use_gene_identity,
            'checkpoint_schema_version': 1,
            'epoch': epoch
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to: {save_path}")
    
    @classmethod
    def load_checkpoint(cls, ckpt_path: str, device: str = 'cuda') -> 'MultiScaleGenAR':
        """
        Load model from checkpoint
        
        Args:
            ckpt_path: Path to checkpoint
            device: Device to load model on
            
        Returns:
            Loaded MultiScaleGenAR model
        """
        checkpoint = safe_torch_load(
            ckpt_path,
            map_location='cpu',
        )
        
        # Create model with saved configuration
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            num_genes=checkpoint['num_genes'],
            scale_dims=checkpoint['scale_dims'],
            embed_dim=checkpoint['embed_dim'],
            num_heads=checkpoint['num_heads'],
            num_layers=checkpoint['num_layers'],
            mlp_ratio=checkpoint.get('mlp_ratio', 4.0),
            drop_rate=checkpoint.get('drop_rate', 0.0),
            attn_drop_rate=checkpoint.get('attn_drop_rate', 0.0),
            drop_path_rate=checkpoint.get('drop_path_rate', 0.1),
            histology_feature_dim=checkpoint['histology_feature_dim'],
            spatial_coord_dim=checkpoint['spatial_coord_dim'],
            condition_embed_dim=checkpoint['condition_embed_dim'],
            cond_drop_rate=checkpoint.get('cond_drop_rate', 0.1),
            norm_eps=checkpoint.get('norm_eps', 1.0e-6),
            shared_aln=checkpoint.get('shared_aln', False),
            attn_l2_norm=checkpoint.get('attn_l2_norm', True),
            adaptive_sigma_alpha=checkpoint.get('adaptive_sigma_alpha', 0.1),  # Default for backward compatibility
            adaptive_sigma_beta=checkpoint.get('adaptive_sigma_beta', 1.0),   # Default for backward compatibility
            prediction_mode=checkpoint.get('prediction_mode', 'discrete'),
            continuous_loss=checkpoint.get('continuous_loss', 'mse'),
            continuous_loss_alpha=checkpoint.get('continuous_loss_alpha', 0.01),
            continuous_loss_beta=checkpoint.get('continuous_loss_beta', 0.1),
            library_scale=checkpoint.get('library_scale', 10000.0),
            scale_loss_weights=checkpoint.get('scale_loss_weights'),
            final_loss_mode=checkpoint.get('final_loss_mode', 'gaussian_kl'),
            ablation_protocol=checkpoint.get(
                'ablation_protocol',
                'published',
            ),
            use_gene_identity=checkpoint.get(
                'use_gene_identity',
                cls.DEFAULT_USE_GENE_IDENTITY,
            ),
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from: {ckpt_path}")
        if 'epoch' in checkpoint:
            logger.info(f"Loaded model from epoch: {checkpoint['epoch']}")
        
        return model
    
    def get_model_info(self) -> Dict:
        """Return model dimensions and parameter counts."""
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
            'prediction_mode': self.prediction_mode,
            'final_loss_mode': self.final_loss_mode,
            'ablation_protocol': self.ablation_protocol,
            'use_gene_identity': self.use_gene_identity,
            'scale_loss_weights': list(self.scale_loss_weights),
            'maximum_sequence_length': 1 + sum(self.scale_dims)
        }

    def enable_kv_cache(self):
        """Reject an optimization that is unsafe for full-sequence scale passes."""
        raise RuntimeError(
            "KV caching is not implemented for GenAR's cumulative full-sequence "
            "next-scale passes"
        )
    
    def disable_kv_cache(self):
        """Clear any block-level cache left by legacy callers."""
        for block in self.transformer_blocks:
            block.enable_kv_cache(False)
    
    def enable_multi_scale_gene_modulation(self):
        """Reject runtime changes to the model structure."""
        raise RuntimeError("Gene identity modulation cannot be changed at runtime")

    def disable_multi_scale_gene_modulation(self):
        """Reject runtime changes to the model structure."""
        raise RuntimeError("Gene identity modulation cannot be changed at runtime")

# Backward compatibility alias
GenARModel = MultiScaleGenAR
