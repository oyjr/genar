"""No-FiLM ablation for GenAR."""

from typing import Dict, List, Optional, Tuple

from .multiscale_genar import MultiScaleGenAR


class MultiScaleGenARNoFiLM(MultiScaleGenAR):
    """GenAR with gene-identity embeddings and FiLM modulation disabled."""

    DEFAULT_USE_GENE_IDENTITY = False

    def __init__(
        self,
        vocab_size: int,
        num_genes: int = 200,
        scale_dims: Tuple[int, ...] = (1, 4, 8, 40, 100, 200),
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        histology_feature_dim: int = 1024,
        spatial_coord_dim: int = 2,
        condition_embed_dim: int = 768,
        cond_drop_rate: float = 0.1,
        norm_eps: float = 1e-6,
        shared_aln: bool = False,
        attn_l2_norm: bool = True,
        device: str = 'cuda',
        adaptive_sigma_alpha: float = 0.1,
        adaptive_sigma_beta: float = 1.0,
        prediction_mode: str = 'discrete',
        continuous_loss: str = 'mse',
        continuous_loss_alpha: float = 0.01,
        continuous_loss_beta: float = 0.1,
        library_scale: float = 10000.0,
        scale_loss_weights: Optional[List[float]] = None,
        final_loss_mode: str = 'gaussian_kl',
        ablation_protocol: str = 'normalized',
        use_gene_identity: bool = False,
    ):
        if use_gene_identity:
            raise ValueError(
                "MultiScaleGenARNoFiLM cannot enable gene identity modulation"
            )
        if str(prediction_mode).lower() != 'discrete':
            raise ValueError(
                "The no-FiLM ablation supports discrete prediction only"
            )
        super().__init__(
            vocab_size=vocab_size,
            num_genes=num_genes,
            scale_dims=scale_dims,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            histology_feature_dim=histology_feature_dim,
            spatial_coord_dim=spatial_coord_dim,
            condition_embed_dim=condition_embed_dim,
            cond_drop_rate=cond_drop_rate,
            norm_eps=norm_eps,
            shared_aln=shared_aln,
            attn_l2_norm=attn_l2_norm,
            device=device,
            adaptive_sigma_alpha=adaptive_sigma_alpha,
            adaptive_sigma_beta=adaptive_sigma_beta,
            prediction_mode='discrete',
            continuous_loss=continuous_loss,
            continuous_loss_alpha=continuous_loss_alpha,
            continuous_loss_beta=continuous_loss_beta,
            library_scale=library_scale,
            scale_loss_weights=scale_loss_weights,
            final_loss_mode=final_loss_mode,
            ablation_protocol=ablation_protocol,
            use_gene_identity=False,
        )

    def get_model_info(self) -> Dict:
        info = super().get_model_info()
        info['ablation_version'] = 'no_film'
        return info


GenARModelNoFiLM = MultiScaleGenARNoFiLM
