"""Foundation-only baseline model.

This module defines a light-weight predictor that relies solely on
precomputed histology embeddings (UNI / Conch / ResNet18).  It serves as a
sanity-check baseline to demonstrate how much performance is contributed by
the downstream architecture beyond the foundation encoder itself.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FoundationOnlyRegressor(nn.Module):
    """Predict gene tokens directly from fixed histology features.

    The model is intentionally simple: a small MLP maps a single-spot
    embedding to ``num_genes`` categorical distributions over gene token
    vocabularies.  It exposes the same interface as the main VAR-ST model so
    that it can be dropped into the existing Lightning training pipeline as a
    baseline.
    """

    def __init__(
        self,
        histology_feature_dim: int,
        num_genes: int = 200,
        vocab_size: int = 501,
        hidden_dim: int = 512,
        num_hidden_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        self.histology_feature_dim = histology_feature_dim
        self.num_genes = num_genes
        self.vocab_size = vocab_size

        self.input_proj = nn.Linear(histology_feature_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, num_genes * vocab_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        histology_features: torch.Tensor,
        target_genes: Optional[torch.Tensor] = None,
        top_k: int = 1,
        spatial_coords: Optional[torch.Tensor] = None,
        **unused_kwargs,
    ) -> dict:
        """Run the baseline predictor.

        Args:
            histology_features: Tensor of shape ``[batch, feature_dim]`` (or
                ``[batch, seq, feature_dim]`` which will be flattened).
            target_genes: Optional tensor of integer tokens ``[batch, num_genes]``.
            top_k: Unused argument kept for API compatibility with VAR-ST.

        Returns:
            Dictionary containing logits, predictions and optional losses.
        """

        # Ensure tensor is 2D (batch, feature_dim)
        if histology_features.dim() == 1:
            histology_features = histology_features.unsqueeze(0)
        if histology_features.dim() == 3:
            batch, spots, feat = histology_features.shape
            histology_features = histology_features.reshape(batch * spots, feat)

        x = histology_features.float()
        x = self.input_proj(x)
        x = self.activation(x)
        x = self.dropout(x)

        for layer in self.hidden_layers:
            residual = x
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = x + residual  # lightweight residual to ease optimisation

        logits = self.head(x)
        logits = logits.view(-1, self.num_genes, self.vocab_size)
        predictions = logits.argmax(dim=-1).float()

        outputs = {
            'logits': logits,
            'predictions': predictions,
        }

        if target_genes is not None:
            target = target_genes.view(-1).long()
            logits_flat = logits.view(-1, self.vocab_size)
            loss = F.cross_entropy(logits_flat, target)
            outputs['loss'] = loss
            outputs['loss_final'] = loss
            outputs['full_target'] = target_genes

        return outputs
