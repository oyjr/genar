"""
Multi-Scale Gene VAR for Spatial Transcriptomics

This package implements a multi-scale VAR model for spatial transcriptomics
based on the original VAR architecture.

Main Components:
- MultiScaleGeneVAR: Main model class
- GeneAdaLNSelfAttn: AdaLN self-attention block
- GeneAdaLNBeforeHead: AdaLN before output head
- ConditionProcessor: Enhanced condition processing
- DropPath: Stochastic depth utility

Author: Assistant
Date: 2024
"""

from .two_stage_var_st import MultiScaleGeneVAR, VARST
from .gene_var_transformer import (
    GeneAdaLNSelfAttn,
    GeneAdaLNBeforeHead, 
    ConditionProcessor,
    DropPath,
    SelfAttention,
    FFN
)

__all__ = [
    'MultiScaleGeneVAR',
    'VARST',  # Backward compatibility alias
    'GeneAdaLNSelfAttn',
    'GeneAdaLNBeforeHead',
    'ConditionProcessor',
    'DropPath',
    'SelfAttention',
    'FFN'
] 