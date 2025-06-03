from .model_interface import ModelInterface
from .MFBP.MFBP import MFBP
from .VAR.var_gene_wrapper import VARGeneWrapper

MODELS = {
    'MFBP': MFBP,
    'VAR_ST': VARGeneWrapper,
}

__all__ = ['MFBP', 'VARGeneWrapper', 'MODELS']