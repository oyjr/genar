from .model_interface import ModelInterface
from .MFBP.MFBP import MFBP
from .VAR.var_gene_wrapper import VARGeneWrapper
from .VAR.two_stage_var_st import TwoStageVARST

MODELS = {
    'MFBP': MFBP,
    'VAR_ST': VARGeneWrapper,
    'TWO_STAGE_VAR_ST': TwoStageVARST,
}

__all__ = ['MFBP', 'VARGeneWrapper', 'TwoStageVARST', 'MODELS']