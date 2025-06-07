from .model_interface import ModelInterface
from .MFBP.MFBP import MFBP
from .VAR.two_stage_var_st import TwoStageVARST

MODELS = {
    'MFBP': MFBP,
    'TWO_STAGE_VAR_ST': TwoStageVARST,
}

__all__ = ['MFBP', 'TwoStageVARST', 'MODELS']