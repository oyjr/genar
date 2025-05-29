from .model_interface import ModelInterface
from .MFBP.MFBP import MFBP
from .VAR.VAR_ST_Complete import VAR_ST_Complete

MODELS = {
    'MFBP': MFBP,
    'VAR_ST': VAR_ST_Complete,
}

__all__ = ['MFBP', 'VAR_ST_Complete', 'MODELS']