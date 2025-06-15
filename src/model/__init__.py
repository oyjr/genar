from .model_interface import ModelInterface
from .VAR.two_stage_var_st import MultiScaleGeneVAR, VARST

MODELS = {
    'VAR_ST': MultiScaleGeneVAR,  # 使用新的多尺度模型
}

__all__ = ['MultiScaleGeneVAR', 'VARST', 'MODELS']