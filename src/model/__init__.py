from .VAR.two_stage_var_st import MultiScaleGeneVAR, VARST
from .foundation_baseline import FoundationOnlyRegressor

MODELS = {
    'VAR_ST': MultiScaleGeneVAR,  # 使用新的多尺度模型
    'FOUNDATION_BASELINE': FoundationOnlyRegressor,
}

from .model_interface import ModelInterface

__all__ = ['MultiScaleGeneVAR', 'VARST', 'FoundationOnlyRegressor', 'MODELS', 'ModelInterface']
