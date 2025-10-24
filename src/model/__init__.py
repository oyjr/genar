from .genar.multiscale_genar import MultiScaleGenAR, GenARModel
from .foundation_baseline import FoundationOnlyRegressor

MODELS = {
    'GENAR': MultiScaleGenAR,
    'FOUNDATION_BASELINE': FoundationOnlyRegressor,
}

from .model_interface import ModelInterface

__all__ = ['MultiScaleGenAR', 'GenARModel', 'FoundationOnlyRegressor', 'MODELS', 'ModelInterface']
