from .genar.multiscale_genar import MultiScaleGenAR, GenARModel
from .foundation_baseline import FoundationOnlyRegressor

MODELS = {
    'GENAR': MultiScaleGenAR,
    'FOUNDATION_BASELINE': FoundationOnlyRegressor,
}

# Delayed until after MODELS is populated because model_interface imports the
# registry through model_utils.
from .model_interface import ModelInterface  # noqa: E402

__all__ = [
    'MODELS',
    'FoundationOnlyRegressor',
    'GenARModel',
    'ModelInterface',
    'MultiScaleGenAR',
]
