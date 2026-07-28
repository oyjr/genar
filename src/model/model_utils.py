"""Utility helpers for model configuration, loading, and preprocessing."""

import inspect
import importlib
import logging
from typing import Dict

import torch

from addict import Dict as AddictDict

from . import MODELS

# Default constants
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_GRADIENT_CLIP = 1.0


class ModelUtils:
    """Utility collection used by ModelInterface."""
    
    def __init__(self, config, lightning_module):
        self.config = config
        self.lightning_module = lightning_module
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_config(self, path: str, default=None):
        """Safely extract nested configuration values."""
        parts = path.split('.')
        value = self.config
        
        try:
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, default)
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return default
            return value
        except Exception:
            return default

    def load_model(self):
        """Load the configured model implementation."""
        model_name = self.get_config('MODEL.model_name', 'GENAR')

        # Special-case GenAR variants while preserving dynamic import
        if model_name == 'GENAR':
            try:
                model_variant = self.get_config('MODEL.model_variant', 'original')

                if model_variant == 'no_film':
                    self._logger.info("Loading GenAR (NoFiLM ablation) ...")
                    Model = importlib.import_module(
                        'model.genar.multiscale_genar_no_film'
                    ).MultiScaleGenARNoFiLM
                    self._logger.info("GenAR (NoFiLM) loaded")
                else:
                    self._logger.info("Loading GenAR (original variant) ...")
                    Model = importlib.import_module(
                        'model.genar.multiscale_genar'
                    ).MultiScaleGenAR
                    self._logger.info("GenAR (original) loaded")

                return self.instancialize(Model)

            except Exception as e:
                self._logger.error(f"Failed to load GenAR model: {e!s}")
                raise ValueError(
                    f"GenAR model load failed: {e!s}"
                ) from e

        # Remaining models come from the registry
        if model_name not in MODELS:
            raise ValueError(f"Model '{model_name}' is not registered")

        self._logger.info(f"Loading model: {model_name}")
        ModelClass = MODELS[model_name]
        return self.instancialize(ModelClass)

    def instancialize(self, Model, **other_args):
        """Instantiate the model with config-driven arguments."""
        try:
            # Inspect constructor arguments
            class_args = inspect.getfullargspec(Model.__init__).args[1:]
            
            # Normalise config to a dict
            model_config = self.config.MODEL
            if isinstance(model_config, AddictDict):
                model_config_dict = dict(model_config)
            elif hasattr(model_config, '__dict__'):
                model_config_dict = vars(model_config)
            else:
                model_config_dict = model_config
            
            args = {}
            
            # Populate constructor kwargs
            for arg in class_args:
                if arg in model_config_dict:
                    args[arg] = model_config_dict[arg]
                elif arg == 'config':
                    args[arg] = self.config
                elif arg == 'histology_feature_dim' and 'feature_dim' in model_config_dict:
                    args[arg] = model_config_dict['feature_dim']

            # Checkpoints created before the normalized ablation protocol did
            # not record this field. Preserve their exact behavior.
            if (
                'ablation_protocol' in class_args
                and 'ablation_protocol' not in model_config_dict
            ):
                args['ablation_protocol'] = 'published'
                    
            # Merge explicit overrides
            args.update(other_args)
            
            return Model(**args)
            
        except Exception as e:
            self._logger.error(f"Model instantiation failed: {e!s}")
            raise

    def preprocess_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare batch inputs for the underlying model."""
        # Validate incoming data
        self.validate_inputs(inputs)
        
        # Copy relevant tensors
        processed_inputs = {}
        
        # Histology features
        if 'img' in inputs:
            processed_inputs['histology_features'] = inputs['img']
        # Spatial coordinates
        if 'positions' in inputs:
            processed_inputs['spatial_coords'] = inputs['positions']
        # Gene expression (loss computation happens later)
        if 'target_genes' in inputs:
            processed_inputs['target_genes'] = inputs['target_genes']
        
        # Move tensors to the current device
        for key, value in processed_inputs.items():
            if torch.is_tensor(value):
                processed_inputs[key] = value.to(self.lightning_module.device)
        
        return processed_inputs

    def validate_inputs(self, inputs: Dict[str, torch.Tensor]):
        """Basic validation for required keys and shapes."""
        required_keys = ['img', 'positions']
        
        # Check mandatory keys
        for key in required_keys:
            if key not in inputs:
                raise ValueError(f"Missing required input key: {key}")
                
        # Expected tensor dimensions per field
        expected_dims = {
            'img': [2],
            'target_genes': [2],
            'raw_target_genes': [2],
            'positions': [2],
            'spot_idx': [1],
            'gene_ids': [1, 2],
        }
        
        # Shape checks
        for key, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                # Use defaults when not explicitly defined
                allowed_dims = expected_dims.get(key, [1, 2, 3])
                
                if tensor.dim() not in allowed_dims:
                    raise ValueError(f"Unexpected tensor rank for {key}: {tensor.shape}; allowed={allowed_dims}")
                if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
                    raise ValueError(f"Input tensor {key} contains NaN or Inf")

        batch_size = inputs['img'].shape[0]
        if inputs['positions'].shape[0] != batch_size:
            raise ValueError("Image and position batch sizes do not match")
        if not torch.is_floating_point(inputs['img']):
            raise ValueError("Histology embeddings must be floating point")
        if not torch.is_floating_point(inputs['positions']):
            raise ValueError("Spatial coordinates must be floating point")
        expected_feature_dim = int(
            self.get_config(
                'MODEL.histology_feature_dim',
                self.get_config('MODEL.feature_dim'),
            )
        )
        if inputs['img'].shape[1] != expected_feature_dim:
            raise ValueError(
                f"Expected histology dimension {expected_feature_dim}, got "
                f"{inputs['img'].shape[1]}"
            )
        if inputs['positions'].shape[1] != 2:
            raise ValueError("Spatial coordinates must have shape [batch, 2]")
        
        # Numeric sanity checks
        if 'target_genes' in inputs:
            targets = inputs['target_genes']
            expected_genes = int(self.get_config('MODEL.num_genes'))
            if targets.shape != (batch_size, expected_genes):
                raise ValueError(
                    f"Expected targets shaped [{batch_size}, {expected_genes}], "
                    f"got {tuple(targets.shape)}"
                )
            if (targets < 0).any():
                raise ValueError("Target gene expression contains negative values")
            prediction_mode = str(
                self.get_config('MODEL.prediction_mode', 'discrete')
            )
            if prediction_mode == 'discrete':
                max_gene_count = int(
                    self.get_config(
                        'MODEL.max_gene_count',
                        self.get_config('max_gene_count', 0),
                    )
                )
                if torch.any(targets > max_gene_count):
                    raise ValueError(
                        "Discrete targets exceed the configured count cap"
                    )


    def scale_learning_rate(self, base_lr: float) -> float:
        """Scale learning rate according to batch size and device count."""
        # Base batch size
        batch_size = self.get_config('DATA.train_dataloader.batch_size')
        if batch_size is None:
            raise ValueError("Missing DATA.train_dataloader.batch_size in config")

        if not hasattr(self.lightning_module, 'trainer') or self.lightning_module.trainer is None:
            raise RuntimeError("Trainer is not attached; cannot scale learning rate")

        if not hasattr(self.lightning_module.trainer, 'world_size'):
            raise RuntimeError("Trainer is missing world_size; cannot scale learning rate")

        num_devices = self.lightning_module.trainer.world_size

        # Effective batch size
        effective_batch_size = batch_size * num_devices

        # Linear scaling rule: lr = base_lr * (effective / base)
        base_batch_size = 32
        scaled_lr = base_lr * (effective_batch_size / base_batch_size)

        self._logger.info("Learning rate scaled: %.6f -> %.6f (batch_size=%s, num_devices=%s)",
                          base_lr, scaled_lr, batch_size, num_devices)

        return scaled_lr

    def get_scheduler_config(self, optimizer):
        """Return the LR scheduler configuration dictionary."""
        scheduler_config = self.get_config('TRAINING.lr_scheduler')
        if scheduler_config is None:
            raise ValueError("Missing TRAINING.lr_scheduler in config")

        if isinstance(scheduler_config, dict):
            scheduler_name = scheduler_config.get('name')
            patience = scheduler_config.get('patience')
            factor = scheduler_config.get('factor')
            monitor = scheduler_config.get('monitor')
            mode = scheduler_config.get('mode', 'min')
            interval = scheduler_config.get('interval', 'epoch')
            frequency = scheduler_config.get('frequency', 1)
            t_max = scheduler_config.get('T_max')
            eta_min = scheduler_config.get('eta_min')
            step_size = scheduler_config.get('step_size')
            gamma = scheduler_config.get('gamma')
        else:
            scheduler_name = getattr(scheduler_config, 'name', None)
            patience = getattr(scheduler_config, 'patience', None)
            factor = getattr(scheduler_config, 'factor', None)
            monitor = getattr(scheduler_config, 'monitor', None)
            mode = getattr(scheduler_config, 'mode', 'min')
            interval = getattr(scheduler_config, 'interval', 'epoch')
            frequency = getattr(scheduler_config, 'frequency', 1)
            t_max = getattr(scheduler_config, 'T_max', None)
            eta_min = getattr(scheduler_config, 'eta_min', None)
            step_size = getattr(scheduler_config, 'step_size', None)
            gamma = getattr(scheduler_config, 'gamma', None)

        if patience is None:
            raise ValueError("TRAINING.lr_scheduler.patience must be set")
        if patience < 0:
            raise ValueError("TRAINING.lr_scheduler.patience must be >= 0")
        if patience == 0:
            return None
        if scheduler_name is None:
            raise ValueError("TRAINING.lr_scheduler.name must be set when patience > 0")

        if scheduler_name == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            if t_max is None:
                raise ValueError("TRAINING.lr_scheduler.T_max must be set for cosine scheduler")
            if eta_min is None:
                raise ValueError("TRAINING.lr_scheduler.eta_min must be set for cosine scheduler")
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        elif scheduler_name == 'step':
            from torch.optim.lr_scheduler import StepLR
            if step_size is None or gamma is None:
                raise ValueError("TRAINING.lr_scheduler.step_size and gamma must be set for step scheduler")
            scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name == 'reduce_on_plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            if factor is None:
                raise ValueError("TRAINING.lr_scheduler.factor must be set for reduce_on_plateau")
            scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)

            if monitor is None:
                raise ValueError("TRAINING.lr_scheduler.monitor must be set for reduce_on_plateau")
            return {
                'scheduler': scheduler,
                'monitor': monitor,
                'interval': interval,
                'frequency': frequency
            }
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        return {
            'scheduler': scheduler,
            'interval': interval,
            'frequency': frequency
        }
