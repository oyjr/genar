"""Utility helpers for model configuration, loading, and preprocessing."""

import inspect
import importlib
import logging
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
                    Model = getattr(importlib.import_module(
                        'model.genar.multiscale_genar_no_film'), 'MultiScaleGenARNoFiLM')
                    self._logger.info("GenAR (NoFiLM) loaded")
                else:
                    self._logger.info("Loading GenAR (original variant) ...")
                    Model = getattr(importlib.import_module(
                        'model.genar.multiscale_genar'), 'MultiScaleGenAR')
                    self._logger.info("GenAR (original) loaded")

                return self.instancialize(Model)

            except Exception as e:
                self._logger.error(f"Failed to load GenAR model: {str(e)}")
                raise ValueError(f"GenAR model load failed: {str(e)}")

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
                    
            # Merge explicit overrides
            args.update(other_args)
            
            return Model(**args)
            
        except Exception as e:
            self._logger.error(f"Model instantiation failed: {str(e)}")
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
        required_keys = ['img']
        
        # Check mandatory keys
        for key in required_keys:
            if key not in inputs:
                raise ValueError(f"Missing required input key: {key}")
                
        # Expected tensor dimensions per field
        expected_dims = {
            'img': [2, 3],
            'target_genes': [2, 3],
            'positions': [2, 3],
            'spot_idx': [1, 2],
            'slide_id': [1],
            'gene_ids': [1, 2],
        }
        
        # Shape checks
        for key, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                # Use defaults when not explicitly defined
                allowed_dims = expected_dims.get(key, [1, 2, 3])
                
                if tensor.dim() not in allowed_dims:
                    raise ValueError(f"Unexpected tensor rank for {key}: {tensor.shape}; allowed={allowed_dims}")
        
        # Numeric sanity checks
        if 'target_genes' in inputs:
            targets = inputs['target_genes']
            if (targets < 0).any():
                raise ValueError("Target gene expression contains negative values")


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
