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
        try:
            # Base batch size
            batch_size = self.get_config('DATA.batch_size', 32)
            
            # Number of devices
            num_gpus = 1
            if hasattr(self.lightning_module.trainer, 'num_devices'):
                num_gpus = self.lightning_module.trainer.num_devices
            elif hasattr(self.lightning_module.trainer, 'gpus'):
                if isinstance(self.lightning_module.trainer.gpus, int):
                    num_gpus = self.lightning_module.trainer.gpus
                elif isinstance(self.lightning_module.trainer.gpus, (list, tuple)):
                    num_gpus = len(self.lightning_module.trainer.gpus)
            
            # Effective batch size
            effective_batch_size = batch_size * num_gpus
            
            # Linear scaling rule: lr = base_lr * (effective / base)
            base_batch_size = 32
            scaled_lr = base_lr * (effective_batch_size / base_batch_size)
            
            self._logger.info(f"Learning rate scaled: {base_lr:.6f} -> {scaled_lr:.6f} "
                            f"(batch_size={batch_size}, num_devices={num_gpus})")
            
            return scaled_lr
            
        except Exception as e:
            self._logger.warning(f"LR scaling failed: {e}; using base LR")
            return base_lr

    def get_scheduler_config(self, optimizer):
        """Return the LR scheduler configuration dictionary."""
        scheduler_config = self.get_config('OPTIMIZER.scheduler', {})
        
        if not scheduler_config:
            return None
            
        try:
            scheduler_name = scheduler_config.get('name', 'cosine')
            
            if scheduler_name == 'cosine':
                from torch.optim.lr_scheduler import CosineAnnealingLR
                T_max = scheduler_config.get('T_max', 100)
                eta_min = scheduler_config.get('eta_min', 0)
                scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
                
            elif scheduler_name == 'step':
                from torch.optim.lr_scheduler import StepLR
                step_size = scheduler_config.get('step_size', 30)
                gamma = scheduler_config.get('gamma', 0.1)
                scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
                
            elif scheduler_name == 'reduce_on_plateau':
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                mode = scheduler_config.get('mode', 'min')
                factor = scheduler_config.get('factor', 0.5)
                patience = scheduler_config.get('patience', 10)
                scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
                
                return {
                    'scheduler': scheduler,
                    'monitor': scheduler_config.get('monitor', 'val_loss'),
                    'interval': 'epoch',
                    'frequency': 1
                }
            else:
                self._logger.warning(f"Unsupported scheduler: {scheduler_name}")
                return None
                
            return {
                'scheduler': scheduler,
                'interval': scheduler_config.get('interval', 'epoch'),
                'frequency': scheduler_config.get('frequency', 1)
            }
            
        except Exception as e:
            self._logger.error(f"Failed to create LR scheduler: {e}")
            return None 
