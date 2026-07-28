import logging
import os
import random
from addict import Dict
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def fix_seed(seed):
    """Fix random seeds across common frameworks."""
    if seed is None:
        raise ValueError("Seed must not be None")
    if isinstance(seed, dict):
        raise ValueError("Seed must be an int, not a dict")
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    logging.getLogger(__name__).info("Seed set to %d", seed)
    return seed

def load_loggers(cfg: Dict):
    """Create the CSV logger used by Lightning."""

    log_path = 'logs/hest'
    if hasattr(cfg.GENERAL, 'log_path') and isinstance(cfg.GENERAL.log_path, str):
        log_path = cfg.GENERAL.log_path

    current_time = cfg.GENERAL.current_time
    config_path = Path(cfg.config)
    model_name = config_path.stem
    version_path = f"{model_name}/{current_time}/{cfg.expr_name}"

    Path(log_path).mkdir(exist_ok=True, parents=True)

    logging.getLogger(__name__).debug("Log dir: %s", os.path.join(log_path, version_path))

    csv_logger = pl_loggers.CSVLogger(
        save_dir=log_path,
        name=model_name,
        version=f'{current_time}/{cfg.expr_name}'
    )

    return [csv_logger]

def load_callbacks(cfg: Dict):
    """Create the configured early-stopping and checkpoint callbacks."""
    
    callbacks = []
    
    default_monitor = cfg.TRAINING.get('monitor', 'train_loss_final')
    default_mode = cfg.TRAINING.get('mode', 'min')
    logging.getLogger(__name__).debug("Default monitor: %s, mode: %s", default_monitor, default_mode)
    
    if 'early_stopping' in cfg.CALLBACKS:
        early_stopping_cfg = cfg.CALLBACKS.early_stopping
        
        if isinstance(early_stopping_cfg, dict):
            monitor = early_stopping_cfg.get('monitor', default_monitor)
            patience = early_stopping_cfg['patience']
            mode = early_stopping_cfg.get('mode', default_mode)
            min_delta = early_stopping_cfg.get('min_delta', 0.0)
        else:
            monitor = getattr(early_stopping_cfg, 'monitor', default_monitor)
            patience = early_stopping_cfg.patience
            mode = getattr(early_stopping_cfg, 'mode', default_mode)
            min_delta = getattr(early_stopping_cfg, 'min_delta', 0.0)
            
        early_stopping = EarlyStopping(
            monitor=str(monitor),
            patience=int(patience),
            mode=str(mode),
            min_delta=float(min_delta)
        )
        callbacks.append(early_stopping)
        logging.getLogger(__name__).debug("EarlyStopping configured: monitor=%s patience=%s", monitor, patience)
    
    if 'model_checkpoint' in cfg.CALLBACKS:
        ckpt_cfg = cfg.CALLBACKS.model_checkpoint
        
        default_filename = f"best-epoch={{epoch:02d}}-{default_monitor}={{{default_monitor}:.4f}}"
        
        if isinstance(ckpt_cfg, dict):
            monitor = ckpt_cfg.get('monitor', default_monitor)
            save_top_k = ckpt_cfg['save_top_k']
            mode = ckpt_cfg.get('mode', default_mode)
            filename = ckpt_cfg.get('filename', default_filename)
        else:
            monitor = getattr(ckpt_cfg, 'monitor', default_monitor)
            save_top_k = ckpt_cfg.save_top_k
            mode = getattr(ckpt_cfg, 'mode', default_mode)
            filename = getattr(ckpt_cfg, 'filename', default_filename)
            
        os.makedirs(cfg.GENERAL.log_path, exist_ok=True)
        
        model_checkpoint = ModelCheckpoint(
            dirpath=cfg.GENERAL.log_path,
            monitor=str(monitor),
            save_top_k=int(save_top_k),
            mode=str(mode),
            filename=filename,
            verbose=True
        )
        callbacks.append(model_checkpoint)
        callback_logger = logging.getLogger(__name__)
        callback_logger.debug(
            "ModelCheckpoint: monitor=%s filename=%s dirpath=%s mode=%s save_top_k=%s",
            monitor,
            filename,
            cfg.GENERAL.log_path,
            mode,
            save_top_k
        )
    
    return callbacks
