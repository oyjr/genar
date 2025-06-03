import os
import random
import yaml
from addict import Dict
from pathlib import Path

import numpy as np
from scipy import sparse
import h5py
import pandas as pd
import scanpy as sc
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from hest import ( STReader, 
                VisiumReader, 
                VisiumHDReader, 
                XeniumReader )


def load_st(path, platform):
    assert platform in ['st', 'visium', 'visium-hd', 'xenium'], "platform must be one of ['st', 'visium', 'visium-hd', 'xenium']"
    
    if platform == 'st':    
        st = STReader().auto_read(path)
        
    if platform == 'visium':
        st = VisiumReader().auto_read(path)
        
    if platform == 'visium-hd':
        st = VisiumHDReader().auto_read(path)
        
    if platform == 'xenium':
        st = XeniumReader().auto_read(path)
        
    return st

def map_values(arr, step_size=256):
    """
    Map NumPy array values to integers such that:
    1. The minimum value is mapped to 0
    2. Values within 256 of each other are mapped to the same integer
    
    Args:
    arr (np.ndarray): Input NumPy array of numeric values
    
    Returns:
    tuple: 
        - NumPy array of mapped integer values 
    """
    if arr.size == 0:
        return np.array([]), {}
    
    # Sort the unique values
    unique_values = np.sort(np.unique(arr))
    
    mapping = {}
    current_key = 0
    
    mapping[unique_values[0]] = 0
    current_value = unique_values[0]

    for i in range(1, len(unique_values)):
        if unique_values[i] - current_value > step_size:
            current_key += 1
            current_value = unique_values[i] 
        
        mapping[unique_values[i]] = current_key
    
    mapped_arr = np.vectorize(mapping.get)(arr)
    
    return mapped_arr

def pxl_to_array(pixel_crds, step_size):
    x_crds = map_values(pixel_crds[:,0], step_size)
    y_crds = map_values(pixel_crds[:,1], step_size)
    dst = np.stack((x_crds, y_crds), axis=1)
    return dst

def fix_seed(seed):
    """固定随机种子"""
    if isinstance(seed, dict):
        seed = seed.get('seed', 42)  # 如果是字典，获取seed值，默认42
    seed = int(seed)  # 确保是整数
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"已设置随机种子：{seed}")
    return seed

# 注意：load_config 和 merge_configs 函数已被移除
# 现在使用内置的简化配置系统，不再需要外部YAML文件

# Load loggers
def load_loggers(cfg: Dict):
    """Return Logger instance for Trainer in List.

    Args:
        cfg (Dict): Dict containing configuration.

    Returns:
        List: _description_
    """
    
    # 设置默认日志路径
    log_path = 'logs/hest'
    if hasattr(cfg.GENERAL, 'log_path'):
        if isinstance(cfg.GENERAL.log_path, str):
            log_path = cfg.GENERAL.log_path
            
    current_time = cfg.GENERAL.current_time
    
    # 从配置文件路径获取日志名称
    config_path = Path(cfg.config)
    model_name = config_path.stem  # 获取文件名（不含扩展名）
    
    # 构建日志路径（移除fold概念）
    version_path = f"{model_name}/{current_time}/{cfg.expr_name}"
    
    # 确保日志目录存在
    Path(log_path).mkdir(exist_ok=True, parents=True)
    
    print(f'---->Log dir: {os.path.join(log_path, version_path)}')
    
    # Wandb logger
    wandb_logger = pl_loggers.WandbLogger(
        save_dir=log_path,
        name=f'{model_name}-{current_time}-{cfg.expr_name}', 
        project='ST_prediction'
    )
    
    # CSV logger
    csv_logger = pl_loggers.CSVLogger(
        save_dir=log_path,
        name=model_name,
        version=f'{current_time}/{cfg.expr_name}'
    )
    
    loggers = [wandb_logger, csv_logger]
    
    return loggers

# load Callback
def load_callbacks(cfg: Dict):
    """Return Early stopping and Checkpoint Callbacks. 

    Args:
        cfg (Dict): Dict containing configuration.

    Returns:
        List: Return List containing the Callbacks.
    """
    
    Mycallbacks = []
    
    # 🔧 根据模型类型和训练阶段设置不同的监控策略
    model_name = getattr(cfg.MODEL, 'model_name', '')
    training_stage = getattr(cfg.MODEL, 'training_stage', 1)
    
    # 动态设置监控指标和checkpoint命名
    if model_name == 'TWO_STAGE_VAR_ST':
        if training_stage == 1:
            # Stage 1: VQVAE训练 - 监控基因重建质量
            default_monitor = 'val_mse'
            default_mode = 'min'
            checkpoint_filename = 'stage1-best-epoch={epoch:02d}-val_mse={val_mse:.4f}'
            print(f"🔧 Stage 1 (VQVAE): 监控指标={default_monitor}, 模式={default_mode}")
        elif training_stage == 2:
            # Stage 2: VAR训练 - 监控token预测准确率
            default_monitor = 'val_accuracy'
            default_mode = 'max'
            checkpoint_filename = 'stage2-best-epoch={epoch:02d}-val_acc={val_accuracy:.4f}'
            print(f"🔧 Stage 2 (VAR): 监控指标={default_monitor}, 模式={default_mode}")
        else:
            raise ValueError(f"不支持的训练阶段: {training_stage}")
    else:
        # 其他模型保持原有设置
        default_monitor = 'val_loss'
        default_mode = 'min'
        checkpoint_filename = 'best-epoch={epoch:02d}-{val_mse:.4f}'
        print(f"🔧 标准模型: 监控指标={default_monitor}, 模式={default_mode}")
    
    # 处理early stopping配置
    if 'early_stopping' in cfg.CALLBACKS:
        early_stopping_cfg = cfg.CALLBACKS.early_stopping
        
        # 处理字典和Namespace两种类型
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
            monitor=str(monitor),  # 确保是字符串
            patience=int(patience),  # 确保是整数
            mode=str(mode),  # 确保是字符串
            min_delta=float(min_delta)  # 确保是浮点数
        )
        Mycallbacks.append(early_stopping)
        print(f"   ✅ Early Stopping: {monitor} (patience={patience})")
    
    # 处理model checkpoint配置
    if 'model_checkpoint' in cfg.CALLBACKS:
        ckpt_cfg = cfg.CALLBACKS.model_checkpoint
        
        # 处理字典和Namespace两种类型
        if isinstance(ckpt_cfg, dict):
            monitor = ckpt_cfg.get('monitor', default_monitor)
            save_top_k = ckpt_cfg['save_top_k']
            mode = ckpt_cfg.get('mode', default_mode)
        else:
            monitor = getattr(ckpt_cfg, 'monitor', default_monitor)
            save_top_k = ckpt_cfg.save_top_k
            mode = getattr(ckpt_cfg, 'mode', default_mode)
            
        # 确保目录存在
        os.makedirs(cfg.GENERAL.log_path, exist_ok=True)
        
        model_checkpoint = ModelCheckpoint(
            dirpath=cfg.GENERAL.log_path,  # 直接使用配置文件中指定的路径
            monitor=str(monitor),  # 确保是字符串
            save_top_k=int(save_top_k),  # 确保是整数
            mode=str(mode),  # 确保是字符串
            filename=checkpoint_filename  # 使用动态生成的文件名
        )
        Mycallbacks.append(model_checkpoint)
        print(f"   ✅ Model Checkpoint: {monitor} -> {checkpoint_filename}")
    
    return Mycallbacks

def save_hdf5(output_fpath, 
                  asset_dict, 
                  attr_dict= None, 
                  mode='a', 
                  auto_chunk = True,
                  chunk_size = None):
    """
    output_fpath: str, path to save h5 file
    asset_dict: dict, dictionary of key, val to save
    attr_dict: dict, dictionary of key: {k,v} to save as attributes for each key
    mode: str, mode to open h5 file
    auto_chunk: bool, whether to use auto chunking
    chunk_size: if auto_chunk is False, specify chunk size
    """
    with h5py.File(output_fpath, mode) as f:
        for key, val in asset_dict.items():
            data_shape = val.shape
            if len(data_shape) == 1:
                val = np.expand_dims(val, axis=1)
                data_shape = val.shape

            if key not in f: # if key does not exist, create dataset
                data_type = val.dtype
                if data_type == np.object_: 
                    data_type = h5py.string_dtype(encoding='utf-8')
                if auto_chunk:
                    chunks = True # let h5py decide chunk size
                else:
                    chunks = (chunk_size,) + data_shape[1:]
                try:
                    dset = f.create_dataset(key, 
                                            shape=data_shape, 
                                            chunks=chunks,
                                            maxshape=(None,) + data_shape[1:],
                                            dtype=data_type)
                    ### Save attribute dictionary
                    if attr_dict is not None:
                        if key in attr_dict.keys():
                            for attr_key, attr_val in attr_dict[key].items():
                                dset.attrs[attr_key] = attr_val
                    dset[:] = val
                except:
                    print(f"Error encoding {key} of dtype {data_type} into hdf5")
                
            else:
                dset = f[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                assert dset.dtype == val.dtype
                dset[-data_shape[0]:] = val
        
        # if attr_dict is not None:
        #     for key, attr in attr_dict.items():
        #         if (key in asset_dict.keys()) and (len(asset_dict[key].attrs.keys())==0):
        #             for attr_key, attr_val in attr.items():
        #                 dset[key].attrs[attr_key] = attr_val
                
    return output_fpath