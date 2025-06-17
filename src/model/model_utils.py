"""
模型工具函数模块
包含配置管理、模型加载、数据处理等工具函数
"""

import os
import inspect
import importlib
import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from addict import Dict as AddictDict

# 默认常量
DEFAULT_NUM_GENES = 200
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_GRADIENT_CLIP = 1.0


class ModelUtils:
    """模型工具函数类"""
    
    def __init__(self, config, lightning_module):
        self.config = config
        self.lightning_module = lightning_module
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 获取标准化设置
        self.normalize = self.get_config('DATA.normalize', True)

    def get_config(self, path: str, default=None):
        """安全地获取配置值"""
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
        """加载Multi-Scale Gene VAR模型"""
        try:
            self._logger.info("加载Multi-Scale Gene VAR模型...")
            Model = getattr(importlib.import_module(
                'model.VAR.two_stage_var_st'), 'MultiScaleGeneVAR')
            
            # 实例化模型
            model = self.instancialize(Model)
            self._logger.info("Multi-Scale Gene VAR模型加载成功")
            
            return model
            
        except Exception as e:
            self._logger.error(f"加载Multi-Scale Gene VAR模型时出错：{str(e)}")
            raise ValueError(f'Multi-Scale Gene VAR模型加载失败: {str(e)}')

    def instancialize(self, Model, **other_args):
        """实例化模型"""
        try:
            # 获取模型初始化参数
            class_args = inspect.getfullargspec(Model.__init__).args[1:]
            
            # 处理不同类型的配置对象
            model_config = self.config.MODEL
            if isinstance(model_config, AddictDict):
                model_config_dict = dict(model_config)
            elif hasattr(model_config, '__dict__'):
                model_config_dict = vars(model_config)
            else:
                model_config_dict = model_config
            
            args = {}
            
            # 从配置中获取参数
            for arg in class_args:
                if arg in model_config_dict:
                    args[arg] = model_config_dict[arg]
                elif arg == 'config':
                    args[arg] = self.config
                elif arg == 'histology_feature_dim' and 'feature_dim' in model_config_dict:
                    args[arg] = model_config_dict['feature_dim']
                    
            # 添加其他参数
            args.update(other_args)
            
            return Model(**args)
            
        except Exception as e:
            self._logger.error(f"模型实例化失败：{str(e)}")
            raise

    def preprocess_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预处理输入数据"""
        # 验证输入
        self.validate_inputs(inputs)
        
        # 创建处理后的输入副本
        processed_inputs = {}
        
        # 组织学特征处理
        if 'img' in inputs:
            processed_inputs['histology_features'] = inputs['img']
        # 空间坐标处理
        if 'positions' in inputs:
            processed_inputs['spatial_coords'] = inputs['positions']
        # 基因表达数据处理 - 保留原始逻辑，让_common_step处理推理逻辑
        if 'target_genes' in inputs:
            processed_inputs['target_genes'] = inputs['target_genes']
        
        # 确保张量在正确的设备上
        for key, value in processed_inputs.items():
            if torch.is_tensor(value):
                processed_inputs[key] = value.to(self.lightning_module.device)
        
        return processed_inputs

    def validate_inputs(self, inputs: Dict[str, torch.Tensor]):
        """验证输入数据的有效性"""
        required_keys = ['img']
        
        # 检查必需的键
        for key in required_keys:
            if key not in inputs:
                raise ValueError(f"缺少必需的输入键: {key}")
                
        # 定义不同字段的期望维度
        expected_dims = {
            'img': [2, 3],           # 图像特征: (batch, features) 或 (batch, seq, features)
            'target_genes': [2, 3],   # 基因表达: (batch, genes) 或 (batch, seq, genes)
            'positions': [2, 3],      # 空间坐标: (batch, coords) 或 (batch, seq, coords)
            'spot_idx': [1, 2],       # spot索引: (batch,) 或 (batch, seq)
            'slide_id': [1],          # slide标识: (batch,)
            'gene_ids': [1, 2],       # 基因ID: (batch,) 或 (batch, genes)
        }
        
        # 验证张量形状
        for key, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                # 获取该字段的期望维度，如果未定义则允许1-3维
                allowed_dims = expected_dims.get(key, [1, 2, 3])
                
                if tensor.dim() not in allowed_dims:
                    raise ValueError(f"{key}维度错误: {tensor.shape}，期望维度: {allowed_dims}")
        
        # 验证数值范围
        if 'target_genes' in inputs:
            targets = inputs['target_genes']
            if (targets < 0).any():
                raise ValueError("目标基因表达值包含负数")

    def apply_log2_normalization_for_evaluation(self, predictions: torch.Tensor, 
                                                targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用log2(x+1)标准化用于评估"""
        try:
            # 应用log2(x+1)变换
            pred_log2 = torch.log2(predictions + 1.0)
            target_log2 = torch.log2(targets + 1.0)
            
            # 检查NaN值
            if torch.isnan(pred_log2).any() or torch.isnan(target_log2).any():
                self._logger.warning("Log2变换后发现NaN值，使用原始值")
                return predictions, targets
                
            return pred_log2, target_log2
            
        except Exception as e:
            self._logger.warning(f"Log2标准化失败: {e}，使用原始值")
            return predictions, targets



    def scale_learning_rate(self, base_lr: float) -> float:
        """根据批次大小和GPU数量缩放学习率"""
        try:
            # 获取有效批次大小
            batch_size = self.get_config('DATA.batch_size', 32)
            
            # 获取GPU数量
            num_gpus = 1
            if hasattr(self.lightning_module.trainer, 'num_devices'):
                num_gpus = self.lightning_module.trainer.num_devices
            elif hasattr(self.lightning_module.trainer, 'gpus'):
                if isinstance(self.lightning_module.trainer.gpus, int):
                    num_gpus = self.lightning_module.trainer.gpus
                elif isinstance(self.lightning_module.trainer.gpus, (list, tuple)):
                    num_gpus = len(self.lightning_module.trainer.gpus)
            
            # 计算有效批次大小
            effective_batch_size = batch_size * num_gpus
            
            # 线性缩放规则：lr = base_lr * (effective_batch_size / base_batch_size)
            base_batch_size = 32  # 基准批次大小
            scaled_lr = base_lr * (effective_batch_size / base_batch_size)
            
            self._logger.info(f"学习率缩放: {base_lr:.6f} -> {scaled_lr:.6f} "
                            f"(batch_size={batch_size}, num_gpus={num_gpus})")
            
            return scaled_lr
            
        except Exception as e:
            self._logger.warning(f"学习率缩放失败: {e}，使用原始学习率")
            return base_lr

    def get_scheduler_config(self, optimizer):
        """获取学习率调度器配置"""
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
                self._logger.warning(f"不支持的调度器: {scheduler_name}")
                return None
                
            return {
                'scheduler': scheduler,
                'interval': scheduler_config.get('interval', 'epoch'),
                'frequency': scheduler_config.get('frequency', 1)
            }
            
        except Exception as e:
            self._logger.error(f"创建学习率调度器失败: {e}")
            return None 