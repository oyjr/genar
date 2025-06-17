"""
模型指标计算和评估模块
简化版本，只保留核心的PCC指标计算功能
"""

import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

# 默认常量
DEFAULT_NUM_GENES = 200
MIN_VARIANCE_THRESHOLD = 1e-8


class ModelMetrics:
    """简化的模型指标计算和管理类"""
    
    def __init__(self, config, lightning_module):
        self.config = config
        self.lightning_module = lightning_module
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 获取基因数量
        self.num_genes = self._get_config('MODEL.num_genes', DEFAULT_NUM_GENES)
        self._logger.info(f"VAR_ST模型使用基因数量: {self.num_genes}")
        
    def _get_config(self, path: str, default=None):
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

    def log_model_specific_metrics(self, phase: str, results_dict: Dict[str, Any]):
        """记录模型特定的指标"""
        try:
            # 获取batch_size
            batch_size = 1
            if 'predictions' in results_dict:
                preds = results_dict['predictions']
                if torch.is_tensor(preds):
                    batch_size = preds.size(0) if preds.dim() > 0 else 1
            
            # 记录模型输出的统计信息
            if 'predictions' in results_dict:
                preds = results_dict['predictions']
                if torch.is_tensor(preds):
                    pred_mean = preds.mean()
                    pred_std = preds.std()
                    pred_min = preds.min()
                    pred_max = preds.max()
                    
                    self.lightning_module.log(f'{phase}_pred_mean', pred_mean, 
                                            batch_size=batch_size, sync_dist=True)
                    self.lightning_module.log(f'{phase}_pred_std', pred_std, 
                                            batch_size=batch_size, sync_dist=True)
                    self.lightning_module.log(f'{phase}_pred_range', pred_max - pred_min, 
                                            batch_size=batch_size, sync_dist=True)
            
            # 记录其他模型特定信息
            for key, value in results_dict.items():
                if key.startswith('loss_') and torch.is_tensor(value):
                    self.lightning_module.log(f'{phase}_{key}', value.item(), 
                                            batch_size=batch_size, sync_dist=True)
                    
        except Exception as e:
            self._logger.debug(f"记录模型特定指标时出现问题: {e}")

    def calculate_gene_correlations(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """计算基因级别的相关系数"""
        num_genes = y_true.shape[1]
        correlations = np.zeros(num_genes)
        
        for i in range(num_genes):
            true_gene = y_true[:, i]
            pred_gene = y_pred[:, i]
            
            # 处理常数值
            if np.std(true_gene) == 0 or np.std(pred_gene) == 0:
                correlations[i] = 0.0
            else:
                corr = np.corrcoef(true_gene, pred_gene)[0, 1]
                correlations[i] = 0.0 if np.isnan(corr) else corr
        
        return correlations

    def calculate_comprehensive_pcc_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                           apply_log2: bool = True) -> Dict[str, float]:
        """
        计算综合PCC指标 - 与推理脚本保持一致
        
        Args:
            predictions: 预测值
            targets: 真实值
            apply_log2: 是否应用log2(x+1)变换，如果数据已经变换过则设为False
        """
        import numpy as np
        
        # 转换为numpy数组
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # 添加数值范围检查
        self._logger.debug(f"输入数据范围 - 预测值: [{predictions.min():.2f}, {predictions.max():.2f}], "
                          f"真实值: [{targets.min():.2f}, {targets.max():.2f}]")
        
        if apply_log2:
            # 应用log2(x+1)变换用于评估指标计算（与推理脚本保持一致）
            y_true_log2 = np.log2(targets + 1.0)
            y_pred_log2 = np.log2(predictions + 1.0)
            
            # 检查NaN值
            if np.isnan(y_true_log2).any() or np.isnan(y_pred_log2).any():
                self._logger.warning("⚠️ Log2变换后发现NaN值，将使用原始值")
                y_true_log2 = targets
                y_pred_log2 = predictions
            else:
                self._logger.debug(f"Log2变换后数据范围 - 预测值: [{y_pred_log2.min():.2f}, {y_pred_log2.max():.2f}], "
                                  f"真实值: [{y_true_log2.min():.2f}, {y_true_log2.max():.2f}]")
        else:
            # 数据已经是log2变换后的，直接使用
            y_true_log2 = targets
            y_pred_log2 = predictions
            self._logger.debug("使用已经log2变换的数据计算PCC指标")
        
        # 计算基因级别的相关性
        num_genes = y_true_log2.shape[1]
        correlations = np.zeros(num_genes)
        
        for i in range(num_genes):
            true_gene = y_true_log2[:, i]
            pred_gene = y_pred_log2[:, i]
            
            # 处理常数值
            if np.std(true_gene) == 0 or np.std(pred_gene) == 0:
                correlations[i] = 0.0
            else:
                corr = np.corrcoef(true_gene, pred_gene)[0, 1]
                correlations[i] = 0.0 if np.isnan(corr) else corr
        
        # 排序相关性
        sorted_corr = np.sort(correlations)[::-1]
        
        # 计算PCC指标
        pcc_10 = np.mean(sorted_corr[:10]) if len(sorted_corr) >= 10 else np.mean(sorted_corr)
        pcc_50 = np.mean(sorted_corr[:50]) if len(sorted_corr) >= 50 else np.mean(sorted_corr)
        pcc_200 = np.mean(sorted_corr[:200]) if len(sorted_corr) >= 200 else np.mean(sorted_corr)
        
        # 计算MSE和MAE（使用log2变换后的值）
        mse = np.mean((y_true_log2 - y_pred_log2) ** 2)
        mae = np.mean(np.abs(y_true_log2 - y_pred_log2))
        
        # 计算RVD (Relative Variance Difference)（使用log2变换后的值）
        pred_var = np.var(y_pred_log2, axis=0)
        true_var = np.var(y_true_log2, axis=0)
        
        valid_mask = true_var > MIN_VARIANCE_THRESHOLD
        if np.sum(valid_mask) > 0:
            rvd = np.mean(((pred_var[valid_mask] - true_var[valid_mask]) ** 2) / (true_var[valid_mask] ** 2))
        else:
            rvd = 0.0
        
        return {
            'pcc_10': float(pcc_10),
            'pcc_50': float(pcc_50), 
            'pcc_200': float(pcc_200),
            'mse': float(mse),
            'mae': float(mae),
            'rvd': float(rvd)
        } 