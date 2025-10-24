"""Model metrics helpers focused on PCC-based evaluation."""

import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

# Default constants
DEFAULT_NUM_GENES = 200
MIN_VARIANCE_THRESHOLD = 1e-8


class ModelMetrics:
    """Lightweight metric computation utility."""
    
    def __init__(self, config, lightning_module):
        self.config = config
        self.lightning_module = lightning_module
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Resolve number of genes
        self.num_genes = self._get_config('MODEL.num_genes', DEFAULT_NUM_GENES)
        self._logger.info(f"GenAR configured with {self.num_genes} genes")

    def _get_config(self, path: str, default=None):
        """Safely fetch nested configuration values."""
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
        """Log additional per-phase statistics."""
        try:
            # Determine batch size from predictions when available
            batch_size = 1
            if 'predictions' in results_dict:
                preds = results_dict['predictions']
                if torch.is_tensor(preds):
                    batch_size = preds.size(0) if preds.dim() > 0 else 1
            
            # Log prediction statistics
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
            
            # Log auxiliary tensor metrics (e.g., loss components)
            for key, value in results_dict.items():
                if key.startswith('loss_') and torch.is_tensor(value):
                    self.lightning_module.log(f'{phase}_{key}', value.item(), 
                                            batch_size=batch_size, sync_dist=True)
                    
        except Exception as e:
            self._logger.debug(f"Failed to log model-specific metrics: {e}")

    def calculate_gene_correlations(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute Pearson correlations per gene."""
        num_genes = y_true.shape[1]
        correlations = np.zeros(num_genes)
        
        for i in range(num_genes):
            true_gene = y_true[:, i]
            pred_gene = y_pred[:, i]
            
            # Guard against constant vectors
            if np.std(true_gene) == 0 or np.std(pred_gene) == 0:
                correlations[i] = 0.0
            else:
                corr = np.corrcoef(true_gene, pred_gene)[0, 1]
                correlations[i] = 0.0 if np.isnan(corr) else corr
        
        return correlations

    def calculate_comprehensive_pcc_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, 
                                           apply_log2: bool = True) -> Dict[str, float]:
        """Compute aggregated PCC metrics shared with the inference script.

        Args:
            predictions: Model predictions.
            targets: Ground-truth values.
            apply_log2: Apply log2(x+1) transformation if the inputs are raw counts.
        """
        import numpy as np
        
        # Convert tensors to numpy arrays
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # Track value ranges for debugging
        self._logger.debug(f"Input ranges - predictions: [{predictions.min():.2f}, {predictions.max():.2f}], "
                          f"targets: [{targets.min():.2f}, {targets.max():.2f}]")
        
        if apply_log2:
            # Apply log2(x+1) transformation for evaluation
            y_true_log2 = np.log2(targets + 1.0)
            y_pred_log2 = np.log2(predictions + 1.0)
            
            # Guard against NaN values
            if np.isnan(y_true_log2).any() or np.isnan(y_pred_log2).any():
                self._logger.warning("NaN detected after log2 transform; falling back to raw values")
                y_true_log2 = targets
                y_pred_log2 = predictions
            else:
                self._logger.debug(f"Log2 ranges - predictions: [{y_pred_log2.min():.2f}, {y_pred_log2.max():.2f}], "
                                  f"targets: [{y_true_log2.min():.2f}, {y_true_log2.max():.2f}]")
        else:
            # Already provided in log space
            y_true_log2 = targets
            y_pred_log2 = predictions
            self._logger.debug("Using pre-transformed log2 inputs")

        # Gene-level correlations
        num_genes = y_true_log2.shape[1]
        correlations = np.zeros(num_genes)
        
        for i in range(num_genes):
            true_gene = y_true_log2[:, i]
            pred_gene = y_pred_log2[:, i]
            
            # Guard against constant vectors
            if np.std(true_gene) == 0 or np.std(pred_gene) == 0:
                correlations[i] = 0.0
            else:
                corr = np.corrcoef(true_gene, pred_gene)[0, 1]
                correlations[i] = 0.0 if np.isnan(corr) else corr
        
        # Sort correlations descending
        sorted_corr = np.sort(correlations)[::-1]
        
        # Aggregate PCC values
        pcc_10 = np.mean(sorted_corr[:10]) if len(sorted_corr) >= 10 else np.mean(sorted_corr)
        pcc_50 = np.mean(sorted_corr[:50]) if len(sorted_corr) >= 50 else np.mean(sorted_corr)
        pcc_200 = np.mean(sorted_corr[:200]) if len(sorted_corr) >= 200 else np.mean(sorted_corr)
        
        # Error metrics based on log-transformed values
        mse = np.mean((y_true_log2 - y_pred_log2) ** 2)
        mae = np.mean(np.abs(y_true_log2 - y_pred_log2))
        
        # Relative variance difference (log domain)
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
