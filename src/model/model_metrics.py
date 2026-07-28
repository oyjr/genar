"""Metrics used during training and evaluation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.special import gammaln
from scipy.stats import spearmanr


class ModelMetrics:
    """Compute log-space headline metrics and raw-count diagnostics."""

    _EPS = 1.0e-12

    def __init__(self, config, lightning_module=None):
        self.config = config
        self.lightning_module = lightning_module
        self._logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.num_genes = self._get_config('MODEL.num_genes')
        if self.num_genes is None:
            raise ValueError("Missing MODEL.num_genes in config")

    def _get_config(self, path: str, default=None):
        value = self.config
        for part in path.split('.'):
            if isinstance(value, dict):
                if part not in value:
                    return default
                value = value[part]
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return default
        return value

    def should_apply_log2(self) -> bool:
        """Discrete tokens are evaluated in log2(count+1); regression is pre-log."""
        mode = self._get_config(
            'MODEL.prediction_mode',
            self._get_config('prediction_mode', 'discrete'),
        )
        return str(mode).lower() != 'continuous'

    def log_model_specific_metrics(
        self,
        phase: str,
        results_dict: Dict[str, Any],
    ) -> None:
        """Log lightweight prediction statistics during Lightning execution."""
        if self.lightning_module is None:
            return
        predictions = results_dict.get('predictions')
        if not torch.is_tensor(predictions):
            return
        batch_size = predictions.size(0) if predictions.ndim else 1
        self.lightning_module.log(
            f'{phase}_pred_mean',
            predictions.mean(),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.lightning_module.log(
            f'{phase}_pred_std',
            predictions.std(),
            batch_size=batch_size,
            sync_dist=True,
        )
        self.lightning_module.log(
            f'{phase}_pred_range',
            predictions.max() - predictions.min(),
            batch_size=batch_size,
            sync_dist=True,
        )

    @staticmethod
    def _as_numpy_2d(values, name: str) -> np.ndarray:
        if torch.is_tensor(values):
            values = values.detach().cpu().numpy()
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a 2D [spots, genes] array")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return array

    def calculate_gene_correlations(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """Return per-gene Pearson correlations; undefined genes are NaN."""
        true = self._as_numpy_2d(y_true, 'y_true')
        pred = self._as_numpy_2d(y_pred, 'y_pred')
        if true.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true={true.shape}, y_pred={pred.shape}"
            )

        true_centered = true - true.mean(axis=0, keepdims=True)
        pred_centered = pred - pred.mean(axis=0, keepdims=True)
        true_norm = np.sqrt(np.sum(np.square(true_centered), axis=0))
        pred_norm = np.sqrt(np.sum(np.square(pred_centered), axis=0))
        valid = (true_norm > self._EPS) & (pred_norm > self._EPS)

        correlations = np.full(true.shape[1], np.nan, dtype=np.float64)
        correlations[valid] = (
            np.sum(
                true_centered[:, valid] * pred_centered[:, valid],
                axis=0,
            )
            / (true_norm[valid] * pred_norm[valid])
        )
        correlations[valid] = np.clip(correlations[valid], -1.0, 1.0)
        return correlations

    def calculate_comprehensive_pcc_metrics(
        self,
        predictions,
        targets,
        apply_log2: bool = True,
    ) -> Dict[str, Any]:
        """Compute PCC-k plus MSE/MAE in the paper's standard evaluation space."""
        pred = self._as_numpy_2d(predictions, 'predictions')
        true = self._as_numpy_2d(targets, 'targets')
        if pred.shape != true.shape:
            raise ValueError(
                f"Shape mismatch: predictions={pred.shape}, targets={true.shape}"
            )

        if apply_log2:
            if np.any(pred < 0) or np.any(true < 0):
                raise ValueError("Raw-count log2 evaluation requires non-negative values")
            pred_eval = np.log2(pred + 1.0)
            true_eval = np.log2(true + 1.0)
        else:
            pred_eval = pred
            true_eval = true

        correlations = self.calculate_gene_correlations(true_eval, pred_eval)
        valid_correlations = correlations[np.isfinite(correlations)]
        if valid_correlations.size == 0:
            self._logger.warning(
                "No genes have defined Pearson correlation; reporting NaN "
                "PCC values without interrupting loss-monitored training"
            )
        sorted_correlations = np.sort(valid_correlations)[::-1]

        def top_mean(k: int) -> float:
            if not sorted_correlations.size:
                return float('nan')
            return float(np.mean(sorted_correlations[: min(k, len(sorted_correlations))]))

        target_variance = np.var(true_eval, axis=0)
        prediction_variance = np.var(pred_eval, axis=0)
        variance_mask = target_variance > self._EPS
        rvd = float('nan')
        if np.any(variance_mask):
            rvd = float(
                np.mean(
                    (
                        (
                            prediction_variance[variance_mask]
                            - target_variance[variance_mask]
                        )
                        / target_variance[variance_mask]
                    )
                    ** 2
                )
            )

        return {
            'pcc_10': top_mean(10),
            'pcc_50': top_mean(50),
            'pcc_200': top_mean(200),
            'mse': float(np.mean(np.square(true_eval - pred_eval))),
            'mae': float(np.mean(np.abs(true_eval - pred_eval))),
            'rvd': rvd,
            'valid_pcc_genes': int(valid_correlations.size),
            'undefined_pcc_genes': int(correlations.size - valid_correlations.size),
        }

    @classmethod
    def estimate_nb_dispersion(cls, true_counts: np.ndarray) -> np.ndarray:
        """Estimate gene-wise NB theta with a method-of-moments diagnostic."""
        counts = cls._as_numpy_2d(true_counts, 'true_counts')
        means = counts.mean(axis=0)
        variances = counts.var(axis=0)
        adjusted = np.maximum(variances - means, 1.0e-6)
        theta = np.square(means) / adjusted
        return np.clip(theta, 1.0e-3, None)

    @classmethod
    def negative_binomial_nll(
        cls,
        predicted_mean: np.ndarray,
        true_counts: np.ndarray,
        dispersion: Optional[np.ndarray] = None,
    ) -> float:
        """Mean NB NLL parameterized by predicted mean and gene-wise theta."""
        mean = cls._as_numpy_2d(predicted_mean, 'predicted_mean')
        observed = cls._as_numpy_2d(true_counts, 'true_counts')
        if mean.shape != observed.shape:
            raise ValueError(
                f"Shape mismatch: predicted_mean={mean.shape}, "
                f"true_counts={observed.shape}"
            )
        if np.any(mean < 0) or np.any(observed < 0):
            raise ValueError("NB diagnostics require non-negative counts")

        theta = (
            cls.estimate_nb_dispersion(observed)
            if dispersion is None
            else np.asarray(dispersion, dtype=np.float64)
        )
        if theta.shape != (observed.shape[1],):
            raise ValueError(
                f"dispersion must have shape ({observed.shape[1]},)"
            )
        if np.any(theta <= 0) or not np.isfinite(theta).all():
            raise ValueError("dispersion must be finite and positive")

        mu = np.clip(mean, 1.0e-8, None)
        theta = theta.reshape(1, -1)
        log_probability = (
            gammaln(observed + theta)
            - gammaln(theta)
            - gammaln(observed + 1.0)
            + theta * (np.log(theta) - np.log(theta + mu))
            + observed * (np.log(mu) - np.log(theta + mu))
        )
        return float(-np.mean(log_probability))

    def calculate_raw_count_diagnostics(
        self,
        predicted_counts,
        true_counts,
        dispersion: Optional[np.ndarray] = None,
        gene_means: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute Table 7-9 style diagnostics in physical count space."""
        pred = self._as_numpy_2d(predicted_counts, 'predicted_counts')
        true = self._as_numpy_2d(true_counts, 'true_counts')
        if pred.shape != true.shape:
            raise ValueError(
                f"Shape mismatch: predicted_counts={pred.shape}, "
                f"true_counts={true.shape}"
            )
        if np.any(pred < 0) or np.any(true < 0):
            raise ValueError("Raw-count diagnostics require non-negative values")

        spearman_per_gene = np.full(true.shape[1], np.nan, dtype=np.float64)
        for gene_index in range(true.shape[1]):
            spearman_per_gene[gene_index] = self._safe_vector_spearman(
                pred[:, gene_index],
                true[:, gene_index],
            )
        valid_spearman = spearman_per_gene[np.isfinite(spearman_per_gene)]
        mean_spearman = (
            float(valid_spearman.mean())
            if valid_spearman.size
            else float('nan')
        )
        true_zero = true == 0
        predicted_zero = pred == 0
        zero_tp = int(np.logical_and(true_zero, predicted_zero).sum())
        predicted_zero_count = int(predicted_zero.sum())
        true_zero_count = int(true_zero.sum())
        zero_precision = (
            zero_tp / predicted_zero_count if predicted_zero_count else 0.0
        )
        zero_recall = zero_tp / true_zero_count if true_zero_count else 0.0
        zero_f1 = (
            2.0 * zero_precision * zero_recall
            / (zero_precision + zero_recall)
            if zero_precision + zero_recall
            else 0.0
        )

        predicted_library = pred.sum(axis=1)
        true_library = true.sum(axis=1)
        library_spearman = self._safe_vector_spearman(
            predicted_library,
            true_library,
        )
        library_pearson = self._safe_vector_pearson(
            predicted_library,
            true_library,
        )

        correlations = self.calculate_gene_correlations(
            np.log2(true + 1.0),
            np.log2(pred + 1.0),
        )
        means = (
            true.mean(axis=0)
            if gene_means is None
            else np.asarray(gene_means, dtype=np.float64)
        )
        if means.shape != (true.shape[1],):
            raise ValueError(
                f"gene_means must have shape ({true.shape[1]},)"
            )
        if not np.isfinite(means).all() or np.any(means < 0):
            raise ValueError("gene_means must be finite and non-negative")
        bins = {
            'low_lt_10': means < 10,
            'medium_10_to_100': (means >= 10) & (means <= 100),
            'high_gt_100': means > 100,
        }
        stratified: Dict[str, Dict[str, Any]] = {}
        for name, mask in bins.items():
            values = correlations[mask & np.isfinite(correlations)]
            stratified[name] = {
                'gene_count': int(mask.sum()),
                'valid_pcc_gene_count': int(values.size),
                'pcc': float(np.mean(values)) if values.size else None,
            }

        return {
            'spearman': self._optional_float(mean_spearman),
            'spearman_per_gene': [
                self._optional_float(value)
                for value in spearman_per_gene
            ],
            'valid_spearman_genes': int(valid_spearman.size),
            'mse_count': float(np.mean(np.square(true - pred))),
            'mae_count': float(np.mean(np.abs(true - pred))),
            'nb_nll': self.negative_binomial_nll(pred, true, dispersion),
            'zero_precision': float(zero_precision),
            'zero_recall': float(zero_recall),
            'zero_f1': float(zero_f1),
            'zero_true_positive_count': zero_tp,
            'predicted_zero_count': predicted_zero_count,
            'true_zero_count': true_zero_count,
            'library_size_spearman': self._optional_float(library_spearman),
            'library_size_pearson': self._optional_float(library_pearson),
            'expression_bins': stratified,
        }

    @classmethod
    def _safe_vector_pearson(
        cls,
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        first_centered = first - first.mean()
        second_centered = second - second.mean()
        denominator = np.sqrt(
            np.sum(np.square(first_centered))
            * np.sum(np.square(second_centered))
        )
        if denominator <= cls._EPS:
            return float('nan')
        return float(
            np.sum(first_centered * second_centered) / denominator
        )

    @classmethod
    def _safe_vector_spearman(
        cls,
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        if (
            np.ptp(first) <= cls._EPS
            or np.ptp(second) <= cls._EPS
        ):
            return float('nan')
        return float(spearmanr(first, second).statistic)

    @staticmethod
    def _optional_float(value) -> Optional[float]:
        value = float(value)
        return value if np.isfinite(value) else None
