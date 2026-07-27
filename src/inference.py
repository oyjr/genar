#!/usr/bin/env python3
"""Run strict, checkpoint-driven GenAR inference and paper diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import DataLoader

from configs import (
    DATASETS,
    DEFAULT_DATA_ROOT,
    ENCODER_FEATURE_DIMS,
    PAPER_BATCH_SIZE,
    PAPER_SEED,
)
from dataset.hest_dataset import STDataset
from model import ModelInterface
from model.model_metrics import ModelMetrics
from utils import fix_seed


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run GenAR inference from a self-describing checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--ckpt-path',
        '--ckpt_path',
        dest='ckpt_path',
        required=True,
    )
    parser.add_argument(
        '--dataset',
        required=True,
        choices=list(DATASETS),
    )
    parser.add_argument(
        '--slide-id',
        '--slide_id',
        dest='slide_id',
        required=True,
    )
    parser.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    parser.add_argument('--encoder', choices=list(ENCODER_FEATURE_DIMS))
    parser.add_argument(
        '--output-dir',
        '--output_dir',
        dest='output_dir',
        default='./inference_results',
    )
    parser.add_argument(
        '--gpu-id',
        '--gpu_id',
        dest='gpu_id',
        type=int,
        default=0,
        help='Use -1 for CPU',
    )
    parser.add_argument(
        '--batch-size',
        '--batch_size',
        dest='batch_size',
        type=int,
        default=PAPER_BATCH_SIZE,
    )
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-gene-count', type=int)
    parser.add_argument('--top-k', type=int)
    parser.add_argument('--seed', type=int, default=PAPER_SEED)
    parser.add_argument('--dispersion-file', type=Path)
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument(
        '--allow-slide-override',
        action='store_true',
        help=(
            'Evaluate a slide outside the checkpoint test split. The original '
            'checkpoint training split remains authoritative for diagnostics.'
        ),
    )
    parser.add_argument(
        '--allow-legacy-pickle',
        action='store_true',
        help=(
            'Allow unsafe pickle loading for a trusted pre-release checkpoint '
            'whose config contains addict.Dict'
        ),
    )
    return parser.parse_args()


def setup_device(gpu_id: int) -> torch.device:
    if gpu_id == -1:
        return torch.device('cpu')
    if gpu_id < 0:
        raise ValueError("--gpu-id must be -1 or a non-negative CUDA index")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --gpu-id -1 for CPU")
    if gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"Invalid GPU index {gpu_id}; visible count is "
            f"{torch.cuda.device_count()}"
        )
    return torch.device(f'cuda:{gpu_id}')


def _safe_checkpoint_load(
    checkpoint_path: str,
    device: torch.device,
    allow_legacy_pickle: bool,
) -> Dict[str, Any]:
    """Prefer weights-only loading; require an explicit trust flag for legacy."""
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=True,
        )
    except Exception as exc:
        if not allow_legacy_pickle:
            raise ValueError(
                "Checkpoint could not be loaded in safe weights-only mode. "
                "If this is a trusted legacy GenAR checkpoint, retry with "
                "--allow-legacy-pickle; never use that flag for an untrusted file."
            ) from exc
        warnings.warn(
            "Loading a trusted legacy checkpoint with pickle enabled.",
            RuntimeWarning,
            stacklevel=2,
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False,
        )

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a dictionary")
    return checkpoint


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    allow_legacy_pickle: bool = False,
) -> Tuple[ModelInterface, Any]:
    """Restore a strict model and its full scientific configuration."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = _safe_checkpoint_load(
        checkpoint_path,
        device,
        allow_legacy_pickle,
    )
    hyper_parameters = checkpoint.get('hyper_parameters')
    state_dict = checkpoint.get('state_dict')
    if not isinstance(hyper_parameters, dict):
        raise ValueError("Checkpoint is missing hyper_parameters")
    if 'config' not in hyper_parameters:
        raise ValueError("Checkpoint hyper_parameters are missing config")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint is missing state_dict")

    schema = checkpoint.get(
        'genar_checkpoint_schema_version',
        hyper_parameters.get('checkpoint_schema_version'),
    )
    if schema is None:
        warnings.warn(
            "Checkpoint predates the release schema marker; validating all "
            "required fields before use.",
            RuntimeWarning,
            stacklevel=2,
        )
    elif int(schema) != 1:
        raise ValueError(f"Unsupported checkpoint schema version: {schema}")

    model = ModelInterface(hyper_parameters['config'])
    incompatibilities = model.load_state_dict(state_dict, strict=True)
    if incompatibilities.missing_keys or incompatibilities.unexpected_keys:
        raise ValueError(
            "Strict checkpoint load reported incompatible keys: "
            f"{incompatibilities}"
        )
    model.to(device)
    model.eval()
    return model, model.config


def validate_checkpoint_contract(
    config,
    args: argparse.Namespace,
) -> None:
    """Prevent silent dataset, encoder, or vocabulary mismatches."""
    required = [
        'expr_name',
        'encoder_name',
        'max_gene_count',
        'MODEL',
        'DATA',
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(
            "Checkpoint config is missing: " + ', '.join(missing)
        )
    if config.expr_name != args.dataset:
        raise ValueError(
            f"Dataset mismatch: checkpoint={config.expr_name}, "
            f"CLI={args.dataset}"
        )
    if args.encoder and config.encoder_name != args.encoder:
        raise ValueError(
            f"Encoder mismatch: checkpoint={config.encoder_name}, "
            f"CLI={args.encoder}"
        )
    if (
        args.max_gene_count is not None
        and int(config.max_gene_count) != args.max_gene_count
    ):
        raise ValueError(
            f"Count-cap mismatch: checkpoint={config.max_gene_count}, "
            f"CLI={args.max_gene_count}"
        )
    expected_vocab = int(config.max_gene_count) + 1
    if (
        str(config.MODEL.model_name).upper() == 'GENAR'
        and int(config.MODEL.vocab_size) != expected_vocab
    ):
        raise ValueError(
            f"Checkpoint vocab_size={config.MODEL.vocab_size} does not match "
            f"max_gene_count+1={expected_vocab}"
        )
    configured_test_slides = _configured_slides(
        config,
        data_key='test_slides',
        root_key='slide_test',
    )
    if (
        configured_test_slides
        and args.slide_id not in configured_test_slides
        and not getattr(args, 'allow_slide_override', False)
    ):
        raise ValueError(
            f"Slide {args.slide_id} is outside the checkpoint test split "
            f"{configured_test_slides}; pass --allow-slide-override only for "
            "an intentional transfer evaluation"
        )


def _configured_slides(
    config,
    data_key: str,
    root_key: Optional[str],
) -> list[str]:
    values = config.DATA.get(data_key)
    if values:
        if isinstance(values, str):
            return [
                item.strip()
                for item in values.split(',')
                if item.strip()
            ]
        return [str(item) for item in values]
    root_value = config.get(root_key, '') if root_key else ''
    return [
        item.strip()
        for item in str(root_value).split(',')
        if item.strip()
    ]


def _ordered_gene_hash(genes) -> str:
    return hashlib.sha256(
        ('\n'.join(str(gene) for gene in genes) + '\n').encode('utf-8')
    ).hexdigest()


def validate_dataset_contract(config, dataset: STDataset) -> None:
    """Verify the checkpoint's selected-gene ordering against local data."""
    expected_hash = config.DATA.get('selected_gene_order_sha256')
    if expected_hash is None:
        warnings.warn(
            "Checkpoint does not contain a selected-gene-order hash; this "
            "legacy data contract cannot be independently verified.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    actual_hash = _ordered_gene_hash(dataset.genes)
    if actual_hash != str(expected_hash):
        raise ValueError(
            "Selected-gene order does not match the checkpoint: "
            f"expected {expected_hash}, got {actual_hash}"
        )


def checkpoint_training_slides(config, dataset: STDataset) -> list[str]:
    """Recover the immutable training split used to fit the checkpoint."""
    configured = _configured_slides(
        config,
        data_key='train_slides',
        root_key=None,
    )
    all_slides = {
        slide
        for split in dataset.slide_splits.values()
        for slide in split
    }
    if configured:
        unknown = set(configured).difference(all_slides)
        if unknown:
            raise ValueError(
                "Checkpoint training slides are absent from local data: "
                + ', '.join(sorted(unknown))
            )
        return configured

    excluded = set(
        _configured_slides(
            config,
            data_key='val_slides',
            root_key='slide_val',
        )
        + _configured_slides(
            config,
            data_key='test_slides',
            root_key='slide_test',
        )
    )
    training = [
        slide
        for slide in dataset.slide_splits['train']
        if slide not in excluded
    ]
    if not training:
        raise ValueError(
            "Could not recover a non-empty checkpoint training split"
        )
    return training


def create_test_dataloader(
    config,
    slide_id: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, STDataset]:
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    prediction_mode = str(
        config.MODEL.get(
            'prediction_mode',
            getattr(config, 'prediction_mode', 'discrete'),
        )
    )
    dataset = STDataset(
        mode='test',
        data_path=config.data_path,
        expr_name=config.expr_name,
        slide_val=slide_id,
        slide_test=slide_id,
        encoder_name=config.encoder_name,
        max_gene_count=int(config.max_gene_count),
        prediction_mode=prediction_mode,
        library_scale=float(config.MODEL.get('library_scale', 10000.0)),
        grouping_mode=str(config.DATA.get('grouping_mode', 'kmeans')),
        grouping_seed=int(config.DATA.get('grouping_seed', 42)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device_is_cuda(config),
        persistent_workers=num_workers > 0,
    )
    return loader, dataset


def device_is_cuda(config) -> bool:
    """Return whether CUDA pinning is useful for this process."""
    return torch.cuda.is_available()


def run_inference(
    model: ModelInterface,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Collect model-space outputs, untruncated counts, and weighted loss."""
    predictions = []
    targets = []
    raw_targets = []
    weighted_loss = 0.0
    sample_count = 0

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(device, non_blocking=True)
            results = model.manual_inference_step(batch, phase='test')
            current_batch = int(results['predictions'].shape[0])
            predictions.append(results['predictions'].cpu())
            targets.append(results['targets'].cpu())
            raw_targets.append(batch['raw_target_genes'].detach().cpu())
            weighted_loss += float(results['loss_final']) * current_batch
            sample_count += current_batch

    if not predictions or sample_count == 0:
        raise ValueError("Inference dataloader produced no samples")
    return (
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
        torch.cat(raw_targets, dim=0),
        weighted_loss / sample_count,
    )


def predictions_to_raw_counts(
    predictions: torch.Tensor,
    raw_targets: torch.Tensor,
    prediction_mode: str,
    library_scale: float,
) -> np.ndarray:
    """Map model outputs to physical counts for diagnostic evaluation."""
    pred = predictions.detach().cpu().numpy().astype(np.float64)
    true = raw_targets.detach().cpu().numpy().astype(np.float64)
    if prediction_mode == 'discrete':
        return np.clip(np.rint(pred), 0, None)

    normalized_counts = np.clip(np.expm1(pred), 0, None)
    # The controlled regression target is library-normalized. Raw-count
    # diagnostics therefore use the observed per-spot depth solely for this
    # inverse diagnostic, matching the baseline evaluation transform.
    observed_library = true.sum(axis=1, keepdims=True)
    return normalized_counts * observed_library / float(library_scale)


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    raw_targets: torch.Tensor,
    config,
    dispersion: Optional[np.ndarray],
    training_gene_means: np.ndarray,
) -> Dict[str, Any]:
    metrics = ModelMetrics(config)
    standard = metrics.calculate_comprehensive_pcc_metrics(
        predictions,
        targets,
        apply_log2=metrics.should_apply_log2(),
    )
    prediction_mode = str(config.MODEL.get('prediction_mode', 'discrete'))
    raw_predictions = predictions_to_raw_counts(
        predictions,
        raw_targets,
        prediction_mode,
        float(config.MODEL.get('library_scale', 10000.0)),
    )
    raw_true = raw_targets.numpy().astype(np.float64)
    raw = metrics.calculate_raw_count_diagnostics(
        raw_predictions,
        raw_true,
        dispersion=dispersion,
        gene_means=training_gene_means,
    )

    pred_np = predictions.numpy()
    target_np = targets.numpy()
    gene_correlations = metrics.calculate_gene_correlations(
        np.log2(target_np + 1.0)
        if metrics.should_apply_log2()
        else target_np,
        np.log2(pred_np + 1.0)
        if metrics.should_apply_log2()
        else pred_np,
    )
    return {
        'standard': standard,
        'raw_count': raw,
        'raw_predictions': raw_predictions,
        'raw_targets': raw_true,
        'gene_correlations': gene_correlations,
    }


def _load_dispersion(
    path: Optional[Path],
    num_genes: int,
) -> Optional[np.ndarray]:
    if path is None:
        return None
    values = np.load(path, allow_pickle=False)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (num_genes,):
        raise ValueError(
            f"Dispersion file must have shape ({num_genes},), got "
            f"{values.shape}"
        )
    return values


def collect_training_count_statistics(
    dataset: STDataset,
    training_slides: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """Estimate expression-bin means and NB theta from training slides only."""
    training_slides = list(
        dataset.slide_splits.get('train', [])
        if training_slides is None
        else training_slides
    )
    if not training_slides:
        raise ValueError(
            "No training slides are available for raw-count diagnostics"
        )

    sum_counts = None
    sum_squares = None
    total_spots = 0
    for slide_id in training_slides:
        adata = dataset._load_st(slide_id)
        values = adata.X
        if sparse.issparse(values):
            values = values.toarray()
        values = np.asarray(values, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != len(dataset.genes)
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or not np.allclose(values, np.rint(values), atol=1.0e-4, rtol=0)
        ):
            raise ValueError(
                f"Invalid raw-count matrix for training slide {slide_id}"
            )
        slide_sum = values.sum(axis=0)
        slide_sum_squares = np.square(values).sum(axis=0)
        if sum_counts is None:
            sum_counts = slide_sum
            sum_squares = slide_sum_squares
        else:
            sum_counts += slide_sum
            sum_squares += slide_sum_squares
        total_spots += values.shape[0]

    if total_spots == 0 or sum_counts is None or sum_squares is None:
        raise ValueError("Training slides contain no spots")
    means = sum_counts / total_spots
    variances = np.maximum(
        sum_squares / total_spots - np.square(means),
        0.0,
    )
    theta = np.square(means) / np.maximum(
        variances - means,
        1.0e-6,
    )
    theta = np.clip(theta, 1.0e-3, 1.0e6)
    return {
        'slide_ids': training_slides,
        'total_spots': total_spots,
        'gene_means': means,
        'dispersion': theta,
    }


def _sha256_small_text(path: Path) -> str:
    """Hash the small gene-list provenance input."""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_results(
    metrics: Dict[str, Any],
    predictions: torch.Tensor,
    targets: torch.Tensor,
    average_loss: float,
    output_dir: str,
    slide_id: str,
    dataset: STDataset,
    config,
    checkpoint_path: str,
    save_predictions: bool,
    training_statistics: Dict[str, Any],
    dispersion_source: str,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gene_file = Path(config.data_path) / 'processed_data' / 'selected_gene_list.txt'
    summary = {
        'schema_version': 1,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': config.expr_name,
        'slide_id': slide_id,
        'checkpoint': str(Path(checkpoint_path).resolve()),
        'prediction_mode': str(config.MODEL.get('prediction_mode', 'discrete')),
        'encoder': config.encoder_name,
        'num_samples': int(predictions.shape[0]),
        'num_genes': int(predictions.shape[1]),
        'max_gene_count': int(config.max_gene_count),
        'scale_dims': list(config.MODEL.get('scale_dims', [])),
        'average_final_scale_loss': float(average_loss),
        'selected_gene_list_sha256': _sha256_small_text(gene_file),
        'raw_diagnostic_training_slides': training_statistics['slide_ids'],
        'raw_diagnostic_training_spots': int(
            training_statistics['total_spots']
        ),
        'nb_dispersion_source': dispersion_source,
        'standard_metrics': metrics['standard'],
        'raw_count_diagnostics': metrics['raw_count'],
        'raw_count_note': (
            'Continuous-mode raw diagnostics use observed per-spot library '
            'depth to invert library normalization; discrete GenAR does not.'
        ),
    }
    with (output / f'{slide_id}_inference_summary.json').open(
        'w',
        encoding='utf-8',
        newline='\n',
    ) as handle:
        json.dump(_json_ready(summary), handle, indent=2, allow_nan=False)
        handle.write('\n')

    gene_rows = []
    for index, gene_name in enumerate(dataset.genes):
        gene_rows.append(
            {
                'gene_index': index,
                'gene_name': gene_name,
                'pcc': metrics['gene_correlations'][index],
                'prediction_mean_model_space': float(
                    predictions[:, index].mean()
                ),
                'target_mean_model_space': float(targets[:, index].mean()),
                'prediction_mean_count': float(
                    metrics['raw_predictions'][:, index].mean()
                ),
                'target_mean_count': float(
                    metrics['raw_targets'][:, index].mean()
                ),
            }
        )
    pd.DataFrame(gene_rows).to_csv(
        output / f'{slide_id}_gene_statistics.csv',
        index=False,
        lineterminator='\n',
    )

    if save_predictions:
        np.savez_compressed(
            output / f'{slide_id}_predictions.npz',
            predictions=predictions.numpy(),
            targets=targets.numpy(),
            predicted_counts=metrics['raw_predictions'],
            true_counts=metrics['raw_targets'],
            gene_names=np.asarray(dataset.genes),
        )


def print_results(metrics: Dict[str, Any], average_loss: float) -> None:
    standard = metrics['standard']
    raw = metrics['raw_count']
    print(
        "GenAR inference complete: "
        f"loss={average_loss:.6f}, "
        f"PCC-10/50/200={standard['pcc_10']:.4f}/"
        f"{standard['pcc_50']:.4f}/{standard['pcc_200']:.4f}, "
        f"MSE={standard['mse']:.6f}, MAE={standard['mae']:.6f}"
    )
    print(
        "Raw-count diagnostics: "
        f"Spearman={raw['spearman']}, "
        f"MSE={raw['mse_count']:.4f}, MAE={raw['mae_count']:.4f}, "
        f"NB-NLL={raw['nb_nll']:.4f}, Zero-F1={raw['zero_f1']:.4f}"
    )


def main() -> int:
    args = parse_args()
    fix_seed(args.seed)
    device = setup_device(args.gpu_id)
    model, config = load_model_from_checkpoint(
        args.ckpt_path,
        device,
        allow_legacy_pickle=args.allow_legacy_pickle,
    )
    validate_checkpoint_contract(config, args)

    dataset_info = DATASETS[args.dataset]
    dataset_path = os.path.abspath(
        os.path.join(args.data_root, dataset_info['dir_name'])
    )
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    config.data_path = dataset_path
    if args.top_k is not None:
        if args.top_k < 1 or args.top_k > int(config.MODEL.vocab_size):
            raise ValueError(
                f"--top-k must be in [1, {config.MODEL.vocab_size}]"
            )
        model.inference_top_k = args.top_k

    loader, dataset = create_test_dataloader(
        config,
        args.slide_id,
        args.batch_size,
        args.num_workers,
    )
    validate_dataset_contract(config, dataset)
    predictions, targets, raw_targets, average_loss = run_inference(
        model,
        loader,
        device,
    )
    training_statistics = collect_training_count_statistics(
        dataset,
        checkpoint_training_slides(config, dataset),
    )
    loaded_dispersion = _load_dispersion(
        args.dispersion_file,
        predictions.shape[1],
    )
    dispersion = (
        loaded_dispersion
        if loaded_dispersion is not None
        else training_statistics['dispersion']
    )
    dispersion_source = (
        str(args.dispersion_file.resolve())
        if args.dispersion_file is not None
        else 'estimated_from_training_slides'
    )
    metrics = calculate_metrics(
        predictions,
        targets,
        raw_targets,
        config,
        dispersion,
        training_statistics['gene_means'],
    )
    print_results(metrics, average_loss)
    save_results(
        metrics,
        predictions,
        targets,
        average_loss,
        args.output_dir,
        args.slide_id,
        dataset,
        config,
        args.ckpt_path,
        args.save_predictions,
        training_statistics,
        dispersion_source,
    )
    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Inference failed")
        raise
