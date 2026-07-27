#!/usr/bin/env python3
"""Train the paper-aligned GenAR model and its controlled ablations."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Iterable, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytorch_lightning as pl
import torch
from addict import Dict
from torch.utils.data import DataLoader

from configs import (
    DATASETS,
    DEFAULT_DATA_ROOT,
    ENCODER_FEATURE_DIMS,
    PAPER_BATCH_SIZE,
    PAPER_LEARNING_RATE,
    PAPER_MAX_GENE_COUNT,
    PAPER_NUM_GENES,
    PAPER_SCALE_DIMS,
    PAPER_SEED,
    SCALE_PRESETS,
)
from dataset.hest_dataset import STDataset
from model import ModelInterface
from utils import fix_seed, load_callbacks, load_loggers


logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")


GENAR_CONFIG = {
    'model_name': 'GENAR',
    'model_variant': 'original',
    'num_genes': PAPER_NUM_GENES,
    'histology_feature_dim': 1024,
    'spatial_coord_dim': 2,
    'gene_patch_nums': PAPER_SCALE_DIMS,
    'scale_dims': PAPER_SCALE_DIMS,
    'embed_dim': 512,
    'num_heads': 8,
    'num_layers': 8,
    'mlp_ratio': 3.0,
    'drop_rate': 0.0,
    'attn_drop_rate': 0.0,
    'drop_path_rate': 0.1,
    'condition_embed_dim': 512,
    'cond_drop_rate': 0.1,
    'norm_eps': 1.0e-6,
    'shared_aln': False,
    'attn_l2_norm': True,
    'adaptive_sigma_alpha': 0.01,
    'adaptive_sigma_beta': 1.0,
    'prediction_mode': 'discrete',
    'continuous_loss': 'mse',
    'continuous_loss_alpha': 0.01,
    'continuous_loss_beta': 0.1,
    'library_scale': 10000.0,
    'scale_loss_weights': [1.0] * len(PAPER_SCALE_DIMS),
    'final_loss_mode': 'gaussian_kl',
}

FOUNDATION_BASELINE_CONFIG = {
    'model_name': 'FOUNDATION_BASELINE',
    'num_genes': PAPER_NUM_GENES,
    'hidden_dim': 256,
    'num_hidden_layers': 1,
    'dropout': 0.1,
}

MODEL_CONFIGS = {
    'GENAR': GENAR_CONFIG,
    'FOUNDATION_BASELINE': FOUNDATION_BASELINE_CONFIG,
}

DEFAULT_CONFIG = {
    'GENERAL': {
        'seed': PAPER_SEED,
        'log_path': './logs',
        'debug': False,
    },
    'DATA': {
        'train_dataloader': {
            'batch_size': PAPER_BATCH_SIZE,
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': True,
            'persistent_workers': True,
        },
        'val_dataloader': {
            'batch_size': PAPER_BATCH_SIZE,
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': False,
            'persistent_workers': True,
        },
        'test_dataloader': {
            'batch_size': PAPER_BATCH_SIZE,
            'num_workers': 4,
            'pin_memory': True,
            'shuffle': False,
            'persistent_workers': True,
        },
        'grouping_mode': 'kmeans',
        'grouping_seed': 42,
    },
    'TRAINING': {
        'num_epochs': 200,
        'learning_rate': PAPER_LEARNING_RATE,
        'scale_learning_rate': False,
        'weight_decay': 1.0e-4,
        'mode': 'min',
        'monitor': 'train_loss_final',
        'lr_scheduler': {
            'name': 'reduce_on_plateau',
            'monitor': 'train_loss_final',
            'mode': 'min',
            'patience': 0,
            'factor': 0.5,
        },
        'gradient_clip_val': 1.0,
    },
    'CALLBACKS': {
        'early_stopping': {
            'monitor': 'train_loss_final',
            'patience': 10000,
            'mode': 'min',
            'min_delta': 0.0,
        },
        'model_checkpoint': {
            'monitor': 'train_loss_final',
            'save_top_k': 1,
            'mode': 'min',
            'filename': (
                'best-epoch={epoch:02d}-'
                'train_loss_final={train_loss_final:.6f}'
            ),
        },
    },
    'INFERENCE': {'top_k': 1},
    'MULTI_GPU': {
        'find_unused_parameters': False,
        'accumulate_grad_batches': 1,
    },
}


def validate_scale_dims(
    scale_dims: Iterable[int],
    num_genes: int = PAPER_NUM_GENES,
) -> Tuple[int, ...]:
    """Validate the nested hierarchy required by group-aware upsampling."""
    scales = tuple(int(dim) for dim in scale_dims)
    if not scales:
        raise ValueError("At least one scale dimension is required")
    if scales[-1] != num_genes:
        raise ValueError(
            f"The final scale must be {num_genes}, got {scales[-1]}"
        )
    if any(dim <= 0 for dim in scales):
        raise ValueError("Scale dimensions must be positive")
    if any(left >= right for left, right in zip(scales, scales[1:])):
        raise ValueError("Scale dimensions must be strictly increasing")
    if any(num_genes % dim != 0 for dim in scales):
        raise ValueError(f"Every scale must divide {num_genes}")
    return scales


def parse_scale_dims(value: str) -> Tuple[int, ...]:
    """Parse a comma-separated hierarchy."""
    try:
        scales = tuple(int(item.strip()) for item in value.split(',') if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Scale dimensions must be comma-separated integers"
        ) from exc
    try:
        return validate_scale_dims(scales)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_scale_loss_weights(
    value: str,
    expected_count: int,
) -> list[float]:
    try:
        weights = [
            float(item.strip())
            for item in value.split(',')
            if item.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            "--scale-loss-weights must be comma-separated numbers"
        ) from exc
    if len(weights) != expected_count:
        raise ValueError(
            "--scale-loss-weights must contain exactly "
            f"{expected_count} values"
        )
    if any(weight < 0 for weight in weights) or not any(
        weight > 0 for weight in weights
    ):
        raise ValueError(
            "Scale loss weights must be non-negative with at least one positive"
        )
    return weights


def get_parse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train GenAR with the final-paper defaults',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--dataset',
        choices=list(DATASETS),
        required=True,
    )
    parser.add_argument(
        '--data-root',
        default=DEFAULT_DATA_ROOT,
        help='Root containing PRAD/her2st/kidney/mouse_brain/ccRCC',
    )
    parser.add_argument(
        '--model',
        default='GENAR',
        choices=list(MODEL_CONFIGS),
    )
    parser.add_argument(
        '--encoder',
        choices=list(ENCODER_FEATURE_DIMS),
        help='Defaults to the dataset recommendation (UNI)',
    )
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--epochs', type=int)
    parser.add_argument(
        '--batch-size',
        '--batch_size',
        dest='batch_size',
        type=int,
        help=(
            'Per-process DataLoader batch size (legacy override). When omitted, '
            'the paper global batch is divided evenly across GPU processes.'
        ),
    )
    parser.add_argument(
        '--global-batch-size',
        type=int,
        help=(
            'Global batch across all GPU processes. Defaults to the paper value '
            f'of {PAPER_BATCH_SIZE}.'
        ),
    )
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight-decay', type=float)
    parser.add_argument(
        '--scale-learning-rate',
        action='store_true',
        help='Opt in to legacy linear LR scaling; off for paper alignment',
    )
    parser.add_argument(
        '--patience',
        type=int,
        help='Enable ReduceLROnPlateau when greater than zero',
    )
    parser.add_argument(
        '--strategy',
        default='auto',
        choices=['auto', 'ddp', 'ddp_spawn'],
    )
    parser.add_argument('--sync-batchnorm', action='store_true')
    parser.add_argument(
        '--precision',
        default='32',
        choices=['32', '16-mixed', 'bf16-mixed'],
    )
    parser.add_argument(
        '--allow-nondeterministic',
        action='store_true',
        help='Disable deterministic Trainer mode (not a paper setting)',
    )
    parser.add_argument(
        '--max-gene-count',
        type=int,
        default=PAPER_MAX_GENE_COUNT,
    )
    parser.add_argument(
        '--mode',
        default='train',
        choices=['train', 'test'],
    )
    parser.add_argument('--seed', type=int)
    parser.add_argument('--ckpt-path', '--ckpt_path', dest='ckpt_path')
    parser.add_argument(
        '--scale-config',
        default='paper',
        choices=list(SCALE_PRESETS),
    )
    parser.add_argument(
        '--scale-dims',
        type=parse_scale_dims,
        help='Explicit comma-separated hierarchy ending in 200',
    )
    parser.add_argument(
        '--scale-loss-weights',
        help='One comma-separated non-negative value per scale',
    )
    parser.add_argument(
        '--model-variant',
        default='original',
        choices=['original', 'no_film'],
    )
    parser.add_argument(
        '--prediction-mode',
        default='discrete',
        choices=['discrete', 'continuous'],
    )
    parser.add_argument(
        '--continuous-loss',
        default='mse',
        choices=['mse', 'gaussian_nll'],
    )
    parser.add_argument(
        '--final-loss-mode',
        default='gaussian_kl',
        choices=['gaussian_kl', 'gaussian_nll', 'cross_entropy'],
        help='gaussian_nll is accepted only as a deprecated alias',
    )
    parser.add_argument(
        '--grouping-mode',
        default='kmeans',
        choices=['kmeans', 'random'],
    )
    parser.add_argument('--grouping-seed', type=int, default=42)
    parser.add_argument(
        '--config',
        help='Deprecated and intentionally unsupported',
    )
    return parser


def build_config_from_args(args) -> Dict:
    """Build and validate a fully serializable runtime configuration."""
    if args.config:
        raise ValueError("Legacy --config is unsupported; use explicit CLI flags")
    if args.mode == 'test' and not args.ckpt_path:
        raise ValueError("--ckpt-path is required in test mode")
    if args.ckpt_path and not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")
    if args.gpus < 1:
        raise ValueError("--gpus must be at least one")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.max_gene_count < 1:
        raise ValueError("--max-gene-count must be positive")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.global_batch_size is not None and args.global_batch_size < 1:
        raise ValueError("--global-batch-size must be positive")
    if args.batch_size is not None and args.global_batch_size is not None:
        raise ValueError(
            "Use either --batch-size (per process) or --global-batch-size, "
            "not both"
        )
    if args.lr is not None and args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.weight_decay is not None and args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative")
    if args.patience is not None and args.patience < 0:
        raise ValueError("--patience cannot be negative")

    model_name = args.model.upper()
    dataset_info = DATASETS[args.dataset]
    dataset_path = os.path.abspath(
        os.path.join(args.data_root, dataset_info['dir_name'])
    )
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    encoder_name = args.encoder or dataset_info['recommended_encoder']
    config = Dict(deepcopy(DEFAULT_CONFIG))
    config.MODEL = Dict(deepcopy(MODEL_CONFIGS[model_name]))

    if args.scale_dims is not None and args.scale_config != 'paper':
        raise ValueError(
            "Use either --scale-dims or a non-default --scale-config, not both"
        )
    selected_scales = (
        validate_scale_dims(args.scale_dims)
        if args.scale_dims is not None
        else validate_scale_dims(SCALE_PRESETS[args.scale_config])
    )

    if model_name == 'GENAR':
        if args.prediction_mode == 'continuous' and args.model_variant != 'original':
            raise ValueError(
                "The continuous formulation comparison requires "
                "--model-variant original"
            )
        if (
            args.model_variant == 'no_film'
            and args.final_loss_mode == 'cross_entropy'
        ):
            raise ValueError(
                "The no_film implementation uses the paper Gaussian KL loss"
            )
        final_loss_mode = args.final_loss_mode
        if final_loss_mode == 'gaussian_nll':
            warnings.warn(
                "gaussian_nll is a legacy label for Gaussian soft-token KL; "
                "recording gaussian_kl in the checkpoint.",
                DeprecationWarning,
                stacklevel=2,
            )
            final_loss_mode = 'gaussian_kl'

        weights = (
            parse_scale_loss_weights(
                args.scale_loss_weights,
                len(selected_scales),
            )
            if args.scale_loss_weights
            else [1.0] * len(selected_scales)
        )
        config.MODEL.update(
            {
                'model_variant': args.model_variant,
                'scale_dims': selected_scales,
                'gene_patch_nums': selected_scales,
                'prediction_mode': args.prediction_mode,
                'continuous_loss': args.continuous_loss,
                'scale_loss_weights': weights,
                'final_loss_mode': final_loss_mode,
                'library_scale': 10000.0,
                'vocab_size': args.max_gene_count + 1,
                'max_gene_count': args.max_gene_count,
                'histology_feature_dim': ENCODER_FEATURE_DIMS[encoder_name],
                'feature_dim': ENCODER_FEATURE_DIMS[encoder_name],
            }
        )
        run_parts = ['GENAR']
        if args.model_variant != 'original':
            run_parts.append(args.model_variant)
        if args.prediction_mode != 'discrete':
            run_parts.append(args.prediction_mode)
        if selected_scales != PAPER_SCALE_DIMS:
            run_parts.append('scales-' + '-'.join(map(str, selected_scales)))
        if final_loss_mode != 'gaussian_kl':
            run_parts.append(final_loss_mode)
        if args.grouping_mode != 'kmeans':
            run_parts.append(f'random-grouping-{args.grouping_seed}')
        config.GENERAL.log_path = os.path.join(
            './logs',
            args.dataset,
            '_'.join(run_parts),
        )
    else:
        if args.prediction_mode != 'discrete':
            raise ValueError(
                "FOUNDATION_BASELINE currently implements discrete count-token "
                "prediction; use GENAR for the continuous matched ablation"
            )
        config.MODEL.feature_dim = ENCODER_FEATURE_DIMS[encoder_name]
        config.MODEL.histology_feature_dim = ENCODER_FEATURE_DIMS[encoder_name]
        config.MODEL.vocab_size = args.max_gene_count + 1
        config.MODEL.max_gene_count = args.max_gene_count
        config.MODEL.prediction_mode = 'discrete'
        config.GENERAL.log_path = os.path.join(
            './logs',
            args.dataset,
            model_name,
        )

    if args.epochs is not None:
        config.TRAINING.num_epochs = args.epochs
    if args.lr is not None:
        config.TRAINING.learning_rate = args.lr
    if args.weight_decay is not None:
        config.TRAINING.weight_decay = args.weight_decay
    if args.batch_size is not None:
        per_process_batch_size = args.batch_size
        global_batch_size = args.batch_size * args.gpus
    else:
        global_batch_size = (
            args.global_batch_size
            if args.global_batch_size is not None
            else PAPER_BATCH_SIZE
        )
        if global_batch_size % args.gpus:
            raise ValueError(
                f"Global batch size {global_batch_size} must be divisible by "
                f"--gpus={args.gpus}"
            )
        per_process_batch_size = global_batch_size // args.gpus
    for split in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        config.DATA[split].batch_size = per_process_batch_size
    config.DATA.global_batch_size = global_batch_size
    config.DATA.batch_size_per_process = per_process_batch_size
    for split in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        config.DATA[split].num_workers = args.num_workers
        config.DATA[split].persistent_workers = args.num_workers > 0

    config.TRAINING.scale_learning_rate = args.scale_learning_rate
    if args.patience is not None:
        config.TRAINING.lr_scheduler.patience = args.patience
        config.CALLBACKS.early_stopping.patience = (
            10000 if args.patience == 0 else max(10, args.patience * 2)
        )
    if args.seed is not None:
        config.GENERAL.seed = args.seed

    config.DATA.grouping_mode = args.grouping_mode
    config.DATA.grouping_seed = args.grouping_seed
    config.mode = args.mode
    config.expr_name = args.dataset
    config.data_path = dataset_path
    config.slide_val = dataset_info['val_slides']
    config.slide_test = dataset_info['test_slides']
    config.encoder_name = encoder_name
    config.prediction_mode = args.prediction_mode
    config.gene_count_mode = (
        'discrete_tokens'
        if args.prediction_mode == 'discrete'
        else 'library_normalized_log1p'
    )
    config.max_gene_count = args.max_gene_count
    config.devices = args.gpus
    config.strategy = (
        'ddp'
        if args.gpus > 1 and args.strategy == 'auto'
        else args.strategy
    )
    config.sync_batchnorm = bool(args.sync_batchnorm)
    config.precision = (
        int(args.precision)
        if args.precision == '32'
        else args.precision
    )
    config.deterministic = not args.allow_nondeterministic
    config.GENERAL.current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config.GENERAL.log_path = os.path.join(
        config.GENERAL.log_path,
        config.GENERAL.current_time,
    )
    config.config = 'built-in'
    if args.ckpt_path:
        config.ckpt_path = os.path.abspath(args.ckpt_path)

    logger.info(
        "Configuration: dataset=%s encoder=%s mode=%s scales=%s "
        "batch_global=%s batch_per_process=%s lr=%s max_count=%s prediction=%s",
        args.dataset,
        encoder_name,
        args.mode,
        selected_scales,
        config.DATA.global_batch_size,
        config.DATA.train_dataloader.batch_size,
        config.TRAINING.learning_rate,
        args.max_gene_count,
        args.prediction_mode,
    )
    return config


def _loader_kwargs(config: Dict, split: str) -> dict:
    values = config.DATA[f'{split}_dataloader']
    kwargs = {
        'batch_size': values.batch_size,
        'shuffle': values.shuffle,
        'num_workers': values.num_workers,
        'pin_memory': values.pin_memory,
        'persistent_workers': bool(
            values.persistent_workers and values.num_workers > 0
        ),
    }
    if split == 'train':
        generator = torch.Generator()
        generator.manual_seed(int(config.GENERAL.seed))
        kwargs['generator'] = generator
    return kwargs


def create_dataloaders(config: Dict):
    """Instantiate all split datasets with one shared scientific data contract."""
    base_params = {
        'data_path': config.data_path,
        'expr_name': config.expr_name,
        'slide_val': config.slide_val,
        'slide_test': config.slide_test,
        'encoder_name': config.encoder_name,
        'max_gene_count': config.max_gene_count,
        'prediction_mode': config.prediction_mode,
        'library_scale': config.MODEL.get('library_scale', 10000.0),
        'grouping_mode': config.DATA.grouping_mode,
        'grouping_seed': config.DATA.grouping_seed,
    }
    train_dataset = STDataset(mode='train', **base_params)
    val_dataset = STDataset(mode='val', **base_params)
    test_dataset = STDataset(mode='test', **base_params)
    if not (
        train_dataset.genes
        == val_dataset.genes
        == test_dataset.genes
    ):
        raise ValueError("Dataset splits do not share the same gene order")

    ordered_genes = '\n'.join(train_dataset.genes) + '\n'
    config.DATA.data_contract_version = 1
    config.DATA.selected_gene_order_sha256 = hashlib.sha256(
        ordered_genes.encode('utf-8')
    ).hexdigest()
    config.DATA.train_slides = list(train_dataset.slide_splits['train'])
    config.DATA.val_slides = list(train_dataset.slide_splits['val'])
    config.DATA.test_slides = list(train_dataset.slide_splits['test'])

    return (
        DataLoader(train_dataset, **_loader_kwargs(config, 'train')),
        DataLoader(val_dataset, **_loader_kwargs(config, 'val')),
        DataLoader(test_dataset, **_loader_kwargs(config, 'test')),
    )


def main(config: Dict):
    """Run a deterministic Lightning training or checkpoint evaluation."""
    fix_seed(config.GENERAL.seed)
    pl.seed_everything(config.GENERAL.seed, workers=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required by src/main.py; use src/inference.py --gpu-id -1 "
            "for CPU inference"
        )
    if config.devices > torch.cuda.device_count():
        raise RuntimeError(
            f"Requested {config.devices} GPUs, but only "
            f"{torch.cuda.device_count()} are visible"
        )

    train_loader, val_loader, test_loader = create_dataloaders(config)
    model = ModelInterface(config.to_dict())
    strategy = config.strategy
    if config.devices > 1 and config.strategy == 'ddp':
        from pytorch_lightning.strategies import DDPStrategy

        strategy = DDPStrategy(
            find_unused_parameters=config.MULTI_GPU.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=False,
        )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=config.devices,
        max_epochs=config.TRAINING.num_epochs,
        logger=load_loggers(config),
        callbacks=load_callbacks(config),
        precision=config.precision,
        strategy=strategy,
        sync_batchnorm=config.sync_batchnorm,
        accumulate_grad_batches=config.MULTI_GPU.accumulate_grad_batches,
        enable_progress_bar=True,
        log_every_n_steps=50,
        gradient_clip_val=config.TRAINING.gradient_clip_val,
        deterministic=config.deterministic,
        enable_model_summary=False,
    )

    if config.mode == 'train':
        trainer.fit(model, train_loader, val_loader)
    else:
        trainer.test(model, test_loader, ckpt_path=config.ckpt_path)
    return model


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    runtime_config = build_config_from_args(get_parse().parse_args())
    main(runtime_config)
