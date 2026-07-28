#!/usr/bin/env python3
"""Check the environment and input files before training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

from configs import (  # noqa: E402
    DATASETS,
    ENCODER_FEATURE_DIMS,
    PAPER_BATCH_SIZE,
    PAPER_MAX_GENE_COUNT,
    PAPER_NUM_GENES,
    PAPER_SCALE_DIMS,
)
from safe_io import require_safe_torch_load  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataset', required=True, choices=list(DATASETS))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--encoder', choices=list(ENCODER_FEATURE_DIMS), default='uni')
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument(
        '--global-batch-size',
        type=int,
        default=PAPER_BATCH_SIZE,
    )
    parser.add_argument(
        '--max-gene-count',
        type=int,
        default=PAPER_MAX_GENE_COUNT,
    )
    parser.add_argument('--require-h100', action='store_true')
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Check input files without requiring CUDA',
    )
    return parser.parse_args()


def read_nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    values = [
        line.strip()
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"Required file is empty: {path}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: list[str]) -> str:
    return hashlib.sha256(
        ('\n'.join(values) + '\n').encode('utf-8')
    ).hexdigest()


def package_versions() -> dict[str, str]:
    requirements_path = REPO_ROOT / 'requirements.txt'
    packages = {}
    for line in requirements_path.read_text(encoding='utf-8').splitlines():
        value = line.strip()
        if not value or value.startswith('#'):
            continue
        if value.count('==') != 1:
            raise RuntimeError(
                f"Requirement must be exactly pinned: {value}"
            )
        package, expected = value.split('==')
        packages[package] = expected

    result = {}
    for package, expected in packages.items():
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required package is not installed: {package}"
            ) from exc
        if installed.split('+', 1)[0] != expected:
            raise RuntimeError(
                f"Package version mismatch for {package}: "
                f"expected {expected}, found {installed}"
            )
        result[package] = installed
    return result


def main() -> int:
    args = parse_args()
    require_safe_torch_load()
    if args.gpus < 1:
        raise ValueError("--gpus must be positive")
    if args.global_batch_size != PAPER_BATCH_SIZE:
        raise ValueError(
            f"Paper launcher requires global batch size {PAPER_BATCH_SIZE}"
        )
    if args.global_batch_size % args.gpus:
        raise ValueError(
            f"Global batch size {args.global_batch_size} is not divisible by "
            f"{args.gpus} GPU processes"
        )
    batch_size_per_process = args.global_batch_size // args.gpus
    if args.max_gene_count != PAPER_MAX_GENE_COUNT:
        raise ValueError(
            f"Paper launcher requires count cap {PAPER_MAX_GENE_COUNT}"
        )

    dataset_info = DATASETS[args.dataset]
    dataset_root = (
        args.data_root.expanduser().resolve()
        / dataset_info['dir_name']
    )
    processed = dataset_root / 'processed_data'
    st_dir = dataset_root / 'st'
    embedding_dir = processed / f'spot_features_{args.encoder}'
    for directory in (dataset_root, processed, st_dir, embedding_dir):
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Required directory is missing: {directory}"
            )

    gene_file = processed / 'selected_gene_list.txt'
    genes = read_nonempty_lines(gene_file)
    if len(genes) < PAPER_NUM_GENES:
        raise ValueError(
            f"Expected at least {PAPER_NUM_GENES} selected genes, got "
            f"{len(genes)}"
        )
    selected = genes[:PAPER_NUM_GENES]
    if len(selected) != len(set(selected)):
        raise ValueError("The first 200 selected genes contain duplicates")

    slide_file = processed / 'all_slide_lst.txt'
    slides = read_nonempty_lines(slide_file)
    if len(slides) != len(set(slides)):
        raise ValueError("Slide list contains duplicates")
    held_out = {
        slide.strip()
        for value in (
            dataset_info['val_slides'],
            dataset_info['test_slides'],
        )
        for slide in value.split(',')
        if slide.strip()
    }
    missing_held_out = held_out.difference(slides)
    if missing_held_out:
        raise ValueError(
            "Configured held-out slides are absent: "
            + ', '.join(sorted(missing_held_out))
        )
    if not set(slides).difference(held_out):
        raise ValueError("No training slides remain after held-out exclusion")
    expected_training = [
        slide
        for slide in slides
        if slide not in held_out
    ]

    source_gene_file = processed / 'unclustered_selected_gene_list.txt'
    source_genes = read_nonempty_lines(source_gene_file)
    if len(source_genes) < PAPER_NUM_GENES:
        raise ValueError(
            f"Expected at least {PAPER_NUM_GENES} unclustered genes, got "
            f"{len(source_genes)}"
        )
    source_selected = source_genes[:PAPER_NUM_GENES]
    if len(source_selected) != len(set(source_selected)):
        raise ValueError("The unclustered selected-gene list has duplicates")

    clustering_file = processed / 'clustering_info.json'
    if not clustering_file.is_file():
        raise FileNotFoundError(
            "Missing clustering_info.json; run the paper gene-hierarchy "
            "preprocessing before training"
        )
    with clustering_file.open('r', encoding='utf-8') as handle:
        clustering = json.load(handle)
    if not isinstance(clustering, dict):
        raise ValueError("clustering_info.json must contain an object")
    expected_clustering_fields = {
        'dataset',
        'train_slides',
        'clustered_order',
        'scale_dims',
        'algorithm',
        'selected_gene_count',
        'source_gene_list_sha256',
        'output_gene_list_sha256',
        'excluded_validation_test_slides',
    }
    missing_clustering_fields = expected_clustering_fields.difference(
        clustering
    )
    if missing_clustering_fields:
        raise ValueError(
            "clustering_info.json is missing: "
            + ', '.join(sorted(missing_clustering_fields))
        )
    if clustering['dataset'] != args.dataset:
        raise ValueError("Clustering metadata dataset does not match")
    if list(clustering['train_slides']) != expected_training:
        raise ValueError(
            "Clustering metadata training split does not match all_slide_lst"
        )
    if sorted(clustering['excluded_validation_test_slides']) != sorted(
        held_out
    ):
        raise ValueError(
            "Clustering metadata held-out split does not match configuration"
        )
    if tuple(clustering['scale_dims']) != PAPER_SCALE_DIMS:
        raise ValueError("Clustering metadata does not use the paper hierarchy")
    if clustering['algorithm'] != 'kmeans_hierarchical':
        raise ValueError("Unexpected clustering algorithm")
    if int(clustering['selected_gene_count']) != PAPER_NUM_GENES:
        raise ValueError("Clustering metadata gene count does not equal 200")
    clustered_order = [int(index) for index in clustering['clustered_order']]
    if sorted(clustered_order) != list(range(PAPER_NUM_GENES)):
        raise ValueError("Clustering order is not a complete 200-gene permutation")
    if clustering['source_gene_list_sha256'] != sha256_lines(source_selected):
        raise ValueError("Unclustered selected-gene hash does not match metadata")
    if clustering['output_gene_list_sha256'] != sha256_lines(selected):
        raise ValueError("Clustered selected-gene hash does not match metadata")

    missing_inputs = []
    for slide in slides:
        if not (st_dir / f'{slide}.h5ad').is_file():
            missing_inputs.append(str(st_dir / f'{slide}.h5ad'))
        embedding = embedding_dir / f'{slide}_{args.encoder}.pt'
        if not embedding.is_file():
            missing_inputs.append(str(embedding))
    if missing_inputs:
        preview = '\n'.join(missing_inputs[:10])
        raise FileNotFoundError(
            f"{len(missing_inputs)} slide inputs are missing; first entries:\n"
            f"{preview}"
        )

    visible_gpus = torch.cuda.device_count()
    gpu_inventory = []
    if not args.data_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if visible_gpus < args.gpus:
            raise RuntimeError(
                f"Requested {args.gpus} GPUs, only {visible_gpus} are visible"
            )
        for index in range(args.gpus):
            properties = torch.cuda.get_device_properties(index)
            memory_gib = properties.total_memory / 1024 ** 3
            if args.require_h100 and (
                'H100' not in properties.name or memory_gib < 75
            ):
                raise RuntimeError(
                    f"GPU {index} is not an 80-GB H100: "
                    f"{properties.name}, {memory_gib:.1f} GiB"
                )
            gpu_inventory.append(
                {
                    'index': index,
                    'name': properties.name,
                    'memory_gib': round(memory_gib, 2),
                    'compute_capability': (
                        f"{properties.major}.{properties.minor}"
                    ),
                }
            )

    report = {
        'status': 'PASS',
        'dataset': args.dataset,
        'dataset_directory': dataset_info['dir_name'],
        'encoder': args.encoder,
        'encoder_feature_dim': ENCODER_FEATURE_DIMS[args.encoder],
        'slide_count': len(slides),
        'held_out_slides': sorted(held_out),
        'selected_gene_count': PAPER_NUM_GENES,
        'selected_gene_order_sha256': sha256_lines(selected),
        'selected_gene_list_file_sha256': sha256(gene_file),
        'clustering_info_sha256': sha256(clustering_file),
        'paper_scale_dims': list(PAPER_SCALE_DIMS),
        'global_batch_size': args.global_batch_size,
        'batch_size_per_process': batch_size_per_process,
        'max_gene_count': args.max_gene_count,
        'vocab_size_including_zero': args.max_gene_count + 1,
        'cuda_runtime': torch.version.cuda,
        'hardware_check': 'skipped' if args.data_only else 'passed',
        'visible_gpu_count': visible_gpus,
        'selected_gpus': gpu_inventory,
        'package_versions': package_versions(),
        'hf_hub_offline': os.environ.get('HF_HUB_OFFLINE'),
        'transformers_offline': os.environ.get('TRANSFORMERS_OFFLINE'),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
