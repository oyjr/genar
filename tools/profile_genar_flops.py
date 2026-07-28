#!/usr/bin/env python3
"""Profile forward-only GenAR FLOPs from precomputed histology features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import profiler


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

from configs import (  # noqa: E402
    PAPER_MAX_GENE_COUNT,
    PAPER_NUM_GENES,
    SCALE_PRESETS,
)
from main import GENAR_CONFIG, parse_scale_dims  # noqa: E402
from model.genar.multiscale_genar import MultiScaleGenAR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Profile GenAR inference forward passes. Frozen histology encoder '
            'FLOPs are intentionally excluded, matching the paper protocol.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--histology-dim', type=int, default=1024)
    parser.add_argument('--embed-dim', type=int, default=512)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=8)
    parser.add_argument('--mlp-ratio', type=float, default=3.0)
    parser.add_argument(
        '--max-gene-count',
        type=int,
        default=PAPER_MAX_GENE_COUNT,
    )
    parser.add_argument(
        '--scale-config',
        choices=list(SCALE_PRESETS),
        default='paper',
    )
    parser.add_argument('--scale-dims', type=parse_scale_dims)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--warmup-steps', type=int, default=1)
    parser.add_argument('--profile-steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--json-output', type=Path)
    return parser.parse_args()


def build_model(
    args: argparse.Namespace,
    scales: Tuple[int, ...],
) -> MultiScaleGenAR:
    model = MultiScaleGenAR(
        vocab_size=args.max_gene_count + 1,
        num_genes=PAPER_NUM_GENES,
        scale_dims=scales,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        histology_feature_dim=args.histology_dim,
        spatial_coord_dim=2,
        condition_embed_dim=args.embed_dim,
        adaptive_sigma_alpha=GENAR_CONFIG['adaptive_sigma_alpha'],
        adaptive_sigma_beta=GENAR_CONFIG['adaptive_sigma_beta'],
        prediction_mode='discrete',
        final_loss_mode='gaussian_kl',
        scale_loss_weights=[1.0] * len(scales),
    )
    model.eval()
    return model


def run_forward(
    model: MultiScaleGenAR,
    histology: torch.Tensor,
    spatial: torch.Tensor,
) -> None:
    with torch.inference_mode():
        model(
            histology_features=histology,
            spatial_coords=spatial,
            target_genes=None,
            top_k=1,
        )


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.profile_steps < 1 or args.warmup_steps < 0:
        raise ValueError("Profile steps must be positive and warmup non-negative")
    if args.max_gene_count < 1:
        raise ValueError("--max-gene-count must be positive")
    if args.scale_dims is not None and args.scale_config != 'paper':
        raise ValueError(
            "Use either --scale-dims or a non-default --scale-config"
        )
    scales = (
        args.scale_dims
        if args.scale_dims is not None
        else SCALE_PRESETS[args.scale_config]
    )

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.manual_seed(args.seed)
    model = build_model(args, scales).to(device)
    histology = torch.randn(
        args.batch_size,
        args.histology_dim,
        device=device,
    )
    spatial = torch.randn(args.batch_size, 2, device=device)

    for _ in range(args.warmup_steps):
        run_forward(model, histology, spatial)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    activities = [
        profiler.ProfilerActivity.CUDA
        if device.type == 'cuda'
        else profiler.ProfilerActivity.CPU
    ]
    with profiler.profile(
        activities=activities,
        record_shapes=True,
        with_flops=True,
        profile_memory=False,
    ) as trace:
        for _ in range(args.profile_steps):
            run_forward(model, histology, spatial)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    total_flops = sum(
        float(event.flops or 0)
        for event in trace.key_averages()
    ) / args.profile_steps
    result = {
        'schema_version': 1,
        'scope': (
            'inference forward pass from precomputed histology features; '
            'frozen histology encoder excluded'
        ),
        'device': str(device),
        'batch_size': args.batch_size,
        'num_genes': PAPER_NUM_GENES,
        'max_gene_count': args.max_gene_count,
        'vocab_size_including_zero': args.max_gene_count + 1,
        'scale_dims': list(scales),
        'batch_flops': total_flops,
        'batch_gflops': total_flops / 1.0e9,
        'per_sample_flops': total_flops / args.batch_size,
        'per_sample_gflops': total_flops / args.batch_size / 1.0e9,
        'accounting_note': (
            "torch.profiler reports FLOPs only for operators whose formulas "
            "are implemented by the installed PyTorch build"
        ),
    }
    print(json.dumps(result, indent=2))
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open(
            'w',
            encoding='utf-8',
            newline='\n',
        ) as handle:
            json.dump(result, handle, indent=2)
            handle.write('\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
