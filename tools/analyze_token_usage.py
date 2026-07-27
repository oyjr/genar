#!/usr/bin/env python3
"""Summarize discrete count-token utilization from saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('predictions', type=Path, help='Inference .npz file')
    parser.add_argument('--field', default='predicted_counts')
    parser.add_argument('--count-cap', type=int, required=True)
    parser.add_argument('--json-output', type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count_cap < 1:
        raise ValueError("--count-cap must be positive")
    with np.load(args.predictions, allow_pickle=False) as archive:
        if args.field not in archive:
            raise KeyError(
                f"Field {args.field!r} is absent; available: "
                f"{list(archive.keys())}"
            )
        values = np.asarray(archive[args.field])
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Prediction array must be non-empty and finite")

    tokens = np.clip(
        np.rint(values).astype(np.int64),
        0,
        args.count_cap,
    )
    unique, counts = np.unique(tokens, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    result = {
        'schema_version': 1,
        'prediction_file': str(args.predictions.resolve()),
        'field': args.field,
        'count_cap': args.count_cap,
        'model_vocabulary_size_including_zero': args.count_cap + 1,
        'unique_tokens_used': int(unique.size),
        # The paper table reports unique/count_cap (e.g. 248/500), so retain
        # that convention while also exposing the literal vocabulary ratio.
        'paper_utilization_ratio': float(unique.size / args.count_cap),
        'literal_vocabulary_utilization_ratio': float(
            unique.size / (args.count_cap + 1)
        ),
        'global_entropy_nats': entropy,
        'minimum_token': int(unique.min()),
        'maximum_token': int(unique.max()),
        'total_predictions': int(tokens.size),
    }
    print(json.dumps(result, indent=2))
    if args.json_output:
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
