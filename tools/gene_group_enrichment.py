#!/usr/bin/env python3
"""Compute nominal hypergeometric enrichment for learned gene groups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import hypergeom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--groups',
        type=Path,
        required=True,
        help='TSV columns: group, gene',
    )
    parser.add_argument(
        '--reference',
        type=Path,
        required=True,
        help='TSV columns: term, gene',
    )
    parser.add_argument('--background-size', type=int, default=20000)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def read_mapping(
    path: Path,
    key_column: str,
) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    with path.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        required = {key_column, 'gene'}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns {sorted(required)}"
            )
        for row in reader:
            key = row[key_column].strip()
            gene = row['gene'].strip()
            if key and gene:
                mapping.setdefault(key, set()).add(gene)
    if not mapping:
        raise ValueError(f"No mappings found in {path}")
    return mapping


def main() -> int:
    args = parse_args()
    if args.background_size < 1:
        raise ValueError("--background-size must be positive")
    groups = read_mapping(args.groups, 'group')
    reference = read_mapping(args.reference, 'term')
    all_genes = set().union(*groups.values(), *reference.values())
    if len(all_genes) > args.background_size:
        raise ValueError(
            f"Observed {len(all_genes)} genes, exceeding background "
            f"{args.background_size}"
        )

    rows = []
    for group_name, group_genes in sorted(groups.items()):
        for term, term_genes in sorted(reference.items()):
            overlap = group_genes & term_genes
            p_value = hypergeom.sf(
                len(overlap) - 1,
                args.background_size,
                len(term_genes),
                len(group_genes),
            )
            rows.append(
                {
                    'group': group_name,
                    'term': term,
                    'group_size': len(group_genes),
                    'reference_size': len(term_genes),
                    'overlap_size': len(overlap),
                    'overlap_genes': ','.join(sorted(overlap)),
                    'nominal_p_value': float(p_value),
                    'multiple_testing_adjusted': False,
                    'background_size': args.background_size,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        'schema_version': 1,
        'output': str(args.output.resolve()),
        'test': 'one-sided hypergeometric survival function',
        'p_values': 'nominal; not adjusted for multiple testing',
        'background_size': args.background_size,
        'comparison_count': len(rows),
    }
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
