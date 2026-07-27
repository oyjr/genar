"""
Simple utility functions for gene clustering preprocessing

"""

import os
import numpy as np
import scanpy as sc
from typing import Iterable, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_slide_list(slide_list_file: str) -> List[str]:
    """Load slide identifiers from the list file."""
    with open(slide_list_file, 'r') as f:
        slides = [line.strip() for line in f.readlines() if line.strip()]
    return slides


def load_gene_list(gene_list_file: str) -> List[str]:
    """Load gene identifiers from disk."""
    with open(gene_list_file, 'r', encoding='utf-8') as f:
        genes = [line.strip() for line in f.readlines() if line.strip()]
    return genes


def save_gene_list(gene_list_file: str, genes: List[str]) -> None:
    """Persist gene identifiers to disk."""
    with open(gene_list_file, 'w', encoding='utf-8') as f:
        for gene in genes:
            f.write(f"{gene}\n")


def _resolve_h5ad_path(data_path: str, slide_id: str, h5ad_root: Optional[str]) -> str:
    """Resolve the path to a slide-level h5ad file."""
    if h5ad_root:
        return os.path.join(h5ad_root, f"{slide_id}.h5ad")

    candidates = [
        os.path.join(os.path.dirname(data_path), 'st', f"{slide_id}.h5ad"),
        os.path.join(data_path, 'adata', f"{slide_id}.h5ad"),
        os.path.join(data_path, 'processhest', 'adata', f"{slide_id}.h5ad"),
        os.path.join(os.path.dirname(data_path), 'processhest', 'adata', f"{slide_id}.h5ad"),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    formatted = "\n  - ".join(candidates)
    raise FileNotFoundError(
        "h5ad file not found. Set GENAR_H5AD_ROOT or place the file in one of:\n"
        f"  - {formatted}"
    )


def load_slide_gene_expression(
    data_path: str,
    slide_id: str,
    h5ad_root: Optional[str] = None,
    genes: Optional[List[str]] = None,
) -> np.ndarray:
    """Load the [n_spots, n_genes] expression matrix for a slide."""
    h5ad_file = _resolve_h5ad_path(data_path, slide_id, h5ad_root)
    
    adata = sc.read_h5ad(h5ad_file)
    
    if genes is not None:
        if len(genes) != len(set(genes)):
            raise ValueError("Selected gene list contains duplicate names")
        missing = [gene for gene in genes if gene not in adata.var_names]
        if missing:
            preview = ', '.join(missing[:5])
            raise ValueError(
                f"{slide_id} is missing {len(missing)} selected genes "
                f"(first: {preview})"
            )
        adata = adata[:, genes]

    # Fetch the expression matrix in the exact selected-gene order.
    gene_expr = adata.X
    if hasattr(gene_expr, 'toarray'):
        gene_expr = gene_expr.toarray()
    
    return gene_expr.astype(np.float32)


def get_train_slides(
    data_path: str,
    excluded_slides: Iterable[str],
) -> List[str]:
    """Return training slides after excluding every validation/test slide."""
    slide_list_file = os.path.join(data_path, 'all_slide_lst.txt')
    all_slides = load_slide_list(slide_list_file)
    excluded = {slide for slide in excluded_slides if slide}
    unknown = excluded.difference(all_slides)
    if unknown:
        raise ValueError(
            "Unknown validation/test slide IDs: " + ', '.join(sorted(unknown))
        )
    train_slides = [slide for slide in all_slides if slide not in excluded]
    if not train_slides:
        raise ValueError("No training slides remain after split exclusion")
    logger.info(
        "Training slides: %d (excluded %s)",
        len(train_slides),
        ', '.join(sorted(excluded)) or 'none',
    )
    return train_slides
