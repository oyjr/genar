import logging
import os
import random
import numpy as np
from scipy import sparse
import torch
import scanpy as sc
import anndata as ad
from torch.utils.data import Dataset
from typing import List, Dict, Optional

from configs import ENCODER_FEATURE_DIMS

logger = logging.getLogger(__name__)


class STDataset(Dataset):
    """Spatial transcriptomics dataset wrapper."""

    def __init__(self,
                 mode: str,                    # 'train', 'val', 'test'
                 data_path: str,               # Dataset root
                 expr_name: str,               # Dataset name
                 slide_val: str = '',          # Validation slides
                 slide_test: str = '',         # Test slides
                 encoder_name: str = 'uni',    # Feature encoder
                 max_gene_count: int = 2000,   # Max gene count cap
                 prediction_mode: str = 'discrete',
                 library_scale: float = 10000.0,
                 grouping_mode: str = 'kmeans',
                 grouping_seed: int = 42):
        """
        Args:
            mode: Dataset split ('train', 'val', 'test').
            data_path: Root directory for the dataset.
            expr_name: Dataset identifier.
            slide_val: Comma-separated slide IDs for validation.
            slide_test: Comma-separated slide IDs for testing.
            encoder_name: Encoder type ('uni', 'conch', 'resnet18').
            max_gene_count: Maximum gene count used for tokenisation.
            prediction_mode: Discrete paper model or continuous regression ablation.
            library_scale: Library-size normalization constant in continuous mode.
            grouping_mode: Curated k-means order or deterministic random ablation.
            grouping_seed: Seed used by the random grouping ablation.
        """
        super().__init__()
        
        # Validate constructor arguments
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"mode must be one of ['train', 'val', 'test'], got {mode}")
        if encoder_name not in ['uni', 'conch', 'resnet18']:
            raise ValueError(
                "encoder_name must be one of ['uni', 'conch', 'resnet18'], "
                f"got {encoder_name}"
            )
        if prediction_mode not in {'discrete', 'continuous'}:
            raise ValueError(
                "prediction_mode must be 'discrete' or 'continuous'"
            )
        if grouping_mode not in {'kmeans', 'random'}:
            raise ValueError("grouping_mode must be 'kmeans' or 'random'")
        if max_gene_count < 1:
            raise ValueError("max_gene_count must be positive")
        if library_scale <= 0:
            raise ValueError("library_scale must be positive")
        
        self.mode = mode
        self.data_path = data_path
        self.expr_name = expr_name
        self.encoder_name = encoder_name
        self.max_gene_count = max_gene_count
        self.prediction_mode = prediction_mode
        self.library_scale = float(library_scale)
        self.grouping_mode = grouping_mode
        self.grouping_seed = int(grouping_seed)
        
        # Directory layout
        self.st_dir = os.path.join(data_path, 'st')
        self.processed_dir = os.path.join(data_path, 'processed_data')

        # Embedding directory
        self.emb_dir = os.path.join(self.processed_dir, f"spot_features_{encoder_name}")

        logger.debug("Initialising STDataset: mode=%s, dataset=%s, encoder=%s", mode, expr_name, encoder_name)

        # Load top-200 genes
        self.genes = self._load_gene_list()
        self._apply_grouping_strategy()

        # Slide splits
        self.slide_splits = self._load_slide_splits(slide_val, slide_test)
        self.ids = self.slide_splits[mode]
        if not self.ids:
            raise ValueError(f"No slides are assigned to the '{mode}' split")
        self.int2id = dict(enumerate(self.ids))

        logger.debug("Loaded %d genes across %d slides", len(self.genes), len(self.ids))

        # Cache for evaluation datasets to avoid repeated disk I/O
        self.eval_adata_cache: Dict[str, ad.AnnData] = {}
        self.slide_lengths: Dict[str, int] = {}
        # Each DataLoader worker gets its own process-local cache.  This avoids
        # re-reading an entire slide tensor for every spot.
        self.emb_cache: Dict[str, torch.Tensor] = {}

        # Preload training data when needed
        if mode == 'train':
            self._init_train_mode()

    def _load_gene_list(self) -> List[str]:
        """Load the first 200 genes from the selection list."""
        gene_file = f"{self.processed_dir}/selected_gene_list.txt"
        
        if not os.path.exists(gene_file):
            raise FileNotFoundError(f"Gene list not found: {gene_file}")
        
        with open(gene_file, 'r', encoding='utf-8') as f:
            all_genes = [line.strip() for line in f.readlines() if line.strip()]
        
        if len(all_genes) < 200:
            raise ValueError(f"Dataset contains only {len(all_genes)} genes; expected at least 200")

        selected_genes = all_genes[:200]
        if len(selected_genes) != len(set(selected_genes)):
            raise ValueError("The first 200 selected genes contain duplicates")
        return selected_genes

    def _apply_grouping_strategy(self) -> None:
        """Apply the deterministic random-grouping ablation when requested."""
        if self.grouping_mode == 'random':
            generator = random.Random(self.grouping_seed)
            generator.shuffle(self.genes)

    def _load_slide_splits(self, slide_val: str, slide_test: str) -> Dict[str, List[str]]:
        """Load slide list and split into train/val/test."""
        slide_file = f"{self.processed_dir}/all_slide_lst.txt"

        if not os.path.exists(slide_file):
            raise FileNotFoundError(f"Slide list missing: {slide_file}")
        
        with open(slide_file, 'r', encoding='utf-8') as f:
            all_slides = [line.strip() for line in f.readlines() if line.strip()]
        if not all_slides:
            raise ValueError(f"Slide list is empty: {slide_file}")
        if len(all_slides) != len(set(all_slides)):
            raise ValueError("Slide list contains duplicate IDs")
        
        # Parse validation/test splits
        val_slides = [s.strip() for s in slide_val.split(',') if s.strip()] if slide_val else []
        test_slides = [s.strip() for s in slide_test.split(',') if s.strip()] if slide_test else []
        if len(val_slides) != len(set(val_slides)):
            raise ValueError("Validation slide list contains duplicates")
        if len(test_slides) != len(set(test_slides)):
            raise ValueError("Test slide list contains duplicates")

        # Validate slide IDs
        all_slides_set = set(all_slides)
        for slide in val_slides + test_slides:
            if slide not in all_slides_set:
                raise ValueError(f"Unknown slide ID: {slide}")

        # Remaining slides become training set
        train_slides = [s for s in all_slides if s not in val_slides and s not in test_slides]
        
        return {
            'train': train_slides,
            'val': val_slides,
            'test': test_slides
        }

    def _init_train_mode(self):
        """Preload training data."""
        # Preload AnnData objects for every training slide
        self.adata_dict = {}
        lengths = []

        for slide_id in self.ids:
            adata = self._load_st(slide_id)
            self.adata_dict[slide_id] = adata
            lengths.append(len(adata))

        self.cumlen = np.cumsum(lengths)
        logger.debug("Training mode dataset size: %d", self.cumlen[-1])

    def _load_emb(self, slide_id: str, idx: Optional[int] = None) -> torch.Tensor:
        """Load embeddings for a slide."""
        emb_file = f"{self.emb_dir}/{slide_id}_{self.encoder_name}.pt"

        if not os.path.exists(emb_file):
            raise FileNotFoundError(f"Embedding file not found: {emb_file}")

        if emb_file not in self.emb_cache:
            features = torch.load(
                emb_file,
                map_location='cpu',
                weights_only=True,
            )
            if not torch.is_tensor(features) or features.ndim != 2:
                raise ValueError(
                    f"Expected a 2D tensor in {emb_file}, got "
                    f"{type(features).__name__} with shape "
                    f"{getattr(features, 'shape', None)}"
                )
            expected_dim = ENCODER_FEATURE_DIMS[self.encoder_name]
            if features.shape[1] != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch for {slide_id}: "
                    f"expected {expected_dim}, got {features.shape[1]}"
                )
            if not torch.isfinite(features).all():
                raise ValueError(
                    f"Embedding tensor contains NaN or Inf: {emb_file}"
                )
            features = features.float()
            expected_rows = self.slide_lengths.get(slide_id)
            if (
                expected_rows is not None
                and features.shape[0] != expected_rows
            ):
                raise ValueError(
                    f"Spot/embedding row mismatch for {slide_id}: "
                    f"AnnData={expected_rows}, embeddings={features.shape[0]}"
                )
            self.emb_cache[emb_file] = features
        features = self.emb_cache[emb_file]

        # Standard 2D format
        if idx is not None:
            return features[idx]
        else:
            return features

    def _load_st(self, slide_id: str) -> ad.AnnData:
        """Load the ST AnnData object for a slide and subset genes."""
        st_file = f"{self.st_dir}/{slide_id}.h5ad"
        
        if not os.path.exists(st_file):
            raise FileNotFoundError(f"ST file not found: {st_file}")

        adata = sc.read_h5ad(st_file)
        
        missing_genes = [gene for gene in self.genes if gene not in adata.var_names]
        if missing_genes:
            preview = ', '.join(missing_genes[:5])
            raise ValueError(
                f"Slide {slide_id} is missing {len(missing_genes)} selected genes "
                f"(first: {preview})"
            )

        # Select the curated gene list in its exact hierarchy order.
        adata = adata[:, self.genes].copy()
        
        # Normalize the active coordinate field per slide.
        if 'spatial' in adata.obsm:
            coords = np.asarray(adata.obsm['spatial'], dtype=np.float32)
        elif 'positions' in adata.obsm:
            coords = np.asarray(adata.obsm['positions'], dtype=np.float32)
        else:
            raise ValueError(f"Missing spatial/positions coordinates for slide {slide_id}")
        if (
            coords.ndim != 2
            or coords.shape != (adata.n_obs, 2)
            or not np.isfinite(coords).all()
        ):
            raise ValueError(
                f"Invalid spatial coordinates for slide {slide_id}: "
                f"shape={coords.shape}"
            )
        coord_min = coords.min(axis=0)
        denom = coords.max(axis=0) - coord_min
        if np.any(denom == 0):
            raise ValueError(f"Zero-range spatial coordinates for slide {slide_id}")
        adata.obsm['positions'] = (coords - coord_min) / denom
        self.slide_lengths[slide_id] = int(adata.n_obs)
        return adata

    def _get_cached_eval_adata(self, slide_id: str) -> ad.AnnData:
        """Lazy-load evaluation AnnData objects with caching."""
        if slide_id not in self.eval_adata_cache:
            self.eval_adata_cache[slide_id] = self._load_st(slide_id)
        return self.eval_adata_cache[slide_id]

    def __len__(self) -> int:
        if self.mode == 'train':
            return self.cumlen[-1] if len(self.cumlen) > 0 else 0
        else:
            # Cache total spots during evaluation/testing
            if not hasattr(self, 'total_spots'):
                self.total_spots = sum(len(self._get_cached_eval_adata(slide_id)) for slide_id in self.ids)
            return self.total_spots

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Retrieve a sample for the requested split."""
        if self.mode == 'train':
            return self._get_train_item(index)
        else:
            return self._get_eval_item(index)

    def _get_train_item(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a training sample."""
        # Locate the slide and sample index
        i = int(np.searchsorted(self.cumlen, index, side='right'))
        
        sample_idx = index - (self.cumlen[i-1] if i > 0 else 0)
        slide_id = self.int2id[i]
        
        features = self._load_emb(slide_id, sample_idx)
        adata = self.adata_dict[slide_id]
        expression = adata[sample_idx].X
        positions = adata.obsm['positions'][sample_idx]
        
        return {
            'img': features,
            'target_genes': self._prepare_targets(expression),
            'raw_target_genes': self._raw_gene_expression(expression),
            'positions': torch.FloatTensor(positions),
            'slide_id': slide_id,
            'spot_idx': sample_idx
        }

    def _get_eval_item(self, index: int) -> Dict[str, torch.Tensor]:
        """Return a validation/test sample."""
        # Lazily build cumulative lengths
        if not hasattr(self, 'eval_cumlen'):
            lengths = [len(self._get_cached_eval_adata(slide_id)) for slide_id in self.ids]
            self.eval_cumlen = np.cumsum(lengths)
        
        # Locate slide/sample index
        i = int(np.searchsorted(self.eval_cumlen, index, side='right'))
        
        sample_idx = index - (self.eval_cumlen[i-1] if i > 0 else 0)
        slide_id = self.int2id[i]
        
        # Load embeddings and expression
        features = self._load_emb(slide_id, sample_idx)
        adata = self._get_cached_eval_adata(slide_id)
        expression = adata[sample_idx].X
        positions = adata.obsm['positions'][sample_idx]
        
        return {
            'img': torch.FloatTensor(features),
            'target_genes': self._prepare_targets(expression),
            'raw_target_genes': self._raw_gene_expression(expression),
            'positions': torch.FloatTensor(positions),
            'slide_id': slide_id,
            'spot_idx': sample_idx
        }

    def get_full_slide_for_testing(self, slide_id: str) -> Dict[str, torch.Tensor]:
        """Return full-slide tensors for evaluation utilities."""
        adata = self._get_cached_eval_adata(slide_id)
        features = self._load_emb(slide_id, None)  # [num_spots, feature_dim]
        
        expression = adata.X
        if sparse.issparse(expression):
            expression = expression.toarray()
        
        positions = adata.obsm['positions']
        
        return {
            'img': torch.FloatTensor(features),
            'target_genes': self._prepare_targets(expression),
            'raw_target_genes': self._raw_gene_expression(expression),
            'positions': torch.FloatTensor(positions),
            'slide_id': slide_id,
            'num_spots': adata.n_obs,
            'adata': adata
        }

    def get_test_slide_ids(self) -> List[str]:
        """List slide IDs used for testing."""
        return self.slide_splits['test'] if self.mode != 'test' else self.ids

    def _process_gene_expression(self, gene_expr) -> torch.Tensor:
        """Convert gene expression counts into integer tokens."""
        gene_expr = self._raw_gene_expression(gene_expr).numpy()
        gene_expr = np.round(gene_expr).astype(np.int64)
        return torch.clamp(
            torch.from_numpy(gene_expr).long(),
            0,
            self.max_gene_count,
        )

    def _raw_gene_expression(self, gene_expr) -> torch.Tensor:
        """Return non-negative raw selected-gene counts without count-cap clipping."""
        if sparse.issparse(gene_expr):
            gene_expr = gene_expr.toarray().squeeze()
        else:
            gene_expr = np.asarray(gene_expr).squeeze()

        gene_expr = gene_expr.astype(np.float64, copy=False)
        if not np.isfinite(gene_expr).all():
            raise ValueError("Gene expression contains NaN or infinite values")
        if np.any(gene_expr < -1.0e-6):
            raise ValueError("Raw count input contains negative values")
        gene_expr = np.maximum(0, gene_expr)
        rounded = np.rint(gene_expr)
        if not np.allclose(gene_expr, rounded, atol=1.0e-4, rtol=0):
            raise ValueError(
                "GenAR requires raw integer counts; non-integer expression "
                "values were detected"
            )
        return torch.from_numpy(rounded.astype(np.float32))

    def _prepare_targets(self, gene_expr) -> torch.Tensor:
        """Create discrete count tokens or log-normalized continuous targets."""
        if self.prediction_mode == 'continuous':
            return self._prepare_continuous_expression(gene_expr)
        return self._process_gene_expression(gene_expr)

    def _prepare_continuous_expression(self, gene_expr) -> torch.Tensor:
        """Library-size normalize to 10,000 (default), then apply natural log1p."""
        counts = self._raw_gene_expression(gene_expr).numpy()
        library_size = float(counts.sum())
        if library_size <= 0:
            return torch.zeros_like(torch.from_numpy(counts))
        normalized = counts / library_size * self.library_scale
        return torch.from_numpy(np.log1p(normalized).astype(np.float32))
