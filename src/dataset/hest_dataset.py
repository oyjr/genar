import logging
import os
import pandas as pd
import numpy as np
from scipy import sparse
import torch
import scanpy as sc
import anndata as ad
from torch.utils.data import Dataset
from typing import List, Dict, Optional

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
                 use_augmented: bool = False,  # Use augmented features
                 expand_augmented: bool = False,  # Expand augmented samples
                 max_gene_count: int = 500):   # Max gene count cap
        """
        Args:
            mode: Dataset split ('train', 'val', 'test').
            data_path: Root directory for the dataset.
            expr_name: Dataset identifier.
            slide_val: Comma-separated slide IDs for validation.
            slide_test: Comma-separated slide IDs for testing.
            encoder_name: Encoder type ('uni', 'conch', 'resnet18').
            use_augmented: Whether augmented embeddings are available.
            expand_augmented: Expand augmented spots into individual rows.
            max_gene_count: Maximum gene count used for tokenisation.
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
        
        # Only expand augmented data during training when augmentation is enabled
        if expand_augmented and (not use_augmented or mode != 'train'):
            expand_augmented = False
        
        self.mode = mode
        self.data_path = data_path
        self.expr_name = expr_name
        self.encoder_name = encoder_name
        self.use_augmented = use_augmented
        self.expand_augmented = expand_augmented
        self.max_gene_count = max_gene_count
        
        # Directory layout
        self.st_dir = os.path.join(data_path, 'st')
        self.processed_dir = os.path.join(data_path, 'processed_data')

        # Embedding directory
        emb_suffix = "_aug" if use_augmented else ""
        self.emb_dir = os.path.join(self.processed_dir, f"spot_features_{encoder_name}{emb_suffix}")

        logger.debug("Initialising STDataset: mode=%s, dataset=%s, encoder=%s", mode, expr_name, encoder_name)

        # Load top-200 genes
        self.genes = self._load_gene_list()

        # Slide splits
        self.slide_splits = self._load_slide_splits(slide_val, slide_test)
        self.ids = self.slide_splits[mode]
        self.int2id = dict(enumerate(self.ids))

        logger.debug("Loaded %d genes across %d slides", len(self.genes), len(self.ids))

        # Cache for evaluation datasets to avoid repeated disk I/O
        self.eval_adata_cache: Dict[str, ad.AnnData] = {}

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

        return all_genes[:200]

    def _load_slide_splits(self, slide_val: str, slide_test: str) -> Dict[str, List[str]]:
        """Load slide list and split into train/val/test."""
        slide_file = f"{self.processed_dir}/all_slide_lst.txt"

        if not os.path.exists(slide_file):
            raise FileNotFoundError(f"Slide list missing: {slide_file}")
        
        with open(slide_file, 'r', encoding='utf-8') as f:
            all_slides = [line.strip() for line in f.readlines() if line.strip()]
        
        # Parse validation/test splits
        val_slides = [s.strip() for s in slide_val.split(',') if s.strip()] if slide_val else []
        test_slides = [s.strip() for s in slide_test.split(',') if s.strip()] if slide_test else []

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
        """Preload training data and optional augmentations."""
        # Preload AnnData objects for every training slide
        self.adata_dict = {}
        lengths = []

        for slide_id in self.ids:
            adata = self._load_st(slide_id)
            self.adata_dict[slide_id] = adata
            
            if self.expand_augmented:
                # Augmented inputs expand each spot into seven variants
                lengths.append(len(adata) * 7)
                # Prepare expanded tensors and metadata
                self._prepare_expanded_data(slide_id, adata)
            else:
                lengths.append(len(adata))

        self.cumlen = np.cumsum(lengths)

        if self.expand_augmented:
            logger.debug("Training mode with augmentation expansion: %d samples", self.cumlen[-1])
        else:
            logger.debug("Training mode standard dataset size: %d", self.cumlen[-1])

    def _prepare_expanded_data(self, slide_id: str, adata: ad.AnnData):
        """Materialise the expanded augmented dataset."""
        if not hasattr(self, 'expanded_emb_dict'):
            self.expanded_emb_dict = {}
            self.expanded_adata_dict = {}

        # Load augmented embeddings (stored either as [spots,7,dim] or [spots*7,dim])
        emb = self._load_emb(slide_id, None, 'aug')

        # Normalise tensor shape
        if len(emb.shape) == 3:
            num_spots, num_augs, feature_dim = emb.shape
            expanded_emb = emb.reshape(-1, feature_dim)
        else:
            expanded_emb = emb

        self.expanded_emb_dict[slide_id] = expanded_emb

        # Expand AnnData metadata/expressions
        expanded_obs_data = []
        expanded_X_data = []
        expanded_positions = []
        
        for aug_idx in range(7):
            for spot_idx in range(len(adata)):
                expanded_obs_data.append({
                    'original_spot_id': spot_idx,
                    'aug_id': aug_idx
                })
                expanded_X_data.append(adata.X[spot_idx])
                expanded_positions.append(adata.obsm['positions'][spot_idx])
        
        # Build expanded AnnData object
        obs_df = pd.DataFrame(expanded_obs_data)
        obs_df.index = obs_df.index.astype(str)

        expanded_adata = ad.AnnData(
            X=sparse.vstack(expanded_X_data) if sparse.issparse(adata.X) else np.vstack(expanded_X_data),
            obs=obs_df,
            var=adata.var.copy()
        )
        expanded_adata.obsm['positions'] = np.array(expanded_positions)
        
        self.expanded_adata_dict[slide_id] = expanded_adata

    def _load_emb(self, slide_id: str, idx: Optional[int] = None, mode: str = 'standard') -> torch.Tensor:
        """Load embeddings for a slide in standard or augmented mode."""
        if mode == 'aug' and self.use_augmented:
            emb_file = f"{self.emb_dir}/{slide_id}_{self.encoder_name}_aug.pt"
        else:
            # Fall back to the standard embedding directory
            base_dir = self.emb_dir.replace('_aug', '')
            emb_file = f"{base_dir}/{slide_id}_{self.encoder_name}.pt"

        if not os.path.exists(emb_file):
            raise FileNotFoundError(f"Embedding file not found: {emb_file}")

        features = torch.load(emb_file, map_location='cpu', weights_only=True)

        # Support 3D augmented tensors
        if mode == 'aug' and len(features.shape) == 3:
            if idx is not None:
                return features[idx, 0, :]
            else:
                return features

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
        
        # Select the curated gene list
        adata = adata[:, self.genes].copy()
        
        # Normalise positional coordinates if available
        if 'spatial' in adata.obsm:
            coords = adata.obsm['spatial'].copy()
            coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))
            adata.obsm['positions'] = coords
        elif 'positions' not in adata.obsm:
            # Fallback to random coordinates when missing
            import numpy as np
            adata.obsm['positions'] = np.random.rand(adata.n_obs, 2)
        
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
        """Return a training sample (with optional augmentation)."""
        # Locate the slide and sample index
        i = 0
        while index >= self.cumlen[i]:
            i += 1
        
        sample_idx = index - (self.cumlen[i-1] if i > 0 else 0)
        slide_id = self.int2id[i]
        
        if self.expand_augmented:
            # Use pre-expanded augmented tensors
            features = self.expanded_emb_dict[slide_id][sample_idx]
            expanded_adata = self.expanded_adata_dict[slide_id]
            expression = expanded_adata[sample_idx].X
            positions = expanded_adata.obsm['positions'][sample_idx]
            
            # Augmentation provenance
            original_spot_id = int(expanded_adata.obs['original_spot_id'].iloc[sample_idx])
            aug_id = int(expanded_adata.obs['aug_id'].iloc[sample_idx])
            
            return {
                'img': torch.FloatTensor(features),
                'target_genes': self._process_gene_expression(expression),
                'positions': torch.FloatTensor(positions),
                'slide_id': slide_id,
                'spot_idx': sample_idx,
                'original_spot_id': original_spot_id,
                'aug_id': aug_id
            }
        else:
            # Standard path
            features = self._load_emb(slide_id, sample_idx, 'standard')
            adata = self.adata_dict[slide_id]
            expression = adata[sample_idx].X
            positions = adata.obsm['positions'][sample_idx]
            
            return {
                'img': features,
                'target_genes': self._process_gene_expression(expression),
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
        i = 0
        while index >= self.eval_cumlen[i]:
            i += 1
        
        sample_idx = index - (self.eval_cumlen[i-1] if i > 0 else 0)
        slide_id = self.int2id[i]
        
        # Load embeddings and expression
        features = self._load_emb(slide_id, sample_idx, 'standard')
        adata = self._get_cached_eval_adata(slide_id)
        expression = adata[sample_idx].X
        positions = adata.obsm['positions'][sample_idx]
        
        return {
            'img': torch.FloatTensor(features),
            'target_genes': self._process_gene_expression(expression),
            'positions': torch.FloatTensor(positions),
            'slide_id': slide_id,
            'spot_idx': sample_idx
        }

    def get_full_slide_for_testing(self, slide_id: str) -> Dict[str, torch.Tensor]:
        """Return full-slide tensors for evaluation utilities."""
        features = self._load_emb(slide_id, None, 'standard')  # [num_spots, feature_dim]
        adata = self._get_cached_eval_adata(slide_id)
        
        expression = adata.X
        if sparse.issparse(expression):
            expression = expression.toarray()
        
        positions = adata.obsm['positions']
        
        return {
            'img': torch.FloatTensor(features),
            'target_genes': self._process_gene_expression(expression),
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
        if sparse.issparse(gene_expr):
            gene_expr = gene_expr.toarray().squeeze()
        else:
            gene_expr = np.asarray(gene_expr).squeeze()
        
        # Clamp to non-negative integers within the configured range
        gene_expr = np.maximum(0, gene_expr)
        gene_expr = np.round(gene_expr).astype(np.int64)
        gene_tokens = torch.clamp(torch.from_numpy(gene_expr).long(), 0, self.max_gene_count)
        
        return gene_tokens
