"""
Gene clustering and reordering based on expression similarity

This module implements gene clustering to reorder genes based on their 
spatial expression patterns for better biological coherence in GenAR models.

"""

import os
import json
import hashlib
import numpy as np
import shutil
from datetime import datetime
from sklearn.cluster import KMeans
from typing import List, Tuple, Optional
import logging

from configs import DATASETS, PAPER_NUM_GENES, PAPER_SCALE_DIMS
from .utils import (
    load_gene_list, save_gene_list, get_train_slides, 
    load_slide_gene_expression
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneClusteringProcessor:
    """Gene clustering helper used during preprocessing."""
    
    def __init__(
        self,
        scale_dims: Tuple[int, ...] = PAPER_SCALE_DIMS,
        data_root: Optional[str] = None,
        h5ad_root: Optional[str] = None,
        random_state: int = 42,
    ):
        """
        Args:
            scale_dims: Multi-scale configuration used by GenAR models
        """
        self.scale_dims = scale_dims
        self.data_root = os.path.abspath(data_root or os.environ.get('GENAR_DATA_ROOT', './data'))
        self.h5ad_root = h5ad_root or os.environ.get('GENAR_H5AD_ROOT')
        self.random_state = int(random_state)
        self.datasets = DATASETS
    
    def process_dataset(self, dataset_name: str) -> None:
        """Run the clustering pipeline for a single dataset."""

        if dataset_name not in self.datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        dataset_config = self.datasets[dataset_name]
        dataset_root = os.path.join(self.data_root, dataset_config['dir_name'])
        data_path = os.path.join(dataset_root, 'processed_data')
        excluded_slides = {
            slide.strip()
            for value in (
                dataset_config['val_slides'],
                dataset_config['test_slides'],
            )
            for slide in value.split(',')
            if slide.strip()
        }

        if not os.path.exists(dataset_root):
            raise FileNotFoundError(
                f"Dataset root does not exist: {dataset_root}"
            )
        if not os.path.isdir(data_path):
            raise FileNotFoundError(
                f"Processed-data directory does not exist: {data_path}"
            )
        
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"   data path: {data_path}")
        logger.info(
            "   excluded validation/test slides: %s",
            ', '.join(sorted(excluded_slides)),
        )

        # 1. Backup original gene list
        self._backup_original_gene_list(data_path)

        # 2. Fetch training slides
        train_slides = get_train_slides(data_path, excluded_slides)

        # 3. Load training gene expression
        backup_file = os.path.join(
            data_path,
            'unclustered_selected_gene_list.txt',
        )
        source_genes = load_gene_list(backup_file)[:PAPER_NUM_GENES]
        if len(source_genes) != PAPER_NUM_GENES:
            raise ValueError(
                f"Expected {PAPER_NUM_GENES} source genes, got "
                f"{len(source_genes)}"
            )
        if len(source_genes) != len(set(source_genes)):
            raise ValueError("The source selected-gene list contains duplicates")
        combined_expr = self._load_training_data(
            data_path,
            train_slides,
            source_genes,
        )
        logger.info(f"   Training spots: {combined_expr.shape[0]}")

        # 4. Run clustering
        clustered_order = self._perform_clustering(combined_expr)

        # 5. Save reordered gene list
        self._save_clustered_gene_list(data_path, clustered_order)

        # 6. Persist clustering metadata
        self._save_clustering_info(data_path, dataset_name, train_slides, 
                                  combined_expr.shape[0], clustered_order,
                                  source_genes, excluded_slides)

        logger.info(f"Clustering finished for {dataset_name}")

    def _backup_original_gene_list(self, data_path: str) -> None:
        """Backup the original gene list once."""
        original_file = os.path.join(data_path, 'selected_gene_list.txt')
        backup_file = os.path.join(data_path, 'unclustered_selected_gene_list.txt')
        
        if not os.path.exists(backup_file):
            shutil.copy(original_file, backup_file)
            logger.info(f"Created gene list backup: {backup_file}")
        else:
            logger.info(f"Backup already present: {backup_file}")

    def _load_training_data(
        self,
        data_path: str,
        train_slides: List[str],
        selected_genes: List[str],
    ) -> np.ndarray:
        """Load expression matrices for all training slides."""
        logger.info(f"Loading {len(train_slides)} training slides")
        
        all_expr_data = []
        
        for slide_id in train_slides:
            slide_expr = load_slide_gene_expression(
                data_path,
                slide_id,
                h5ad_root=self.h5ad_root,
                genes=selected_genes,
            )
            if slide_expr.shape[1] != len(selected_genes):
                raise ValueError(
                    f"{slide_id} returned {slide_expr.shape[1]} genes; "
                    f"expected {len(selected_genes)}"
                )
            if not np.isfinite(slide_expr).all():
                raise ValueError(
                    f"{slide_id} expression contains NaN or infinite values"
                )
            if (
                np.any(slide_expr < -1.0e-6)
                or not np.allclose(
                    slide_expr,
                    np.rint(slide_expr),
                    atol=1.0e-4,
                    rtol=0,
                )
            ):
                raise ValueError(
                    f"{slide_id} does not contain non-negative raw integer counts"
                )
            logger.info(
                "   %s: %d spots, %d selected genes",
                slide_id,
                slide_expr.shape[0],
                slide_expr.shape[1],
            )
            all_expr_data.append(slide_expr)

        if not all_expr_data:
            raise ValueError("No training data could be loaded")

        # Merge
        combined_expr = np.concatenate(all_expr_data, axis=0)
        logger.info(f"   Combined training matrix: {combined_expr.shape}")
        
        return combined_expr
    
    def _perform_clustering(self, gene_expr_matrix: np.ndarray) -> np.ndarray:
        """Run two-stage k-means clustering on genes."""
        logger.info("Starting gene clustering")

        # Build [n_genes, n_spots] feature matrix
        gene_features = gene_expr_matrix.T
        logger.info(f"   Gene feature matrix: {gene_features.shape}")

        gene_features_norm, constant_gene_count = (
            self._zscore_gene_profiles(gene_features)
        )
        if constant_gene_count:
            logger.warning(
                "%d selected genes are constant across training spots",
                constant_gene_count,
            )
        logger.info("   Standardisation complete")

        # Stage 1: four coarse clusters
        logger.info("   Stage 1: clustering into four groups")
        kmeans_4 = KMeans(
            n_clusters=4,
            random_state=self.random_state,
            n_init=10,
        )
        major_clusters = kmeans_4.fit_predict(gene_features_norm)

        cluster_sizes = np.bincount(major_clusters)
        logger.info(f"   Group sizes: {cluster_sizes}")

        # Stage 2: refine each group
        logger.info("   Stage 2: refining each group")
        clustered_order = []
        
        for major_group in range(4):
            genes_in_major = np.where(major_clusters == major_group)[0]
            group_features = gene_features_norm[genes_in_major]
            
            if len(genes_in_major) <= 10:
                # Small groups are added as-is
                clustered_order.extend(genes_in_major.tolist())
                logger.info(f"     Group {major_group}: {len(genes_in_major)} genes (no refinement)")
            else:
                # Further split the large group
                n_sub_clusters = max(2, len(genes_in_major) // 12)
                kmeans_sub = KMeans(
                    n_clusters=n_sub_clusters,
                    random_state=self.random_state,
                    n_init=10,
                )
                sub_clusters = kmeans_sub.fit_predict(group_features)
                
                # Append genes grouped by sub-cluster
                for sub_group in range(n_sub_clusters):
                    genes_in_sub = genes_in_major[sub_clusters == sub_group]
                    clustered_order.extend(genes_in_sub.tolist())
                
                logger.info(f"     Group {major_group}: {len(genes_in_major)} genes -> {n_sub_clusters} sub-clusters")

        clustered_order = np.array(clustered_order)
        if sorted(clustered_order.tolist()) != list(
            range(gene_expr_matrix.shape[1])
        ):
            raise ValueError(
                "Clustering did not produce a complete gene permutation"
            )
        logger.info(f"   Clustering reordered {len(clustered_order)} genes")
        
        return clustered_order

    @staticmethod
    def _zscore_gene_profiles(
        gene_features: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Z-score every gene row across spots, preserving constant rows as zero."""
        values = np.asarray(gene_features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("gene_features must be a non-empty 2D matrix")
        if not np.isfinite(values).all():
            raise ValueError("gene_features contains NaN or infinite values")
        means = values.mean(axis=1, keepdims=True)
        stds = values.std(axis=1, keepdims=True)
        constant_count = int(np.sum(stds[:, 0] == 0))
        safe_stds = np.where(stds > 0, stds, 1.0)
        normalized = (values - means) / safe_stds
        return normalized, constant_count
    
    def _save_clustered_gene_list(self, data_path: str, clustered_order: np.ndarray) -> None:
        """Write the reordered gene list back to disk."""
        # Load original list
        backup_file = os.path.join(data_path, 'unclustered_selected_gene_list.txt')
        original_genes = load_gene_list(backup_file)[:PAPER_NUM_GENES]
        if len(original_genes) != PAPER_NUM_GENES:
            raise ValueError(
                f"Expected {PAPER_NUM_GENES} genes in {backup_file}"
            )
        
        # Reorder according to clustering
        clustered_genes = [original_genes[i] for i in clustered_order]

        # Persist
        output_file = os.path.join(data_path, 'selected_gene_list.txt')
        save_gene_list(output_file, clustered_genes)

        logger.info(f"Saved clustered gene list to {output_file}")
    
    def _save_clustering_info(self, data_path: str, dataset_name: str,
                            train_slides: List[str], total_spots: int,
                            clustered_order: np.ndarray,
                            source_genes: List[str],
                            excluded_slides) -> None:
        """Persist metadata about the clustering run."""
        clustering_info = {
            'schema_version': 1,
            'dataset': dataset_name,
            'train_slides': train_slides,
            'total_spots': total_spots,
            'clustered_order': clustered_order.tolist(),
            'scale_dims': list(self.scale_dims),
            'timestamp': datetime.now().isoformat(),
            'algorithm': 'kmeans_hierarchical',
            'selected_gene_count': len(source_genes),
            'source_gene_list_sha256': hashlib.sha256(
                ('\n'.join(source_genes) + '\n').encode('utf-8')
            ).hexdigest(),
            'output_gene_list_sha256': hashlib.sha256(
                (
                    '\n'.join(
                        source_genes[index]
                        for index in clustered_order
                    )
                    + '\n'
                ).encode('utf-8')
            ).hexdigest(),
            'excluded_validation_test_slides': sorted(excluded_slides),
            'parameters': {
                'stage1_clusters': 4,
                'normalization': 'per_gene_zscore_across_training_spots',
                'random_state': self.random_state,
                'n_init': 10,
            }
        }
        
        info_file = os.path.join(data_path, 'clustering_info.json')
        with open(info_file, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(clustering_info, f, indent=2)
            f.write('\n')
        
        logger.info(f"Saved clustering metadata to {info_file}")

    def process_all_datasets(self) -> None:
        """Run clustering for every dataset in the registry."""
        failures = {}
        for dataset_name in self.datasets:
            try:
                self.process_dataset(dataset_name)
            except Exception as e:
                logger.error(f"Failed to process {dataset_name}: {e}")
                failures[dataset_name] = str(e)
        if failures:
            details = '; '.join(
                f"{name}: {message}"
                for name, message in failures.items()
            )
            raise RuntimeError(
                "One or more datasets failed; all-datasets preprocessing is "
                f"incomplete: {details}"
            )
