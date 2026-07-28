from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from addict import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from dataset.hest_dataset import STDataset
from inference import (
    _ordered_gene_hash,
    checkpoint_training_slides,
    collect_training_count_statistics,
    validate_dataset_contract,
)
from preprocess.gene_clustering import GeneClusteringProcessor
from preprocess.extract_embeddings import resolve_spot_geometry
from preprocess.utils import (
    get_train_slides,
    load_slide_gene_expression,
)


class DatasetAndPreprocessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / 'PRAD'
        self.st_dir = self.dataset_root / 'st'
        self.processed = self.dataset_root / 'processed_data'
        self.embeddings = self.processed / 'spot_features_uni'
        self.st_dir.mkdir(parents=True)
        self.embeddings.mkdir(parents=True)

        self.genes = [f'G{index:03d}' for index in range(200)]
        (self.processed / 'selected_gene_list.txt').write_text(
            '\n'.join(self.genes) + '\n',
            encoding='utf-8',
        )
        (self.processed / 'all_slide_lst.txt').write_text(
            'S1\nS2\nS3\n',
            encoding='utf-8',
        )
        counts = np.arange(600, dtype=np.float32).reshape(3, 200)
        counts[0, 0] = 2501
        counts[0, 1] = 0
        adata = ad.AnnData(
            X=counts,
            var={'gene_name': self.genes},
        )
        adata.var_names = self.genes
        adata.obsm['spatial'] = np.asarray(
            [[0.0, 0.0], [2.0, 1.0], [4.0, 3.0]],
            dtype=np.float32,
        )
        adata.write_h5ad(self.st_dir / 'S1.h5ad')
        for slide_id, offset in (('S2', 1.0), ('S3', 2.0)):
            training = ad.AnnData(
                X=np.full((3, 200), offset, dtype=np.float32),
            )
            training.var_names = self.genes
            training.obsm['spatial'] = np.asarray(
                [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]],
                dtype=np.float32,
            )
            training.write_h5ad(self.st_dir / f'{slide_id}.h5ad')
        torch.save(
            torch.arange(3 * 1024, dtype=torch.float32).reshape(3, 1024),
            self.embeddings / 'S1_uni.pt',
        )

    def tearDown(self):
        self.temporary.cleanup()

    def make_dataset(self, **overrides):
        arguments = {
            'mode': 'test',
            'data_path': str(self.dataset_root),
            'expr_name': 'PRAD',
            'slide_val': '',
            'slide_test': 'S1',
            'encoder_name': 'uni',
            'max_gene_count': 2000,
        }
        arguments.update(overrides)
        return STDataset(**arguments)

    def test_discrete_tokens_clip_but_raw_targets_do_not(self):
        dataset = self.make_dataset()
        item = dataset[0]
        self.assertEqual(int(item['target_genes'][0]), 2000)
        self.assertEqual(float(item['raw_target_genes'][0]), 2501.0)
        self.assertEqual(int(item['target_genes'][1]), 0)
        self.assertEqual(item['target_genes'].dtype, torch.int64)
        first = dataset._load_emb('S1')
        second = dataset._load_emb('S1')
        self.assertEqual(first.data_ptr(), second.data_ptr())

    def test_non_raw_expression_is_rejected(self):
        dataset = self.make_dataset()
        with self.assertRaises(ValueError):
            dataset._raw_gene_expression(
                np.asarray([0.0, 1.25] + [0.0] * 198)
            )
        with self.assertRaises(ValueError):
            dataset._raw_gene_expression(
                np.asarray([-1.0] + [0.0] * 199)
            )

    def test_continuous_target_is_library_normalized_log1p(self):
        dataset = self.make_dataset(prediction_mode='continuous')
        item = dataset[1]
        recovered = torch.expm1(item['target_genes'])
        self.assertAlmostEqual(float(recovered.sum()), 10000.0, places=2)

    def test_existing_positions_are_still_normalized_per_slide(self):
        path = self.st_dir / 'S1.h5ad'
        adata = ad.read_h5ad(path)
        coordinates = adata.obsm.pop('spatial')
        adata.obsm['positions'] = coordinates
        adata.write_h5ad(path)

        dataset = self.make_dataset()
        item = dataset[1]
        np.testing.assert_allclose(
            item['positions'].numpy(),
            np.asarray([0.5, 1.0 / 3.0], dtype=np.float32),
        )

    def test_random_grouping_is_deterministic(self):
        first = self.make_dataset(
            grouping_mode='random',
            grouping_seed=13,
        )
        second = self.make_dataset(
            grouping_mode='random',
            grouping_seed=13,
        )
        self.assertEqual(first.genes, second.genes)
        self.assertNotEqual(first.genes, self.genes)

    def test_preprocess_selects_gene_names_in_requested_order(self):
        selected = [self.genes[7], self.genes[1], self.genes[199]]
        matrix = load_slide_gene_expression(
            str(self.processed),
            'S1',
            h5ad_root=str(self.st_dir),
            genes=selected,
        )
        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(float(matrix[1, 0]), 207.0)
        self.assertEqual(float(matrix[1, 1]), 201.0)

    def test_training_split_excludes_validation_and_test(self):
        slides = get_train_slides(
            str(self.processed),
            {'S1', 'S2'},
        )
        self.assertEqual(slides, ['S3'])

    def test_raw_diagnostic_statistics_use_training_slides_only(self):
        dataset = self.make_dataset()
        statistics = collect_training_count_statistics(dataset)
        self.assertEqual(statistics['slide_ids'], ['S2', 'S3'])
        self.assertEqual(statistics['total_spots'], 6)
        np.testing.assert_allclose(statistics['gene_means'], 1.5)

    def test_checkpoint_locks_gene_order_and_training_split(self):
        dataset = self.make_dataset()
        config = Dict(
            {
                'DATA': {
                    'selected_gene_order_sha256': _ordered_gene_hash(
                        dataset.genes
                    ),
                    'train_slides': ['S2'],
                    'val_slides': [],
                    'test_slides': ['S1'],
                },
                'slide_val': '',
                'slide_test': 'S1',
            }
        )
        validate_dataset_contract(config, dataset)
        training_slides = checkpoint_training_slides(config, dataset)
        self.assertEqual(training_slides, ['S2'])
        statistics = collect_training_count_statistics(
            dataset,
            training_slides,
        )
        np.testing.assert_allclose(statistics['gene_means'], 1.0)

        config.DATA.selected_gene_order_sha256 = '0' * 64
        with self.assertRaises(ValueError):
            validate_dataset_contract(config, dataset)

    def test_validation_and_test_splits_must_be_disjoint(self):
        with self.assertRaisesRegex(ValueError, 'must be disjoint'):
            self.make_dataset(slide_val='S1', slide_test='S1')

    def test_zscore_is_per_gene_across_spots(self):
        features = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [7.0, 7.0, 7.0],
            ]
        )
        normalized, constant_count = (
            GeneClusteringProcessor._zscore_gene_profiles(features)
        )
        self.assertEqual(constant_count, 1)
        self.assertAlmostEqual(float(normalized[0].mean()), 0.0)
        self.assertAlmostEqual(float(normalized[0].std()), 1.0)
        np.testing.assert_array_equal(normalized[1], np.zeros(3))

    def test_spot_centers_remain_in_full_resolution_coordinates(self):
        adata = ad.AnnData(X=np.zeros((2, 1), dtype=np.float32))
        adata.obsm['spatial'] = np.asarray(
            [[100.0, 200.0], [300.0, 400.0]],
            dtype=np.float32,
        )
        adata.uns['spatial'] = {
            'ST': {
                'scalefactors': {
                    'tissue_hires_scalef': 0.1,
                    'spot_diameter_fullres': 80.0,
                }
            }
        }
        centers, diameter = resolve_spot_geometry(adata, 'S1')
        np.testing.assert_array_equal(centers, adata.obsm['spatial'])
        self.assertEqual(diameter, 80.0)


if __name__ == '__main__':
    unittest.main()
