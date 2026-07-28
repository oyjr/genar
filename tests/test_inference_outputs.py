from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from addict import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from inference import (
    calculate_metrics,
    predictions_to_raw_counts,
    save_results,
)


class InferenceOutputTests(unittest.TestCase):
    def setUp(self):
        self.config = Dict(
            {
                'MODEL': {
                    'num_genes': 4,
                    'prediction_mode': 'discrete',
                    'library_scale': 10000.0,
                    'scale_dims': [1, 2, 4],
                },
                'expr_name': 'PRAD',
                'encoder_name': 'uni',
                'max_gene_count': 8,
            }
        )
        self.predictions = torch.tensor(
            [
                [0, 1, 0, 2],
                [0, 2, 0, 2],
                [1, 2, 2, 2],
                [2, 4, 3, 2],
            ],
            dtype=torch.float32,
        )
        self.targets = torch.tensor(
            [
                [0, 1, 0, 3],
                [0, 2, 1, 3],
                [1, 3, 2, 3],
                [2, 4, 3, 3],
            ],
            dtype=torch.float32,
        )

    def test_calculation_and_json_npz_outputs_are_end_to_end_valid(self):
        metrics = calculate_metrics(
            self.predictions,
            self.targets,
            self.targets,
            self.config,
            dispersion=np.ones(4),
            training_gene_means=np.asarray([1.0, 20.0, 150.0, 2.0]),
        )
        self.assertIn('pcc_200', metrics['standard'])
        self.assertIn('zero_f1', metrics['raw_count'])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / 'data'
            processed = data_root / 'processed_data'
            processed.mkdir(parents=True)
            (processed / 'selected_gene_list.txt').write_text(
                'A\nB\nC\nD\n',
                encoding='utf-8',
            )
            self.config.data_path = str(data_root)
            dataset = type('DatasetStub', (), {'genes': ['A', 'B', 'C', 'D']})()
            output = root / 'output'
            save_results(
                metrics,
                self.predictions,
                self.targets,
                average_loss=1.25,
                output_dir=str(output),
                slide_id='S1',
                dataset=dataset,
                config=self.config,
                checkpoint_path=str(root / 'model.ckpt'),
                save_predictions=True,
                training_statistics={
                    'slide_ids': ['S2'],
                    'total_spots': 4,
                },
                dispersion_source='unit_test',
            )
            summary_path = output / 'S1_inference_summary.json'
            summary = json.loads(summary_path.read_text(encoding='utf-8'))
            self.assertEqual(summary['schema_version'], 1)
            self.assertEqual(summary['num_genes'], 4)
            self.assertEqual(summary['checkpoint_file'], 'model.ckpt')
            self.assertNotIn(str(root), summary_path.read_text(encoding='utf-8'))
            with np.load(
                output / 'S1_predictions.npz',
                allow_pickle=False,
            ) as arrays:
                self.assertEqual(
                    arrays['predicted_counts'].shape,
                    (4, 4),
                )

    def test_continuous_inverse_uses_observed_library_depth(self):
        normalized = torch.log1p(
            torch.tensor([[2500.0, 7500.0]], dtype=torch.float32)
        )
        raw_target = torch.tensor([[2.0, 6.0]])
        recovered = predictions_to_raw_counts(
            normalized,
            raw_target,
            prediction_mode='continuous',
            library_scale=10000.0,
        )
        np.testing.assert_allclose(recovered, [[2.0, 6.0]], rtol=1.0e-5)


if __name__ == '__main__':
    unittest.main()
