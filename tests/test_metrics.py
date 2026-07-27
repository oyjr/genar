from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from addict import Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from model.model_metrics import ModelMetrics


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.metrics = ModelMetrics(
            Dict(
                {
                    'MODEL': {
                        'num_genes': 4,
                        'prediction_mode': 'discrete',
                    }
                }
            )
        )
        self.true = np.asarray(
            [
                [0, 1, 0, 3],
                [0, 2, 1, 3],
                [1, 3, 2, 3],
                [2, 4, 3, 3],
            ],
            dtype=np.float64,
        )
        self.pred = np.asarray(
            [
                [0, 1, 0, 2],
                [0, 2, 0, 2],
                [1, 2, 2, 2],
                [2, 4, 3, 2],
            ],
            dtype=np.float64,
        )

    def test_constant_gene_is_reported_not_fatal(self):
        result = self.metrics.calculate_comprehensive_pcc_metrics(
            self.pred,
            self.true,
            apply_log2=True,
        )
        self.assertEqual(result['undefined_pcc_genes'], 1)
        self.assertEqual(result['valid_pcc_genes'], 3)
        self.assertTrue(np.isfinite(result['pcc_200']))

    def test_all_constant_predictions_report_undefined_pcc(self):
        result = self.metrics.calculate_comprehensive_pcc_metrics(
            np.zeros_like(self.true),
            self.true,
            apply_log2=True,
        )
        self.assertEqual(result['valid_pcc_genes'], 0)
        self.assertEqual(result['undefined_pcc_genes'], 4)
        self.assertTrue(np.isnan(result['pcc_10']))
        self.assertTrue(np.isfinite(result['mse']))

    def test_raw_count_diagnostics_include_paper_analyses(self):
        result = self.metrics.calculate_raw_count_diagnostics(
            self.pred,
            self.true,
        )
        for key in (
            'spearman',
            'mse_count',
            'mae_count',
            'nb_nll',
            'zero_precision',
            'zero_recall',
            'zero_f1',
            'library_size_spearman',
            'library_size_pearson',
            'expression_bins',
        ):
            self.assertIn(key, result)
        self.assertTrue(np.isfinite(result['nb_nll']))
        self.assertGreaterEqual(result['zero_f1'], 0.0)
        self.assertLessEqual(result['zero_f1'], 1.0)

    def test_nb_dispersion_shape_validation(self):
        with self.assertRaises(ValueError):
            self.metrics.negative_binomial_nll(
                self.pred,
                self.true,
                dispersion=np.ones(3),
            )


if __name__ == '__main__':
    unittest.main()
