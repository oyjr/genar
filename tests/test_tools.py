from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


class ToolTests(unittest.TestCase):
    def test_token_usage_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / 'predictions.npz'
            np.savez_compressed(
                predictions,
                predicted_counts=np.asarray([[0, 1, 1], [2, 2, 5]]),
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / 'tools' / 'analyze_token_usage.py'),
                    str(predictions),
                    '--count-cap',
                    '5',
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        result = json.loads(completed.stdout)
        self.assertEqual(result['unique_tokens_used'], 4)
        self.assertEqual(result['model_vocabulary_size_including_zero'], 6)

    def test_nominal_enrichment_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            groups = root / 'groups.tsv'
            reference = root / 'reference.tsv'
            output = root / 'enrichment.csv'
            groups.write_text(
                'group\tgene\nA\tG1\nA\tG2\nB\tG3\n',
                encoding='utf-8',
            )
            reference.write_text(
                'term\tgene\nT1\tG1\nT1\tG4\nT2\tG3\n',
                encoding='utf-8',
            )
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / 'tools' / 'gene_group_enrichment.py'),
                    '--groups',
                    str(groups),
                    '--reference',
                    str(reference),
                    '--background-size',
                    '20000',
                    '--output',
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            table = pd.read_csv(output)
        self.assertEqual(len(table), 4)
        self.assertFalse(table['multiple_testing_adjusted'].any())


if __name__ == '__main__':
    unittest.main()
