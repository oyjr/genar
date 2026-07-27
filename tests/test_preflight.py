import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'genar_preflight',
    REPO_ROOT / 'scripts' / 'preflight.py',
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not load scripts/preflight.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class PreflightTests(unittest.TestCase):
    def test_data_only_preflight_does_not_require_cuda(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary)
            dataset_root = data_root / 'PRAD'
            processed = dataset_root / 'processed_data'
            st_dir = dataset_root / 'st'
            embeddings = processed / 'spot_features_uni'
            st_dir.mkdir(parents=True)
            embeddings.mkdir(parents=True)

            slides = ['TRAIN1', 'MEND145']
            genes = [f'GENE{index:03d}' for index in range(200)]
            (processed / 'all_slide_lst.txt').write_text(
                '\n'.join(slides) + '\n',
                encoding='utf-8',
            )
            for filename in (
                'selected_gene_list.txt',
                'unclustered_selected_gene_list.txt',
            ):
                (processed / filename).write_text(
                    '\n'.join(genes) + '\n',
                    encoding='utf-8',
                )
            gene_hash = preflight.sha256_lines(genes)
            (processed / 'clustering_info.json').write_text(
                json.dumps(
                    {
                        'dataset': 'PRAD',
                        'train_slides': ['TRAIN1'],
                        'clustered_order': list(range(200)),
                        'scale_dims': [1, 4, 8, 40, 100, 200],
                        'algorithm': 'kmeans_hierarchical',
                        'selected_gene_count': 200,
                        'source_gene_list_sha256': gene_hash,
                        'output_gene_list_sha256': gene_hash,
                        'excluded_validation_test_slides': ['MEND145'],
                    }
                ),
                encoding='utf-8',
            )
            for slide in slides:
                (st_dir / f'{slide}.h5ad').touch()
                (embeddings / f'{slide}_uni.pt').touch()

            argv = [
                'preflight.py',
                '--dataset',
                'PRAD',
                '--data-root',
                str(data_root),
                '--encoder',
                'uni',
                '--data-only',
            ]
            stdout = io.StringIO()
            with mock.patch.object(sys, 'argv', argv):
                with mock.patch.object(
                    preflight,
                    'package_versions',
                    return_value={'torch': 'test'},
                ):
                    with mock.patch.object(
                        preflight.torch.cuda,
                        'device_count',
                        return_value=0,
                    ):
                        with contextlib.redirect_stdout(stdout):
                            self.assertEqual(preflight.main(), 0)

            report = json.loads(stdout.getvalue())
            self.assertEqual(report['status'], 'PASS')
            self.assertEqual(report['hardware_check'], 'skipped')
            self.assertEqual(report['selected_gpus'], [])


if __name__ == '__main__':
    unittest.main()
