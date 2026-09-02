import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_hest import patterns, prepare_metadata, sample_ids


class PublicReleaseTests(unittest.TestCase):
    def test_registries_match(self):
        model_registry = json.loads(
            (ROOT / "models.json").read_text(encoding="utf-8")
        )
        dataset_registry = json.loads(
            (ROOT / "configs" / "datasets.json").read_text(encoding="utf-8")
        )
        models = model_registry["models"]
        datasets = dataset_registry["datasets"]
        expected = {
            "PRAD": (23, "MEND145", "prad"),
            "her2st": (36, "SPA148", "her2st"),
            "kidney": (23, "NCBI697", "kidney"),
            "mouse_brain": (14, "NCBI667", "mouse_brain"),
            "ccRCC": (24, "INT2", "ccrcc"),
        }
        self.assertEqual(set(models), set(expected))
        self.assertEqual(set(datasets), set(expected))
        self.assertEqual(
            dataset_registry["revision"],
            "42fa9a56fa7f439e466b7e306cda4bd224e4fe84",
        )
        for name, (count, test_slide, filename) in expected.items():
            ids = sample_ids(datasets[name])
            self.assertEqual(len(ids), count)
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(datasets[name]["test"], test_slide)
            self.assertIn(test_slide, ids)
            self.assertEqual(
                models[name],
                {
                    "checkpoint": f"checkpoints/{filename}.ckpt",
                    "genes": f"genes/{filename}.txt",
                },
            )
            expected_patterns = {
                *(f"st/{slide_id}.h5ad" for slide_id in ids),
                *(f"wsis/{slide_id}.*" for slide_id in ids),
            }
            self.assertEqual(set(patterns(datasets[name])), expected_patterns)

    def test_metadata_directory_creation_without_network(self):
        datasets = json.loads(
            (ROOT / "configs" / "datasets.json").read_text(encoding="utf-8")
        )["datasets"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gene_source = root / "prad.txt"
            gene_source.write_bytes(b"GENE1\nGENE2\n")
            output = root / "data"
            processed = prepare_metadata(datasets["PRAD"], output, gene_source)

            self.assertEqual(processed, output / "PRAD" / "processed_data")
            self.assertEqual(
                (processed / "all_slide_lst.txt")
                .read_text(encoding="utf-8")
                .splitlines(),
                sample_ids(datasets["PRAD"]),
            )
            self.assertEqual(
                (processed / "selected_gene_list.txt").read_bytes(),
                gene_source.read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
