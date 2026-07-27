from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from configs import PAPER_SCALE_DIMS
from model.genar.multiscale_genar import MultiScaleGenAR
from model.genar.multiscale_genar_no_film import MultiScaleGenARNoFiLM


logging.disable(logging.INFO)
torch.set_num_threads(1)


def tiny_model(**overrides):
    arguments = {
        'vocab_size': 9,
        'num_genes': 4,
        'scale_dims': (1, 2, 4),
        'embed_dim': 4,
        'num_heads': 1,
        'num_layers': 1,
        'mlp_ratio': 2.0,
        'histology_feature_dim': 2,
        'spatial_coord_dim': 2,
        'condition_embed_dim': 4,
        'drop_path_rate': 0.0,
        'adaptive_sigma_alpha': 0.01,
        'adaptive_sigma_beta': 1.0,
    }
    arguments.update(overrides)
    return MultiScaleGenAR(**arguments)


class MultiScaleCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.discrete = tiny_model()

    def test_mixed_integer_fractional_soft_targets_match_manual_kl(self):
        targets = torch.tensor(
            [
                [0, 0, 1, 0],
                [2, 2, 2, 2],
            ],
            dtype=torch.long,
        )
        soft_target = self.discrete._create_hierarchical_targets(targets)[1]
        logits = torch.randn(2, 2, 9)
        actual = self.discrete._compute_soft_label_loss(logits, soft_target)

        distribution = torch.zeros_like(logits)
        distribution.scatter_add_(
            -1,
            soft_target['floor_targets'].unsqueeze(-1),
            (1.0 - soft_target['weights']).unsqueeze(-1),
        )
        distribution.scatter_add_(
            -1,
            soft_target['ceil_targets'].unsqueeze(-1),
            soft_target['weights'].unsqueeze(-1),
        )
        self.assertTrue(
            torch.allclose(
                distribution.sum(dim=-1),
                torch.ones(2, 2),
            )
        )
        expected = F.kl_div(
            F.log_softmax(logits, dim=-1),
            distribution,
            reduction='batchmean',
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_discrete_training_and_top1_inference(self):
        target = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        histology = torch.randn(1, 2)
        spatial = torch.rand(1, 2)
        self.discrete.train()
        training = self.discrete(histology, spatial, target)
        self.assertEqual(training['predictions'].shape, (1, 4))
        self.assertTrue(torch.isfinite(training['loss']))

        self.discrete.eval()
        first = self.discrete(histology, spatial, top_k=1)
        second = self.discrete(histology, spatial, top_k=1)
        self.assertTrue(
            torch.equal(
                first['generated_sequence'],
                second['generated_sequence'],
            )
        )

    def test_discrete_fractional_targets_fail_closed(self):
        target = torch.tensor([[0.0, 1.5, 2.0, 3.0]])
        histology = torch.randn(1, 2)
        spatial = torch.rand(1, 2)
        self.discrete.train()
        with self.assertRaisesRegex(ValueError, 'integer count tokens'):
            self.discrete(histology, spatial, target)

    def test_continuous_regression_training_and_inference(self):
        model = tiny_model(prediction_mode='continuous')
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            sum(
                parameter.numel()
                for parameter in self.discrete.parameters()
            ),
        )
        target = torch.rand(1, 4)
        histology = torch.randn(1, 2)
        spatial = torch.rand(1, 2)
        model.train()
        training = model(histology, spatial, target)
        self.assertEqual(training['predictions'].shape, (1, 4))
        self.assertTrue(torch.isfinite(training['loss']))
        model.eval()
        inference = model(histology, spatial)
        self.assertEqual(inference['generated_sequence'].shape, (1, 4))

    def test_paper_hierarchy_sparse_soft_targets_never_have_zero_mass(self):
        model = MultiScaleGenAR(
            vocab_size=9,
            num_genes=200,
            scale_dims=PAPER_SCALE_DIMS,
            embed_dim=4,
            num_heads=1,
            num_layers=1,
            mlp_ratio=2.0,
            histology_feature_dim=2,
            condition_embed_dim=4,
            drop_path_rate=0.0,
        )
        target = torch.zeros(2, 200, dtype=torch.long)
        target[0, 17] = 1
        target[1] = torch.arange(200) % 9
        hierarchy = model._create_hierarchical_targets(target)
        for dimension, scale_target in zip(PAPER_SCALE_DIMS[:-1], hierarchy[:-1]):
            logits = torch.randn(2, dimension, 9)
            loss = model._compute_soft_label_loss(logits, scale_target)
            self.assertTrue(torch.isfinite(loss))
        mapping = model.gene_upsampling.group_mappings['scale_3_to_4']
        covered_targets = sorted(
            target_index
            for source_targets in mapping
            for target_index in source_targets
        )
        self.assertEqual(covered_targets, list(range(100)))

    def test_no_film_ablation_uses_same_correct_soft_distribution(self):
        model = MultiScaleGenARNoFiLM(
            vocab_size=9,
            num_genes=4,
            scale_dims=(1, 2, 4),
            embed_dim=4,
            num_heads=1,
            num_layers=1,
            mlp_ratio=2.0,
            histology_feature_dim=2,
            condition_embed_dim=4,
            drop_path_rate=0.0,
        )
        target = torch.tensor([[0, 0, 1, 0]], dtype=torch.long)
        soft_target = model._create_hierarchical_targets(target)[1]
        loss = model._compute_soft_label_loss(
            torch.randn(1, 2, 9),
            soft_target,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(model.use_gene_identity)
        self.assertFalse(
            any('gene_identity' in key or 'film_layer' in key
                for key in model.state_dict())
        )

        histology = torch.randn(1, 2)
        spatial = torch.rand(1, 2)
        model.train()
        training = model(histology, spatial, target)
        self.assertEqual(training['predictions'].shape, (1, 4))
        model.eval()
        inference = model(histology, spatial, top_k=1)
        self.assertEqual(inference['generated_sequence'].shape, (1, 4))

    def test_standalone_checkpoint_preserves_architecture(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'standalone.pt'
            self.discrete.save_checkpoint(str(checkpoint), epoch=3)
            restored = MultiScaleGenAR.load_checkpoint(
                str(checkpoint),
                device='cpu',
            )
        self.assertEqual(restored.mlp_ratio, self.discrete.mlp_ratio)
        self.assertEqual(restored.scale_dims, self.discrete.scale_dims)
        for expected, actual in zip(
            self.discrete.state_dict().values(),
            restored.state_dict().values(),
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_no_film_checkpoint_preserves_nondefault_architecture(self):
        model = MultiScaleGenARNoFiLM(
            vocab_size=9,
            num_genes=4,
            scale_dims=(1, 2, 4),
            embed_dim=4,
            num_heads=1,
            num_layers=1,
            mlp_ratio=3.0,
            histology_feature_dim=2,
            condition_embed_dim=4,
            cond_drop_rate=0.2,
            norm_eps=1.0e-5,
            drop_path_rate=0.0,
            scale_loss_weights=[1.0, 2.0, 3.0],
        )
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'no_film.pt'
            model.save_checkpoint(str(checkpoint), epoch=5)
            payload = torch.load(
                checkpoint,
                map_location='cpu',
                weights_only=True,
            )
            self.assertFalse(payload.pop('use_gene_identity'))
            torch.save(payload, checkpoint)
            restored = MultiScaleGenARNoFiLM.load_checkpoint(
                str(checkpoint),
                device='cpu',
            )
        self.assertEqual(restored.mlp_ratio, 3.0)
        self.assertEqual(restored.cond_drop_rate, 0.2)
        self.assertEqual(restored.norm_eps, 1.0e-5)
        self.assertEqual(restored.scale_loss_weights, [1.0, 2.0, 3.0])
        self.assertFalse(restored.use_gene_identity)
        for expected, actual in zip(
            model.state_dict().values(),
            restored.state_dict().values(),
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_kv_cache_fails_closed_for_cumulative_scale_passes(self):
        with self.assertRaisesRegex(RuntimeError, "not implemented"):
            self.discrete.enable_kv_cache()
        self.discrete.disable_kv_cache()

        no_film = MultiScaleGenARNoFiLM(
            vocab_size=17,
            num_genes=8,
            scale_dims=(1, 2, 4, 8),
            embed_dim=16,
            num_heads=4,
            num_layers=1,
            mlp_ratio=2.0,
            histology_feature_dim=12,
            condition_embed_dim=16,
            device='cpu',
        )
        with self.assertRaisesRegex(RuntimeError, "not implemented"):
            no_film.enable_kv_cache()
        no_film.disable_kv_cache()
        self.assertEqual(
            no_film.get_model_info()['maximum_sequence_length'],
            1 + sum(no_film.scale_dims),
        )


if __name__ == '__main__':
    unittest.main()
