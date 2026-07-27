from __future__ import annotations

import logging
import sys
import tempfile
import unittest
from pathlib import Path

import pytorch_lightning as pl
import torch
from addict import Dict
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from configs import (
    PAPER_BATCH_SIZE,
    PAPER_LEARNING_RATE,
    PAPER_MAX_GENE_COUNT,
    PAPER_SCALE_DIMS,
)
from inference import load_model_from_checkpoint
from main import (
    build_config_from_args,
    get_parse,
    validate_scale_dims,
)
from model import ModelInterface


logging.disable(logging.INFO)
torch.set_num_threads(1)


def tiny_interface_config() -> Dict:
    return Dict(
        {
            'MODEL': {
                'model_name': 'GENAR',
                'model_variant': 'original',
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
                'prediction_mode': 'discrete',
                'final_loss_mode': 'gaussian_kl',
                'scale_loss_weights': [1.0, 1.0, 1.0],
            },
            'TRAINING': {
                'learning_rate': 1.0e-4,
                'scale_learning_rate': False,
                'weight_decay': 1.0e-4,
                'num_epochs': 1,
                'gradient_clip_val': 1.0,
                'lr_scheduler': {
                    'name': 'reduce_on_plateau',
                    'patience': 0,
                    'factor': 0.5,
                    'monitor': 'train_loss_final',
                    'mode': 'min',
                },
            },
            'DATA': {
                'train_dataloader': {'batch_size': 64},
                'grouping_mode': 'kmeans',
                'grouping_seed': 42,
            },
            'INFERENCE': {'top_k': 1},
            'expr_name': 'PRAD',
            'encoder_name': 'uni',
            'max_gene_count': 8,
            'prediction_mode': 'discrete',
        }
    )


class ConfigurationAndCheckpointTests(unittest.TestCase):
    def test_final_paper_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'PRAD').mkdir()
            args = get_parse().parse_args(
                ['--dataset', 'PRAD', '--data-root', str(root)]
            )
            config = build_config_from_args(args)
        self.assertEqual(
            tuple(config.MODEL.scale_dims),
            PAPER_SCALE_DIMS,
        )
        self.assertEqual(
            config.DATA.train_dataloader.batch_size,
            PAPER_BATCH_SIZE,
        )
        self.assertEqual(config.DATA.global_batch_size, PAPER_BATCH_SIZE)
        self.assertEqual(
            config.TRAINING.learning_rate,
            PAPER_LEARNING_RATE,
        )
        self.assertFalse(config.TRAINING.scale_learning_rate)
        self.assertEqual(config.max_gene_count, PAPER_MAX_GENE_COUNT)
        self.assertEqual(
            config.MODEL.vocab_size,
            PAPER_MAX_GENE_COUNT + 1,
        )
        self.assertEqual(config.MODEL.final_loss_mode, 'gaussian_kl')
        self.assertTrue(config.deterministic)

    def test_four_gpu_default_preserves_global_paper_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'PRAD').mkdir()
            args = get_parse().parse_args(
                [
                    '--dataset',
                    'PRAD',
                    '--data-root',
                    str(root),
                    '--gpus',
                    '4',
                ]
            )
            config = build_config_from_args(args)
        self.assertEqual(config.DATA.global_batch_size, PAPER_BATCH_SIZE)
        self.assertEqual(
            config.DATA.train_dataloader.batch_size,
            PAPER_BATCH_SIZE // 4,
        )

    def test_scale_validation_rejects_non_nested_layouts(self):
        self.assertEqual(
            validate_scale_dims((1, 2, 4, 8, 40, 100, 200)),
            (1, 2, 4, 8, 40, 100, 200),
        )
        with self.assertRaises(ValueError):
            validate_scale_dims((1, 3, 200))
        with self.assertRaises(ValueError):
            validate_scale_dims((1, 4, 100))

    def test_foundation_baseline_uses_the_same_count_vocabulary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'PRAD').mkdir()
            args = get_parse().parse_args(
                [
                    '--dataset',
                    'PRAD',
                    '--data-root',
                    str(root),
                    '--model',
                    'FOUNDATION_BASELINE',
                ]
            )
            config = build_config_from_args(args)
        self.assertEqual(config.MODEL.vocab_size, 2001)
        self.assertEqual(config.MODEL.max_gene_count, 2000)

    def test_optimizer_uses_exact_unscaled_adam_learning_rate(self):
        module = ModelInterface(tiny_interface_config())
        optimizer = module.configure_optimizers()['optimizer']
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertAlmostEqual(
            optimizer.param_groups[0]['lr'],
            1.0e-4,
        )

    def test_new_checkpoint_round_trip_is_weights_only_safe(self):
        module = ModelInterface(tiny_interface_config())
        checkpoint = {
            'genar_checkpoint_schema_version': 1,
            'hyper_parameters': dict(module.hparams),
            'state_dict': module.state_dict(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.ckpt'
            torch.save(checkpoint, path)
            restored, config = load_model_from_checkpoint(
                str(path),
                torch.device('cpu'),
            )
        self.assertEqual(config.MODEL.num_genes, 4)
        for expected, actual in zip(
            module.state_dict().values(),
            restored.state_dict().values(),
        ):
            self.assertTrue(torch.equal(expected, actual))

    def test_lightning_epoch_monitor_saves_a_restorable_checkpoint(self):
        samples = []
        for index in range(4):
            generator = torch.Generator().manual_seed(index)
            values = torch.tensor([index % 3, 1, 0, 2])
            samples.append(
                {
                    'img': torch.randn(2, generator=generator),
                    'positions': torch.rand(2, generator=generator),
                    'target_genes': values,
                    'raw_target_genes': values.float(),
                }
            )
        loader = DataLoader(samples, batch_size=2)

        with tempfile.TemporaryDirectory() as temporary:
            callback = ModelCheckpoint(
                dirpath=temporary,
                monitor='train_loss_final',
                mode='min',
                save_top_k=1,
                filename='best-{epoch:02d}-{train_loss_final:.6f}',
            )
            trainer = pl.Trainer(
                accelerator='cpu',
                devices=1,
                max_epochs=1,
                callbacks=[callback],
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                deterministic=True,
                num_sanity_val_steps=0,
                log_every_n_steps=1,
            )
            trainer.fit(
                ModelInterface(tiny_interface_config()),
                loader,
                loader,
            )
            self.assertTrue(callback.best_model_path)
            restored, config = load_model_from_checkpoint(
                callback.best_model_path,
                torch.device('cpu'),
            )
        self.assertEqual(config.MODEL.num_genes, 4)
        self.assertEqual(restored.model.num_genes, 4)


if __name__ == '__main__':
    unittest.main()
