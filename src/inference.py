#!/usr/bin/env python3
"""Inference script for GenAR models."""

import os
import sys
import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path

# Ensure project modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# Project modules
from dataset.hest_dataset import STDataset
from model import ModelInterface
from model.model_metrics import ModelMetrics
from utils import fix_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")

# Dataset configuration
DEFAULT_DATA_ROOT = os.environ.get('GENAR_DATA_ROOT', './data')
DATASETS = {
    'PRAD': {
        'dir_name': 'PRAD',
        'val_slides': 'MEND144',
        'test_slides': 'MEND144',
        'recommended_encoder': 'uni'
    },
    'her2st': {
        'dir_name': 'her2st',
        'val_slides': 'SPA148',
        'test_slides': 'SPA148',
        'recommended_encoder': 'conch'
    }
}

# Encoder feature dimensions
ENCODER_FEATURE_DIMS = {
    'uni': 1024,
    'conch': 512,
    'resnet18': 512,
}


def parse_args():
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(
        description='GenAR inference runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python src/inference.py \\
      --ckpt_path logs/PRAD/GENAR/best-epoch=epoch=01-loss=...ckpt \\
      --dataset PRAD --slide_id MEND144 --output_dir ./inference_results
        """
    )

    # Required arguments
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Model checkpoint to load')
    parser.add_argument('--dataset', type=str, required=True, choices=list(DATASETS.keys()),
                        help='Dataset name (PRAD or her2st)')
    parser.add_argument('--slide_id', type=str, required=True,
                        help='Slide identifier to evaluate (e.g. MEND144)')

    # Optional arguments
    parser.add_argument('--data-root', type=str, default=DEFAULT_DATA_ROOT,
                        help='Root directory containing dataset folders '
                             '(default: $GENAR_DATA_ROOT or ./data)')
    parser.add_argument('--encoder', type=str, choices=list(ENCODER_FEATURE_DIMS.keys()),
                        help='Encoder type (defaults to the dataset recommendation)')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Output directory for all artifacts')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU index to use (set -1 for CPU)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Inference batch size (default: 64)')
    parser.add_argument('--max_gene_count', type=int, default=500,
                        help='Maximum gene count value (default: 500)')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Random seed (default: 2021)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Store per-spot predictions as CSV/NPY')
    
    return parser.parse_args()


def setup_device(gpu_id: int):
    """Select the torch device used for inference."""
    if gpu_id == -1:
        device = torch.device('cpu')
        logger.info("Running inference on CPU")
    else:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{gpu_id}')
            logger.info(f"Running inference on GPU {gpu_id}")
        else:
            device = torch.device('cpu')
            logger.warning("CUDA not available, falling back to CPU")

    return device


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Restore a trained Lightning module from disk."""
    logger.info(f"Loading checkpoint: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract configuration from the checkpoint
    if 'hyper_parameters' not in checkpoint:
        raise ValueError("Checkpoint is missing the hyper_parameters section")

    config = checkpoint['hyper_parameters']['config']
    logger.info(f"Restored model configuration: {config.MODEL.model_name}")

    # Create and load the model
    model = ModelInterface(config)
    
    # Load weights
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    logger.info("Model successfully loaded")
    return model, config


def create_test_dataloader(config, slide_id: str, batch_size: int = 64):
    """Build the dataloader used for inference."""
    logger.info(f"Building test dataloader for slide {slide_id}")

    # Shared dataset parameters
    base_params = {
        'data_path': config.data_path,
        'expr_name': config.expr_name,
        'slide_val': slide_id,  # Use the requested slide in both splits
        'slide_test': slide_id,
        'encoder_name': config.encoder_name,
        'max_gene_count': getattr(config, 'max_gene_count', 500),
    }

    # Dataset and loader
    test_dataset = STDataset(mode='test', **base_params)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Test dataset size: {len(test_dataset)} spots")
    return test_loader, test_dataset


def run_inference(model, test_loader, device: torch.device):
    """Run batched inference and collect predictions, targets, and loss."""
    logger.info("Starting inference")
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_losses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move tensor inputs to the selected device
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(device)

            results = model.manual_inference_step(batch, phase='test')

            # Accumulate results
            all_predictions.append(results['predictions'].cpu())
            all_targets.append(results['targets'].cpu())
            all_losses.append(results['loss_final'].item())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")

    # Merge results
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    avg_loss = np.mean(all_losses)

    logger.info(f"Inference complete: {predictions.shape[0]} samples, avg loss {avg_loss:.6f}")

    return predictions, targets, avg_loss


def calculate_detailed_metrics(predictions: torch.Tensor, targets: torch.Tensor):
    """Compute evaluation metrics and summary statistics."""
    logger.info("Computing metrics")

    # Convert to NumPy arrays
    if torch.is_tensor(predictions):
        predictions = predictions.numpy()
    if torch.is_tensor(targets):
        targets = targets.numpy()

    # Create the helper objects expected by ModelMetrics
    class SimpleConfig:
        def __init__(self):
            self.MODEL = type('obj', (object,), {'num_genes': 200})()

    config = SimpleConfig()

    class SimpleLightningModule:
        def log(self, *args, **kwargs):
            pass

    lightning_module = SimpleLightningModule()

    model_metrics = ModelMetrics(config, lightning_module)

    # Pearson correlation metrics with log2 scaling
    pcc_metrics = model_metrics.calculate_comprehensive_pcc_metrics(
        predictions, targets, apply_log2=True
    )

    # Aggregate statistics
    pred_stats = {
        'pred_mean': float(np.mean(predictions)),
        'pred_std': float(np.std(predictions)),
        'pred_min': float(np.min(predictions)),
        'pred_max': float(np.max(predictions)),
    }
    
    target_stats = {
        'target_mean': float(np.mean(targets)),
        'target_std': float(np.std(targets)),
        'target_min': float(np.min(targets)),
        'target_max': float(np.max(targets)),
    }

    # Gene-wise correlations
    gene_correlations = model_metrics.calculate_gene_correlations(targets, predictions)

    # Per-gene statistics
    gene_stats = []
    for i in range(predictions.shape[1]):
        gene_pred = predictions[:, i]
        gene_target = targets[:, i]
        
        gene_stat = {
            'gene_idx': i,
            'correlation': float(gene_correlations[i]),
            'pred_mean': float(np.mean(gene_pred)),
            'target_mean': float(np.mean(gene_target)),
            'pred_std': float(np.std(gene_pred)),
            'target_std': float(np.std(gene_target)),
        }
        gene_stats.append(gene_stat)

    # Sort by correlation
    sorted_gene_stats = sorted(gene_stats, key=lambda x: x['correlation'], reverse=True)
    
    return {
        'pcc_metrics': pcc_metrics,
        'pred_stats': pred_stats,
        'target_stats': target_stats,
        'gene_correlations': gene_correlations,
        'gene_stats': gene_stats,
        'sorted_gene_stats': sorted_gene_stats
    }


def print_results(metrics: dict, avg_loss: float):
    """Print a textual summary of inference results."""
    pcc_metrics = metrics['pcc_metrics']
    pred_stats = metrics['pred_stats']
    target_stats = metrics['target_stats']
    sorted_gene_stats = metrics['sorted_gene_stats']
    
    print("\n" + "="*60)
    print("GenAR inference summary")
    print("="*60)
    
    print("\nKey metrics:")
    print(f"   Loss:            {avg_loss:.6f}")
    print(f"   PCC-10:          {pcc_metrics['pcc_10']:.4f}")
    print(f"   PCC-50:          {pcc_metrics['pcc_50']:.4f}")
    print(f"   PCC-200:         {pcc_metrics['pcc_200']:.4f}")
    print(f"   MSE:             {pcc_metrics['mse']:.6f}")
    print(f"   MAE:             {pcc_metrics['mae']:.6f}")
    print(f"   RVD:             {pcc_metrics['rvd']:.6f}")
    
    print("\nPrediction statistics:")
    print(f"   Mean:            {pred_stats['pred_mean']:.2f}")
    print(f"   Std:             {pred_stats['pred_std']:.2f}")
    print(f"   Range:           [{pred_stats['pred_min']:.2f}, {pred_stats['pred_max']:.2f}]")
    
    print("\nTarget statistics:")
    print(f"   Mean:            {target_stats['target_mean']:.2f}")
    print(f"   Std:             {target_stats['target_std']:.2f}")
    print(f"   Range:           [{target_stats['target_min']:.2f}, {target_stats['target_max']:.2f}]")
    
    print("\nTop-10 correlated genes:")
    for i, gene_stat in enumerate(sorted_gene_stats[:10]):
        print(f"   {i+1:2d}. Gene {gene_stat['gene_idx']:3d}: PCC={gene_stat['correlation']:.4f}")
    
    print("\nBottom-5 correlated genes:")
    for i, gene_stat in enumerate(sorted_gene_stats[-5:]):
        rank = len(sorted_gene_stats) - 4 + i
        print(f"   {rank:2d}. Gene {gene_stat['gene_idx']:3d}: PCC={gene_stat['correlation']:.4f}")
    
    print("\n" + "="*60)


def save_results(metrics: dict, predictions: torch.Tensor, targets: torch.Tensor,
                avg_loss: float, output_dir: str, slide_id: str, save_predictions: bool = False):
    """Write metrics and optional predictions to disk."""
    logger.info(f"Saving outputs to {output_dir}")

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Persist summary metrics
    results_summary = {
        'slide_id': slide_id,
        'timestamp': datetime.now().isoformat(),
        'num_samples': predictions.shape[0],
        'num_genes': predictions.shape[1],
        'avg_loss': avg_loss,
        **metrics['pcc_metrics'],
        **metrics['pred_stats'],
        **metrics['target_stats']
    }
    
    summary_file = os.path.join(output_dir, f'{slide_id}_inference_summary.json')
    import json
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save per-gene statistics
    gene_stats_df = pd.DataFrame(metrics['gene_stats'])
    gene_stats_file = os.path.join(output_dir, f'{slide_id}_gene_statistics.csv')
    gene_stats_df.to_csv(gene_stats_file, index=False)

    # Optionally persist the raw predictions
    if save_predictions:
        predictions_file = os.path.join(output_dir, f'{slide_id}_predictions.npz')
        np.savez_compressed(
            predictions_file,
            predictions=predictions.numpy() if torch.is_tensor(predictions) else predictions,
            targets=targets.numpy() if torch.is_tensor(targets) else targets
        )
        logger.info(f"Stored detailed predictions at {predictions_file}")

    logger.info(f"Saved summary to {summary_file}")
    logger.info(f"Saved gene statistics to {gene_stats_file}")


def main():
    """CLI entry point for inference."""
    args = parse_args()

    # Seed everything
    fix_seed(args.seed)

    # Device selection
    device = setup_device(args.gpu_id)

    # Validate checkpoint file
    if not os.path.exists(args.ckpt_path):
        logger.error(f"Checkpoint not found: {args.ckpt_path}")
        return

    # Validate dataset choice
    if args.dataset not in DATASETS:
        logger.error(f"Unsupported dataset: {args.dataset}")
        return

    dataset_info = DATASETS[args.dataset]
    data_root = os.path.abspath(args.data_root)
    dataset_path = os.path.join(data_root, dataset_info['dir_name'])
    if not os.path.exists(dataset_path):
        logger.warning("Dataset path does not exist: %s", dataset_path)

    # Resolve encoder selection
    encoder_name = args.encoder or dataset_info['recommended_encoder']
    
    logger.info("Inference configuration:")
    logger.info(f"  dataset: {args.dataset}")
    logger.info(f"  slide:   {args.slide_id}")
    logger.info(f"  encoder: {encoder_name}")
    logger.info(f"  checkpoint: {args.ckpt_path}")
    logger.info(f"  output:  {args.output_dir}")
    
    try:
        # Load model
        model, config = load_model_from_checkpoint(args.ckpt_path, device)
        
        # Update dataset-related configuration
        config.data_path = dataset_path
        config.expr_name = args.dataset
        config.encoder_name = encoder_name
        config.max_gene_count = args.max_gene_count
        
        # Build the test dataloader
        test_loader, test_dataset = create_test_dataloader(config, args.slide_id, args.batch_size)
        
        # Run the forward pass
        predictions, targets, avg_loss = run_inference(model, test_loader, device)
        
        # Aggregate metrics
        metrics = calculate_detailed_metrics(predictions, targets)
        
        # Print a textual report
        print_results(metrics, avg_loss)
        
        # Persist results
        save_results(metrics, predictions, targets, avg_loss, 
                    args.output_dir, args.slide_id, args.save_predictions)
        
        logger.info("Inference completed successfully")
        
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code) 
