"""
PyTorch Lightning interface for the GenAR model.
Core Lightning plumbing lives here; specialized helpers handle the details.
"""

import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# Helper utilities
from .model_metrics import ModelMetrics
from .model_utils import ModelUtils

# Logger setup
logging.basicConfig(level=logging.INFO)

# Default hyperparameters
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_GRADIENT_CLIP = 1.0



class ModelInterface(pl.LightningModule):
    """LightningModule wrapper around the GenAR architecture."""

    def __init__(self, config):
        super().__init__()
        
        # Dedicated logger for this interface
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Store config reference
        self.config = config
        # Persist only serializable hyperparameters to avoid OmegaConf issues
        hyperparams = {
            'model_name': getattr(config.MODEL, 'model_name', 'GENAR'),
            'num_genes': getattr(config.MODEL, 'num_genes', 200),
            'learning_rate': getattr(config.TRAINING, 'learning_rate', 1e-4),
            'batch_size': getattr(config.DATA.train_dataloader, 'batch_size', 256),
            'max_epochs': getattr(config.TRAINING, 'num_epochs', 200),
            'dataset': getattr(config, 'expr_name', 'unknown')
        }
        self.save_hyperparameters(hyperparams)

        # Helper utilities
        self.model_utils = ModelUtils(config, self)
        self.model_metrics = ModelMetrics(config, self)
        
        # Load the underlying model
        self._logger.info("Initialising GenAR model interface")
        self.model = self.model_utils.load_model()
        
        # Buffers for validation/test outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Inference parameters
        self.inference_top_k = self.model_utils.get_config('INFERENCE.top_k', 1)

    def _common_step(self, batch, batch_idx, phase: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared logic for train/val/test steps."""
        original_batch = batch.copy() if isinstance(batch, dict) else batch
        processed_batch = self.model_utils.preprocess_inputs(batch)
        
        loss_final = None

        # Validation and Test steps require special handling to avoid data leakage from teacher forcing
        if phase in ['val', 'test']:
            # Pass 1: Get realistic predictions without teacher forcing for metrics
            inference_batch = processed_batch.copy()
            if 'target_genes' in inference_batch:
                _ = inference_batch.pop('target_genes')
            
            with torch.no_grad():
                # Manually set model to eval mode for this pass
                self.model.eval()
                # Use top-k sampling for realistic, non-deterministic predictions
                inference_results = self.model(**inference_batch, top_k=self.inference_top_k)
            
            # Pass 2: Get the loss using ground truth targets
            # The model is already in eval mode from the trainer, so forward_training will be called
            # This pass will use teacher forcing, but only for loss calculation
            loss_results = self.model(**processed_batch)
            loss = self._compute_loss(loss_results, original_batch)
            loss_final = loss_results.get('loss_final', loss)

            # Extract predictions from the inference pass and targets from the original batch
            predictions, targets = self._extract_predictions_and_targets(inference_results, original_batch)

        else: # Training step
            # Single pass for training
            results_dict = self.model(**processed_batch)
            loss = self._compute_loss(results_dict, original_batch)
            loss_final = results_dict.get('loss_final', loss)
            predictions, targets = self._extract_predictions_and_targets(results_dict, original_batch)
        
        # Metric logging handled separately to avoid duplicate entries

        if predictions is not None:
            self.model_metrics.log_model_specific_metrics(
                phase,
                {'predictions': predictions.detach()}
            )

        return loss, loss_final, predictions, targets

    def training_step(self, batch, batch_idx):
        """Lightning training step."""
        loss, loss_final, _, _ = self._common_step(batch, batch_idx, 'train')
        # Show loss on the progress bar
        self.log('train_loss', loss_final, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # Log detailed loss for monitoring
        self.log('train_loss_final', loss_final, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Lightning validation step."""
        loss, loss_final, predictions, targets = self._common_step(batch, batch_idx, 'val')
        
        # Determine the effective batch size
        batch_size = targets.size(0) if hasattr(targets, 'size') else 1
        
        # Log composite validation loss (informational)
        self.log('val_loss', loss, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True,
                batch_size=batch_size,
                sync_dist=True,
                reduce_fx='mean')
        
        # Log final-scale loss (current primary monitor)
        self.log('val_loss_final', loss_final,
                on_step=False, 
                on_epoch=True, 
                prog_bar=True,
                batch_size=batch_size,
                sync_dist=True,
                reduce_fx='mean')
        
        # Collect outputs for detailed PCC calculation (skip sanity checks)
        if not (hasattr(self.trainer, 'sanity_checking') and self.trainer.sanity_checking):
            output = {
                'val_loss': loss_final,
                'predictions': predictions.detach().cpu(),
                'targets': targets.detach().cpu()
            }
            
            # Store for epoch-end aggregation
            self.validation_step_outputs.append(output)

    def test_step(self, batch, batch_idx):
        """Lightning test step."""
        loss, loss_final, predictions, targets = self._common_step(batch, batch_idx, 'test')
        
        # Log final-scale test loss
        self.log('test_loss_final', loss_final, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Collect outputs for PCC computation
        output = {
            'test_loss': loss_final,
            'predictions': predictions.detach().cpu(),
            'targets': targets.detach().cpu()
        }
        
        # Append to buffer
        if not hasattr(self, 'test_step_outputs'):
            self.test_step_outputs = []
        self.test_step_outputs.append(output)
        
        return output

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the training loss with optional fallbacks."""
        try:
            # Prefer the model-provided loss
            if 'loss' in outputs:
                total_loss = outputs['loss']
                
                # Extra training-only diagnostics
                if self.training and 'logits' in outputs:
                    with torch.no_grad():
                        logits = outputs['logits']
                        if 'full_target' in outputs:
                            targets = outputs['full_target']
                        elif 'target_genes' in batch:
                            targets = batch['target_genes']
                        else:
                            targets = None
                        
                        if targets is not None:
                            # Token-level accuracy
                            if logits.dim() == 3:
                                pred_tokens = logits.argmax(dim=-1)
                                targets_flat = targets.view(-1)
                                pred_flat = pred_tokens.view(-1)
                            else:
                                pred_tokens = logits.argmax(dim=-1)
                                targets_flat = targets.view(-1)
                                pred_flat = pred_tokens
                            
                            token_acc = (pred_flat == targets_flat).float().mean()
                            self.log('train_token_accuracy', token_acc, prog_bar=False, sync_dist=False)
                
                self._logger.debug(f"Using model-provided loss={total_loss:.4f}")
                
            else:
                # Fallback for models that don't return 'loss' but 'logits'
                # This part is now less likely to be used with the hierarchical model
                logits = outputs.get('logits')
                if logits is None:
                    raise KeyError("Model outputs must include 'loss' or 'logits'")
                
                targets = batch.get('target_genes')
                if targets is None:
                    raise KeyError("Batch is missing 'target_genes'")

                total_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                self._logger.debug(f"Computed fallback loss={total_loss:.4f}")
            
            # Validate numerical stability
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self._logger.error(f"Invalid loss value: {total_loss.item()}")
                raise ValueError("Loss is NaN or Inf")
                
            return total_loss
            
        except Exception as e:
            self._logger.error(f"Failed to compute loss: {str(e)}")
            self._logger.error(f"Output keys: {list(outputs.keys())}")
            raise

    def _extract_predictions_and_targets(self, results_dict: Dict[str, torch.Tensor], 
                                       batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract predictions and target tensors from a forward pass."""
        # The hierarchical model returns 'predictions' for the final scale during training/validation,
        # and 'generated_sequence' during pure inference.
        if 'predictions' in results_dict:
            predictions = results_dict['predictions']
        elif 'generated_sequence' in results_dict:
            predictions = results_dict['generated_sequence']
        else:
            raise ValueError("Model outputs must include 'predictions' or 'generated_sequence'")
        
        # Targets
        if 'target_genes' not in batch:
            raise ValueError("Batch is missing 'target_genes'")
        targets = batch['target_genes']
        
        # Sanity-check prediction dimensionality
        num_genes = self.model.num_genes
        if predictions.shape[-1] != num_genes:
            raise ValueError(
                f"Prediction dimensionality ({predictions.shape[-1]}) does not match num_genes ({num_genes})"
            )
        
        return predictions.float(), targets.float()

    def configure_optimizers(self):
        """Set up optimizer and optional LR scheduler."""
        weight_decay = float(self.model_utils.get_config('TRAINING.weight_decay', DEFAULT_WEIGHT_DECAY))
        learning_rate = float(self.model_utils.get_config('TRAINING.learning_rate', DEFAULT_LEARNING_RATE))
        
        # Scale LR for the effective device count
        learning_rate = self.model_utils.scale_learning_rate(learning_rate)

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Gradient clipping configuration
        self.trainer.gradient_clip_val = self.model_utils.get_config('TRAINING.gradient_clip_val', DEFAULT_GRADIENT_CLIP)

        # Scheduler
        scheduler_config = self.model_utils.get_scheduler_config(optimizer)
        
        if scheduler_config:
            return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
        else:
            return {'optimizer': optimizer}

    def on_train_epoch_end(self):
        """No-op hook; training data is not accumulated."""
        pass
    
    def on_validation_epoch_end(self):
        """Aggregate validation outputs at epoch end."""
        self._compute_and_log_pcc_metrics('val')
    
    def on_test_epoch_end(self):
        """Aggregate test outputs at epoch end."""
        self._compute_and_log_pcc_metrics('test')

    def _compute_and_log_pcc_metrics(self, phase: str):
        """Compute PCC metrics for a phase and log them."""
        # Map phase to attribute name
        if phase == 'val':
            outputs_attr = 'validation_step_outputs'
        elif phase == 'test':
            outputs_attr = 'test_step_outputs'
        else:
            if self.trainer.is_global_zero:
                self._logger.warning("Unsupported phase '%s'", phase)
            return
        
        if not hasattr(self, outputs_attr):
            if self.trainer.is_global_zero:
                self._logger.warning("No output buffer for phase '%s' (%s)", phase, outputs_attr)
            return
            
        outputs = getattr(self, outputs_attr)
        if not outputs:
            if self.trainer.is_global_zero:
                self._logger.warning("Phase '%s' outputs are empty (likely sanity check)", phase)
            return
        
        try:
            # Gather tensors
            all_predictions = []
            all_targets = []
            
            for output in outputs:
                all_predictions.append(output['predictions'])
                all_targets.append(output['targets'])
            
            # Concatenate along the batch dimension
            predictions = torch.cat(all_predictions, dim=0)  # [N, genes]
            targets = torch.cat(all_targets, dim=0)  # [N, genes]
            
            self._logger.info(f"Phase {phase}: collected {predictions.shape[0]} samples, {predictions.shape[1]} genes")
            
            # Compute PCC metrics (raw token counts -> log2)
            pcc_metrics = self.model_metrics.calculate_comprehensive_pcc_metrics(predictions, targets, apply_log2=True)
            
            # Log metrics
            total_samples = predictions.shape[0]
            for metric_name, value in pcc_metrics.items():
                self.log(f'{phase}_{metric_name}', value, 
                        on_epoch=True, 
                        prog_bar=False, 
                        batch_size=total_samples,
                        sync_dist=True)
            
            # Print detailed results on rank zero
            if self.trainer.is_global_zero:
                phase_loss = float(self.trainer.callback_metrics.get(f'{phase}_loss', 0.0))
                self._logger.info(
                    "Epoch %s %s summary: loss=%.6f pcc10=%.4f pcc50=%.4f pcc200=%.4f mse=%.6f mae=%.6f rvd=%.6f",
                    self.current_epoch,
                    phase.upper(),
                    phase_loss,
                    pcc_metrics['pcc_10'],
                    pcc_metrics['pcc_50'],
                    pcc_metrics['pcc_200'],
                    pcc_metrics['mse'],
                    pcc_metrics['mae'],
                    pcc_metrics['rvd'],
                )

            # Clear buffer
            outputs.clear()
            
        except Exception as e:
            self._logger.error(f"Failed to compute PCC metrics for phase '{phase}': {e}")
            import traceback
            traceback.print_exc()

    def manual_inference_step(self, batch: Dict[str, torch.Tensor], phase: str = 'test') -> Dict[str, torch.Tensor]:
        """Run inference outside of the Lightning trainer while reusing internal logic."""
        model_training_mode = self.model.training
        module_training_mode = self.training

        self.eval()
        self.model.eval()

        with torch.no_grad():
            loss, loss_final, predictions, targets = self._common_step(batch, batch_idx=0, phase=phase)

        if model_training_mode:
            self.model.train()
        if module_training_mode:
            self.train()

        return {
            'loss': loss.detach(),
            'loss_final': loss_final.detach(),
            'predictions': predictions.detach(),
            'targets': targets.detach()
        }

    def on_fit_end(self):
        """Called when training finishes."""
        if not self.trainer.is_global_zero:
            self._logger.info(f"GPU rank {self.trainer.global_rank}: training finished")
            return
        
        self._logger.info("Training finished")





    def on_before_optimizer_step(self, optimizer):
        """Hook executed before each optimizer step."""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), 
            self.trainer.gradient_clip_val
        )
        
        self.log('grad_norm', grad_norm, sync_dist=True)
