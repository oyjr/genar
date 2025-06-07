"""
Two-Stage VAR-ST Model Implementation

This module implements the main two-stage VAR-ST model that combines:
- Stage 1: Multi-scale Gene VQVAE for learning discrete representations
- Stage 2: Gene VAR Transformer for conditional generation

Author: Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from typing import Dict, Optional, Tuple, Union

from .multi_scale_gene_vqvae import MultiScaleGeneVQVAE
from .gene_var_transformer import GeneVARTransformer, ConditionProcessor

logger = logging.getLogger(__name__)


class TwoStageVARST(nn.Module):
    """
    Two-Stage VAR-ST Model for Spatial Transcriptomics
    
    Combines a multi-scale gene VQVAE (Stage 1) with a VAR Transformer (Stage 2)
    for conditional gene expression prediction from histology features.
    
    Architecture:
    - Stage 1: Learn discrete representations of gene expressions using VQVAE
    - Stage 2: Generate discrete tokens conditioned on histology + spatial features
    
    Training Modes:
    - Stage 1: Train only VQVAE, VAR Transformer is not used
    - Stage 2: Train only VAR Transformer, VQVAE is frozen
    - Inference: Use both models in sequence for end-to-end prediction
    """
    
    def __init__(
        self,
        num_genes: int = 200,
        histology_feature_dim: int = 1024,
        spatial_coord_dim: int = 2,
        current_stage: int = 1,
        stage1_ckpt_path: Optional[str] = None,
        # Stage 1 VQVAE parameters
        vqvae_config: Optional[Dict] = None,
        # Stage 2 VAR parameters  
        var_config: Optional[Dict] = None,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.num_genes = num_genes
        self.histology_feature_dim = histology_feature_dim
        self.spatial_coord_dim = spatial_coord_dim
        self.current_stage = current_stage
        self.device = device
        
        # Validate and set configurations
        self.vqvae_config = self._validate_vqvae_config(vqvae_config)
        self.var_config = self._validate_var_config(var_config)
        
        # Initialize Stage 1: Multi-scale Gene VQVAE
        logger.info("Initializing Stage 1: Multi-scale Gene VQVAE")
        self.stage1_vqvae = MultiScaleGeneVQVAE(**self.vqvae_config)
        
        # Initialize Stage 2: Gene VAR Transformer
        logger.info("Initializing Stage 2: Gene VAR Transformer")
        self.stage2_var = GeneVARTransformer(**self.var_config)
        
        # Initialize Condition Processor
        logger.info("Initializing Condition Processor")
        self.condition_processor = ConditionProcessor(
            histology_dim=histology_feature_dim,
            spatial_dim=spatial_coord_dim,
            condition_embed_dim=self.var_config['condition_embed_dim']
        )
        
        # Training stage management
        self.set_training_stage(current_stage, stage1_ckpt_path)
        
        logger.info(f"Two-Stage VAR-ST initialized with stage={current_stage}")
    
    def _validate_vqvae_config(self, config: Optional[Dict]) -> Dict:
        """Validate VQVAE configuration and provide defaults only if None"""
        if config is None:
            return {
                'vocab_size': 4096,
                'embed_dim': 128,
                'beta': 1.0,
                'hierarchical_loss_weight': 0.2,
                'vq_loss_weight': 0.5
            }
        
        required_keys = ['vocab_size', 'embed_dim', 'beta', 'hierarchical_loss_weight', 'vq_loss_weight']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required VQVAE config key: {key}")
        
        return config
    
    def _validate_var_config(self, config: Optional[Dict]) -> Dict:
        """Validate VAR configuration and provide defaults only if None"""
        if config is None:
            return {
                'vocab_size': 4096,
                'embed_dim': 640,
                'num_heads': 8,
                'num_layers': 12,
                'feedforward_dim': 2560,
                'dropout': 0.1,
                'max_sequence_length': 1500,
                'condition_embed_dim': 640
            }
        
        required_keys = ['vocab_size', 'embed_dim', 'num_heads', 'num_layers', 
                        'feedforward_dim', 'dropout', 'max_sequence_length', 'condition_embed_dim']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required VAR config key: {key}")
        
        return config
    
    def _validate_stage2_inputs(self, gene_expression: torch.Tensor, 
                               histology_features: torch.Tensor, 
                               spatial_coords: torch.Tensor):
        """验证Stage 2输入维度"""
        batch_size = gene_expression.size(0)
        
        if gene_expression.shape != (batch_size, self.num_genes):
            raise ValueError(f"Expected gene_expression shape: [{batch_size}, {self.num_genes}], "
                           f"got: {gene_expression.shape}")
        
        if histology_features.shape != (batch_size, self.histology_feature_dim):
            raise ValueError(f"Expected histology_features shape: [{batch_size}, {self.histology_feature_dim}], "
                           f"got: {histology_features.shape}")
        
        if spatial_coords.shape != (batch_size, self.spatial_coord_dim):
            raise ValueError(f"Expected spatial_coords shape: [{batch_size}, {self.spatial_coord_dim}], "
                           f"got: {spatial_coords.shape}")
    
    def _validate_inference_inputs(self, histology_features: torch.Tensor, 
                                  spatial_coords: torch.Tensor):
        """验证推理输入维度"""
        batch_size = histology_features.size(0)
        
        if histology_features.shape[1] != self.histology_feature_dim:
            raise ValueError(f"Expected histology_features dim: {self.histology_feature_dim}, "
                           f"got: {histology_features.shape[1]}")
        
        if spatial_coords.shape != (batch_size, self.spatial_coord_dim):
            raise ValueError(f"Expected spatial_coords shape: [{batch_size}, {self.spatial_coord_dim}], "
                           f"got: {spatial_coords.shape}")
    
    def set_training_stage(self, stage: int, stage1_ckpt_path: Optional[str] = None):
        """
        Set training stage and manage model freezing/unfreezing
        
        Args:
            stage: Training stage (1 or 2)
            stage1_ckpt_path: Path to Stage 1 checkpoint (required for Stage 2)
        """
        if stage not in [1, 2]:
            raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")
        
        self.current_stage = stage
        
        if stage == 1:
            # Stage 1: Train VQVAE only
            logger.info("Setting Stage 1 training mode: VQVAE training")
            self._set_vqvae_trainable(True)
            self._set_var_trainable(False)
            
        elif stage == 2:
            # Stage 2: Load and freeze VQVAE, train VAR Transformer
            if stage1_ckpt_path is None:
                raise ValueError("stage1_ckpt_path is required for Stage 2 training")
            
            logger.info(f"Setting Stage 2 training mode: Loading Stage 1 from {stage1_ckpt_path}")
            self._load_stage1_checkpoint(stage1_ckpt_path)
            self._set_vqvae_trainable(False)
            self._set_var_trainable(True)
    
    def _set_vqvae_trainable(self, trainable: bool):
        """Set VQVAE parameters trainable/frozen"""
        for param in self.stage1_vqvae.parameters():
            param.requires_grad = trainable
        
        if trainable:
            self.stage1_vqvae.train()
            logger.info("Stage 1 VQVAE set to trainable")
        else:
            self.stage1_vqvae.eval()
            logger.info("Stage 1 VQVAE frozen")
    
    def _set_var_trainable(self, trainable: bool):
        """Set VAR Transformer parameters trainable/frozen"""
        for param in self.stage2_var.parameters():
            param.requires_grad = trainable
        
        for param in self.condition_processor.parameters():
            param.requires_grad = trainable
        
        if trainable:
            self.stage2_var.train()
            self.condition_processor.train()
            logger.info("Stage 2 VAR Transformer and Condition Processor set to trainable")
        else:
            self.stage2_var.eval()
            self.condition_processor.eval()
            logger.info("Stage 2 VAR Transformer and Condition Processor frozen")
    
    def _load_stage1_checkpoint(self, ckpt_path: str):
        """Load Stage 1 checkpoint and freeze VQVAE"""
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # Expect standard format with 'model_state_dict' key
        if 'model_state_dict' not in checkpoint:
            raise KeyError("Stage 1 checkpoint must contain 'model_state_dict' key")
        
        self.stage1_vqvae.load_state_dict(checkpoint['model_state_dict'], strict=True)
        logger.info(f"Successfully loaded Stage 1 VQVAE from {ckpt_path}")
    
    def forward(
        self,
        gene_expression: torch.Tensor,
        histology_features: Optional[torch.Tensor] = None,
        spatial_coords: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass based on current training stage
        
        Args:
            gene_expression: [B, num_genes] gene expression data
            histology_features: [B, histology_dim] histology features (Stage 2 only)
            spatial_coords: [B, spatial_dim] spatial coordinates (Stage 2 only)
        
        Returns:
            Dictionary containing losses and predictions based on current stage
        """
        if self.current_stage == 1:
            return self._forward_stage1(gene_expression)
        elif self.current_stage == 2:
            return self._forward_stage2(gene_expression, histology_features, spatial_coords)
        else:
            raise ValueError(f"Invalid stage: {self.current_stage}")
    
    def _forward_stage1(self, gene_expression: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Stage 1 forward: VQVAE training"""
        # VQVAE forward pass
        vqvae_output = self.stage1_vqvae(gene_expression)
        
        # Extract outputs (use correct key names from VQVAE)
        reconstructed = vqvae_output['final_reconstruction']
        vq_loss = vqvae_output['total_vq_loss']
        hierarchical_loss = vqvae_output['total_hierarchical_loss']
        reconstruction_loss = vqvae_output['total_reconstruction_loss']
        total_loss = vqvae_output['total_loss']
        
        return {
            'loss': total_loss,
            'reconstructed': reconstructed,
            'tokens': vqvae_output['tokens'],
            'stage1_losses': {
                'total_loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'hierarchical_loss': hierarchical_loss,
                'vq_loss': vq_loss
            }
        }
    
    def _forward_stage2(
        self,
        gene_expression: torch.Tensor,
        histology_features: torch.Tensor,
        spatial_coords: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Stage 2 forward: VAR Transformer training"""
        if histology_features is None or spatial_coords is None:
            raise ValueError("histology_features and spatial_coords are required for Stage 2")
        
        # 验证输入维度
        self._validate_stage2_inputs(gene_expression, histology_features, spatial_coords)
        
        # Get target tokens from frozen VQVAE
        with torch.no_grad():
            vqvae_output = self.stage1_vqvae(gene_expression)
            tokens = vqvae_output['tokens']  # Dict of tokens for each scale
            
            # Flatten multi-scale tokens into sequence
            token_sequence = []
            for scale in ['global', 'pathway', 'module', 'individual']:
                scale_tokens = tokens[scale].view(tokens[scale].shape[0], -1)  # [B, num_tokens]
                token_sequence.append(scale_tokens)
            
            target_tokens = torch.cat(token_sequence, dim=1)  # [B, total_seq_len]
            target_tokens = target_tokens.detach()
        
        # Process condition information
        condition_embed = self.condition_processor(histology_features, spatial_coords)
        
        # VAR Transformer forward pass
        var_output = self.stage2_var(
            input_tokens=target_tokens,
            condition_embed=condition_embed,
            target_tokens=target_tokens
        )
        
        # Extract VAR loss - Stage 2只使用交叉熵损失
        var_loss = var_output['loss']
        
        output = {
            'loss': var_loss,
            'logits': var_output['logits'],
            'stage2_losses': {
                'var_loss': var_loss,
                'total_loss': var_loss
            }
        }
        
        # 添加VAR Transformer的所有指标
        for metric in ['accuracy', 'perplexity', 'top5_accuracy']:
            if metric in var_output:
                output[metric] = var_output[metric]
        
        return output
    
    def inference(
        self,
        histology_features: torch.Tensor,
        spatial_coords: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-end inference: Generate gene expression from histology + spatial features
        
        Args:
            histology_features: [B, histology_dim] histology features
            spatial_coords: [B, spatial_dim] spatial coordinates
            temperature: Sampling temperature for token generation
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
        
        Returns:
            Dictionary containing predicted gene expression and intermediate results
        """
        if self.current_stage != 2:
            raise ValueError(f"Inference requires Stage 2 setup, but current stage is {self.current_stage}. "
                           "Please set training stage to 2 with proper Stage 1 checkpoint before inference.")
        
        # 验证推理输入维度
        self._validate_inference_inputs(histology_features, spatial_coords)
        
        self.eval()
        
        with torch.no_grad():
            # Process condition information
            condition_embed = self.condition_processor(histology_features, spatial_coords)
            
            # Stage 2: Generate tokens using VAR Transformer
            generated_tokens = self.stage2_var.generate(
                condition_embed=condition_embed,
                max_length=241,  # global(1) + pathway(8) + module(32) + individual(200)
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )  # [B, seq_len]
            
            # Reconstruct multi-scale tokens from flat sequence
            # Sequence order: global(1) + pathway(8) + module(32) + individual(200) = 241
            tokens = {
                'global': generated_tokens[:, 0:1],         # [B, 1]
                'pathway': generated_tokens[:, 1:9],        # [B, 8]
                'module': generated_tokens[:, 9:41],        # [B, 32]
                'individual': generated_tokens[:, 41:241]   # [B, 200]
            }
            
            # Stage 1: Decode tokens to gene expression using VQVAE
            decoded_output = self.stage1_vqvae.decode_from_tokens(tokens)
            
            predicted_gene_expression = decoded_output['final_reconstruction']  # [B, num_genes]
        
        return {
            'predicted_gene_expression': predicted_gene_expression,
            'generated_tokens': generated_tokens,
            'multi_scale_tokens': tokens,
            'multi_scale_reconstructions': decoded_output
        }
    
    def save_stage_checkpoint(self, save_path: str, stage: Optional[int] = None):
        """
        Save checkpoint for specific stage
        
        Args:
            save_path: Path to save checkpoint
            stage: Stage to save (default: current stage)
        """
        if stage is None:
            stage = self.current_stage
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if stage == 1:
            # Save only Stage 1 VQVAE
            checkpoint = {
                'model_state_dict': self.stage1_vqvae.state_dict(),
                'stage': 1,
                'config': self.vqvae_config
            }
        elif stage == 2:
            # Save only Stage 2 VAR Transformer
            checkpoint = {
                'model_state_dict': self.stage2_var.state_dict(),
                'stage': 2,
                'config': self.var_config
            }
        else:
            raise ValueError(f"Invalid stage for saving: {stage}")
        
        torch.save(checkpoint, save_path)
        logger.info(f"Stage {stage} checkpoint saved to {save_path}")
    
    def save_complete_model(self, save_path: str):
        """
        Save complete model with both stages for inference
        
        Args:
            save_path: Path to save complete model
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'stage1_state_dict': self.stage1_vqvae.state_dict(),
            'stage2_state_dict': self.stage2_var.state_dict(),
            'condition_processor_state_dict': self.condition_processor.state_dict(),
            'vqvae_config': self.vqvae_config,
            'var_config': self.var_config,
            'model_class': 'TwoStageVARST',
            'num_genes': self.num_genes,
            'histology_feature_dim': self.histology_feature_dim,
            'spatial_coord_dim': self.spatial_coord_dim
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Complete model saved to {save_path}")
    
    @classmethod
    def load_complete_model(cls, ckpt_path: str, device: str = 'cuda') -> 'TwoStageVARST':
        """
        Load complete model for inference
        
        Args:
            ckpt_path: Path to complete model checkpoint
            device: Device to load model on
        
        Returns:
            Loaded TwoStageVARST model ready for inference
        """
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Extract configs and model parameters - strict validation
        required_keys = ['vqvae_config', 'var_config', 'stage1_state_dict', 'stage2_state_dict',
                        'num_genes', 'histology_feature_dim', 'spatial_coord_dim']
        
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"Missing required key in checkpoint: {key}")
        
        vqvae_config = checkpoint['vqvae_config']
        var_config = checkpoint['var_config']
        num_genes = checkpoint['num_genes']
        histology_feature_dim = checkpoint['histology_feature_dim']
        spatial_coord_dim = checkpoint['spatial_coord_dim']
        
        # Create model instance
        model = cls(
            num_genes=num_genes,
            histology_feature_dim=histology_feature_dim,
            spatial_coord_dim=spatial_coord_dim,
            current_stage=1,  # 先设置为Stage 1，避免要求checkpoint
            vqvae_config=vqvae_config,
            var_config=var_config,
            device=device
        )
        
        # Load state dicts
        model.stage1_vqvae.load_state_dict(checkpoint['stage1_state_dict'])
        model.stage2_var.load_state_dict(checkpoint['stage2_state_dict'])
        
        # Load condition processor if available
        if 'condition_processor_state_dict' in checkpoint:
            model.condition_processor.load_state_dict(checkpoint['condition_processor_state_dict'])
        
        # Now set to Stage 2 mode for inference without checkpoint requirement
        model.current_stage = 2
        model._set_vqvae_trainable(False)
        model._set_var_trainable(False)  # For inference, freeze everything
        
        # Set to eval mode
        model.eval()
        
        logger.info(f"Complete model loaded from {ckpt_path}")
        return model
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        stage1_params = sum(p.numel() for p in self.stage1_vqvae.parameters())
        stage2_params = sum(p.numel() for p in self.stage2_var.parameters())
        condition_params = sum(p.numel() for p in self.condition_processor.parameters())
        
        return {
            'current_stage': self.current_stage,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'stage1_parameters': stage1_params,
            'stage2_parameters': stage2_params,
            'condition_processor_parameters': condition_params,
            'vqvae_config': self.vqvae_config,
            'var_config': self.var_config
        }