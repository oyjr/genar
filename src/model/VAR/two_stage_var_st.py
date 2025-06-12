"""
VAR-ST Model Implementation

This module implements a simplified VAR-ST model for spatial transcriptomics.
The model uses a VAR Transformer to directly predict gene expressions from 
histology features and spatial coordinates.

Author: Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from typing import Dict, Optional, Tuple, Union

from .gene_var_transformer import GeneVARTransformer, ConditionProcessor

logger = logging.getLogger(__name__)


class VARST(nn.Module):
    """
    VAR-ST Model for Spatial Transcriptomics
    
    A simplified single-stage model that uses VAR Transformer to directly
    predict gene expressions from histology features and spatial coordinates.
    
    Architecture:
    - Condition Processor: Process histology + spatial features
    - VAR Transformer: Generate gene expressions using autoregressive modeling
    """
    
    def __init__(
        self,
        num_genes: int = 200,
        histology_feature_dim: int = 1024,
        spatial_coord_dim: int = 2,
        # VAR parameters  
        var_config: Optional[Dict] = None,
        gene_count_mode: str = 'discrete_tokens',  # 🆕 基因计数模式（固定为离散）
        max_gene_count: int = 4095,  # 🆕 最大基因计数值
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.num_genes = num_genes
        self.histology_feature_dim = histology_feature_dim
        self.spatial_coord_dim = spatial_coord_dim
        self.gene_count_mode = 'discrete_tokens'  # 🆕 固定为离散模式
        self.max_gene_count = max_gene_count  # 🆕 最大基因计数
        self.device = device
        
        # Validate and set configurations
        self.var_config = self._validate_var_config(var_config)
        
        # 🆕 根据基因计数模式调整vocab_size（固定为离散模式）
        self.var_config['vocab_size'] = max_gene_count + 1  # +1 for start token
        logger.info(f"🔢 使用离散token模式: vocab_size = {self.var_config['vocab_size']}")
        
        # Initialize Condition Processor
        logger.info("Initializing Condition Processor")
        self.condition_processor = ConditionProcessor(
            histology_dim=histology_feature_dim,
            spatial_dim=spatial_coord_dim,
            condition_embed_dim=self.var_config['condition_embed_dim']
        )
        
        # Initialize VAR Transformer
        logger.info("Initializing VAR Transformer")
        self.var_transformer = GeneVARTransformer(**self.var_config)
        
        logger.info(f"VAR-ST initialized successfully")
    
    def _validate_var_config(self, config: Optional[Dict]) -> Dict:
        """Validate VAR configuration and provide defaults if None"""
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
    
    def forward(
        self,
        histology_features: torch.Tensor,  # [B, 1024]
        spatial_coords: torch.Tensor,      # [B, 2]
        target_genes: Optional[torch.Tensor] = None  # [B, 200] for training
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            histology_features: Histology features [B, 1024]
            spatial_coords: Spatial coordinates [B, 2]  
            target_genes: Target gene expressions [B, 200] (for training)
            
        Returns:
            Dictionary containing predictions and loss (if training)
        """
        
        batch_size = histology_features.size(0)
        
        # Process condition information
        condition_embed = self.condition_processor(histology_features, spatial_coords)
        
        if target_genes is not None:
            # Training/Validation mode: use teacher forcing
            return self._forward_training(condition_embed, target_genes)
        else:
            # Inference mode: autoregressive generation
            return self._forward_inference(condition_embed, batch_size)
    
    def _forward_training(
        self, 
        condition_embed: torch.Tensor,    # [B, embed_dim]
        target_genes: torch.Tensor        # [B, 200]
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with teacher forcing"""
        
        batch_size = condition_embed.size(0)
        device = condition_embed.device
        
        # 离散token模式：target_genes已经是long tensor
        # Prepare input sequence (add start token)
        start_tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        input_sequence = torch.cat([start_tokens, target_genes[:, :-1]], dim=1)  # [B, 200]
        
        # Forward through VAR transformer
        outputs = self.var_transformer(
            input_tokens=input_sequence,
            condition_embed=condition_embed,
            target_tokens=target_genes
        )
        
        # 提取logits并转换为predictions
        logits = outputs['logits']  # [B, 200, vocab_size]
        predictions = logits.argmax(dim=-1)  # [B, 200]
        
        # 返回结果，包含predictions字段用于损失计算
        return {
            'logits': logits,
            'predictions': predictions,
            'loss': outputs['loss'],
            'accuracy': outputs['accuracy'],
            'perplexity': outputs['perplexity'],
            'top5_accuracy': outputs['top5_accuracy']
        }
    
    def _forward_inference(
        self, 
        condition_embed: torch.Tensor,    # [B, embed_dim]
        batch_size: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """Inference forward pass with autoregressive generation"""
        
        # Generate gene expressions
        generated_genes = self.var_transformer.generate(
            condition_embed=condition_embed,
            max_length=self.num_genes,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Remove start token
        generated_genes = generated_genes[:, 1:]  # [B, 200]
        
        return {
            'predictions': generated_genes,
            'generated_sequence': generated_genes
        }
    
    def inference(
        self,
        histology_features: torch.Tensor,
        spatial_coords: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inference mode for gene expression prediction
        
        Args:
            histology_features: Histology features [B, 1024]
            spatial_coords: Spatial coordinates [B, 2]
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            
        Returns:
            Dictionary containing predicted gene expressions
        """
        
        self.eval()
        with torch.no_grad():
            # Process conditions
            condition_embed = self.condition_processor(histology_features, spatial_coords)
            
            # Generate predictions
            return self._forward_inference(
                condition_embed, 
                histology_features.size(0),
                temperature, 
                top_k, 
                top_p
            )
    
    def save_checkpoint(self, save_path: str, epoch: Optional[int] = None):
        """
        Save model checkpoint
        
        Args:
            save_path: Path to save checkpoint
            epoch: Current epoch (optional)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'var_config': self.var_config,
            'num_genes': self.num_genes,
            'histology_feature_dim': self.histology_feature_dim,
            'spatial_coord_dim': self.spatial_coord_dim,
            'epoch': epoch
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to: {save_path}")
    
    @classmethod
    def load_checkpoint(cls, ckpt_path: str, device: str = 'cuda') -> 'VARST':
        """
        Load model from checkpoint
        
        Args:
            ckpt_path: Path to checkpoint
            device: Device to load model on
            
        Returns:
            Loaded VARST model
        """
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Create model with saved configuration
        model = cls(
            num_genes=checkpoint['num_genes'],
            histology_feature_dim=checkpoint['histology_feature_dim'],
            spatial_coord_dim=checkpoint['spatial_coord_dim'],
            var_config=checkpoint['var_config'],
            gene_count_mode=checkpoint['gene_count_mode'],
            max_gene_count=checkpoint['max_gene_count'],
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"Model loaded from: {ckpt_path}")
        if 'epoch' in checkpoint:
            logger.info(f"Loaded model from epoch: {checkpoint['epoch']}")
        
        return model
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        var_params = sum(p.numel() for p in self.var_transformer.parameters())
        condition_params = sum(p.numel() for p in self.condition_processor.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'var_transformer_parameters': var_params,
            'condition_processor_parameters': condition_params,
            'num_genes': self.num_genes,
            'histology_feature_dim': self.histology_feature_dim,
            'spatial_coord_dim': self.spatial_coord_dim,
            'var_config': self.var_config,
            'gene_count_mode': self.gene_count_mode,
            'max_gene_count': self.max_gene_count
        }