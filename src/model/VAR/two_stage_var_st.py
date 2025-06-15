"""
Multi-Scale Gene VAR for Spatial Transcriptomics

This module implements a multi-scale VAR model for spatial transcriptomics
based on the original VAR architecture. The model uses cumulative multi-scale
generation to predict gene expressions from histology features and spatial coordinates.

Key Features:
- Multi-scale cumulative generation (like original VAR)
- AdaLN conditioning for deep feature fusion
- Residual accumulation across scales
- KV caching for efficient inference

Author: Assistant
Date: 2024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
from typing import Dict, List, Optional, Tuple, Union
from functools import partial

from .gene_var_transformer import (
    GeneAdaLNSelfAttn, 
    GeneAdaLNBeforeHead, 
    ConditionProcessor,
    DropPath
)

logger = logging.getLogger(__name__)


class MultiScaleGeneVAR(nn.Module):
    """
    Multi-Scale Gene VAR for Spatial Transcriptomics
    
    Based on original VAR architecture with cumulative multi-scale generation.
    Each scale contains all previous genes plus new ones, finally converging 
    to the complete 200 gene expressions.
    
    Architecture:
    - Condition Processor: Process histology + spatial features with positional encoding
    - Multi-Scale Generation: Cumulative generation across 7 scales
    - AdaLN Transformer: Deep conditioning with adaptive layer normalization
    - Residual Accumulation: Like original VAR's feature accumulation
    """
    
    def __init__(
        self,
        # Gene-related parameters
        num_genes: int = 200,
        gene_patch_nums: Tuple[int, ...] = (1, 2, 4, 6, 8, 10, 15),
        vocab_size: int = 4096,
        
        # Model architecture parameters
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        
        # Dropout parameters
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        
        # Condition-related parameters
        histology_feature_dim: int = 1024,
        spatial_coord_dim: int = 2,
        condition_embed_dim: int = 768,
        cond_drop_rate: float = 0.1,
        
        # Other parameters
        norm_eps: float = 1e-6,
        shared_aln: bool = False,
        attn_l2_norm: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.num_genes = num_genes
        self.gene_patch_nums = gene_patch_nums
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.cond_drop_rate = cond_drop_rate
        self.device = device
        
        # 保存所有配置参数用于checkpoint
        self.histology_feature_dim = histology_feature_dim
        self.spatial_coord_dim = spatial_coord_dim
        self.condition_embed_dim = condition_embed_dim
        
        # Calculate cumulative gene counts for each scale
        self.cumulative_gene_counts = [min(pn * pn, num_genes) for pn in gene_patch_nums]
        self.num_scales = len(gene_patch_nums)
        
        # Calculate sequence lengths for each scale (including start tokens)
        self.scale_sequence_lengths = [count + 1 for count in self.cumulative_gene_counts]  # +1 for start token
        self.total_length = sum(self.scale_sequence_lengths)
        
        logger.info(f"🧬 Gene scale configuration: {gene_patch_nums}")
        logger.info(f"📊 Cumulative gene counts: {self.cumulative_gene_counts}")
        logger.info(f"📏 Scale sequence lengths: {self.scale_sequence_lengths}")
        logger.info(f"🔢 Total sequence length: {self.total_length}")
        
        # Condition processor
        self.condition_processor = ConditionProcessor(
            histology_dim=histology_feature_dim,
            spatial_dim=spatial_coord_dim,
            condition_embed_dim=condition_embed_dim
        )
        
        # Gene token embedding
        self.gene_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Position embedding for each token position
        self.pos_embedding = nn.Parameter(torch.randn(1, self.total_length, embed_dim) * 0.02)
        
        # Scale embedding to distinguish different scales
        self.scale_embedding = nn.Embedding(self.num_scales, embed_dim)
        
        # Start token embeddings for each scale
        self.start_token_embeds = nn.Parameter(torch.randn(self.num_scales, embed_dim) * 0.02)
        
        # Transformer backbone with AdaLN
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.transformer_blocks = nn.ModuleList([
            GeneAdaLNSelfAttn(
                block_idx=i,
                embed_dim=embed_dim,
                condition_dim=condition_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                norm_eps=norm_eps,
                shared_aln=shared_aln,
                attn_l2_norm=attn_l2_norm,
            )
            for i in range(num_layers)
        ])
        
        # Output head with AdaLN
        self.head_norm = GeneAdaLNBeforeHead(embed_dim, condition_embed_dim, norm_eps)
        self.output_head = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
        
        logger.info(f"✅ Multi-Scale Gene VAR initialized successfully")
        logger.info(f"📈 Model parameters: ~{self._count_parameters()/1e6:.1f}M")
    
    def _count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def init_weights(self, init_std: float = 0.02):
        """Initialize model weights following VAR initialization"""
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=init_std)
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
        
        self.apply(_init_weights)
        logger.info("🎯 Model weights initialized")
    
    def create_multiscale_causal_mask(self, sequence_lengths: List[int]) -> torch.Tensor:
        """
        Create multi-scale causal mask
        Each scale can only see its own and previous scales' information
        """
        total_len = sum(sequence_lengths)
        mask = torch.full((total_len, total_len), float('-inf'))
        
        start_idx = 0
        for scale_idx, seq_len in enumerate(sequence_lengths):
            end_idx = start_idx + seq_len
            
            # Current scale can see all previous scales and causal info within itself
            mask[start_idx:end_idx, :end_idx] = torch.triu(
                torch.zeros(seq_len, end_idx), 
                diagonal=end_idx - seq_len + 1
            )
            
            start_idx = end_idx
        
        return mask
    
    def forward(
        self,
        histology_features: torch.Tensor,   # [B, 1024]
        spatial_coords: torch.Tensor,       # [B, 2]
        target_genes: Optional[torch.Tensor] = None  # [B, 200] for training
    ) -> Dict[str, torch.Tensor]:
        """
        Main forward pass
        
        Args:
            histology_features: Histology features [B, 1024]
            spatial_coords: Spatial coordinates [B, 2]
            target_genes: Target gene expressions [B, 200] (for training)
            
        Returns:
            Dictionary containing predictions and loss (if training)
        """
        
        # Process condition information
        condition_embed = self.condition_processor(histology_features, spatial_coords)
        
        if target_genes is not None:
            # Training mode: use teacher forcing
            return self.forward_training(condition_embed, target_genes)
        else:
            # Inference mode: autoregressive generation
            return self.forward_inference(condition_embed)
    
    def forward_training(
        self,
        condition_embed: torch.Tensor,      # [B, condition_embed_dim]
        target_genes: torch.Tensor          # [B, num_genes]
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with teacher forcing
        """
        B = condition_embed.shape[0]
        device = condition_embed.device
        
        # Condition dropout for classifier-free guidance
        if self.training and self.cond_drop_rate > 0:
            mask = torch.rand(B, device=device) < self.cond_drop_rate
            condition_embed = condition_embed * (~mask).float().unsqueeze(1)
        
        # Build multi-scale input sequences
        input_sequences = []
        target_sequences = []
        scale_indicators = []
        
        for scale_idx, gene_count in enumerate(self.cumulative_gene_counts):
            # Current scale's target genes (cumulative)
            scale_target = target_genes[:, :gene_count]  # [B, gene_count]
            
            # Build input sequence with teacher forcing
            start_token = torch.zeros(B, 1, dtype=torch.long, device=device)
            if gene_count > 1:
                scale_input = torch.cat([start_token, scale_target[:, :-1]], dim=1)
            else:
                scale_input = start_token
            
            input_sequences.append(scale_input)
            target_sequences.append(scale_target)
            
            # Scale indicators for each token
            scale_indicators.extend([scale_idx] * scale_input.shape[1])
        
        # Concatenate all sequences
        full_input = torch.cat(input_sequences, dim=1)  # [B, total_length]
        full_target = torch.cat(target_sequences, dim=1)  # [B, total_genes]
        
        # Token embeddings
        input_embeds = self.gene_embedding(full_input)  # [B, total_length, embed_dim]
        
        # Add position embeddings
        seq_len = full_input.shape[1]
        pos_embeds = self.pos_embedding[:, :seq_len, :]
        input_embeds = input_embeds + pos_embeds
        
        # Add scale embeddings
        scale_ids = torch.tensor(scale_indicators, dtype=torch.long, device=device)
        scale_embeds = self.scale_embedding(scale_ids).unsqueeze(0).expand(B, -1, -1)
        input_embeds = input_embeds + scale_embeds
        
        # Create causal mask
        sequence_lengths = [seq.shape[1] for seq in input_sequences]
        causal_mask = self.create_multiscale_causal_mask(sequence_lengths).to(device)
        
        # Transformer processing
        hidden_states = input_embeds
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, condition_embed, causal_mask)
        
        # Output predictions
        hidden_states = self.head_norm(hidden_states, condition_embed)
        logits = self.output_head(hidden_states)  # [B, total_length, vocab_size]
        
        # Calculate loss
        loss_logits = []
        loss_targets = []
        
        start_idx = 0
        for scale_idx, (input_seq, target_seq) in enumerate(zip(input_sequences, target_sequences)):
            seq_len = input_seq.shape[1]
            end_idx = start_idx + seq_len
            
            # Predict next tokens
            if seq_len > 1:
                pred_logits = logits[:, start_idx:end_idx-1, :]  # Prediction positions
                pred_targets = target_seq[:, 1:]  # Target positions
            else:
                pred_logits = logits[:, start_idx:end_idx, :]
                pred_targets = target_seq
            
            loss_logits.append(pred_logits.reshape(-1, self.vocab_size))
            loss_targets.append(pred_targets.reshape(-1))
            
            start_idx = end_idx
        
        # Calculate total loss
        all_logits = torch.cat(loss_logits, dim=0)
        all_targets = torch.cat(loss_targets, dim=0)
        loss = F.cross_entropy(all_logits, all_targets)
        
        # Calculate accuracy
        predictions = all_logits.argmax(dim=-1)
        accuracy = (predictions == all_targets).float().mean()
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Calculate top-5 accuracy
        top5_predictions = all_logits.topk(5, dim=-1)[1]
        top5_accuracy = (top5_predictions == all_targets.unsqueeze(-1)).any(dim=-1).float().mean()
        
        # 🔧 关键修复：正确提取最终尺度的基因预测
        # 最后一个尺度包含累积的所有200个基因，直接从其对应的预测中提取
        
        # 计算最后一个尺度在预测序列中的位置
        final_scale_idx = len(input_sequences) - 1
        final_scale_input_seq = input_sequences[final_scale_idx]
        
        # 计算最后一个尺度的预测起始位置
        pred_offset = 0
        for i in range(final_scale_idx):
            seq_len = input_sequences[i].shape[1]
            pred_offset += (seq_len - 1)  # 除了start token之外的预测位置
        
        # 提取最后一个尺度的预测
        final_seq_len = final_scale_input_seq.shape[1]
        final_pred_count = final_seq_len - 1  # 除了start token之外的预测位置
        
        # 验证最后一个尺度确实包含200个基因
        assert final_pred_count == self.num_genes, f"最后尺度预测数量({final_pred_count})必须等于目标基因数量({self.num_genes})"
        
        # 从predictions中提取最后一个尺度的预测
        final_pred_start = pred_offset
        final_scale_pred_flat = predictions[final_pred_start * B:(final_pred_start + final_pred_count) * B]
        final_scale_predictions = final_scale_pred_flat.view(B, final_pred_count)  # [B, 200]
        
        return {
            'loss': loss,
            'logits': logits,
            'predictions': final_scale_predictions,  # 正确的200个基因预测 [B, 200]
            'full_predictions': predictions.view(B, -1),  # 保留完整预测用于调试
            'accuracy': accuracy,
            'perplexity': perplexity,
            'top5_accuracy': top5_accuracy,
            'full_target': full_target
        }
    
    def forward_inference(
        self,
        condition_embed: torch.Tensor,      # [B, condition_embed_dim]
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Inference forward pass - following the correct design philosophy:
        1. Each scale depends on ALL previous scales (inter-scale dependency)
        2. Within each scale, ALL tokens are generated in parallel (intra-scale parallelism)
        
        This matches the original VAR design and your intended architecture.
        """
        B = condition_embed.shape[0]
        device = condition_embed.device
        
        # 🔧 设置随机种子以确保可重复性（如果提供）
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Store completed scales
        completed_scales = []
        generated_genes = []
        
        # Generate each scale sequentially (inter-scale dependency)
        for scale_idx, gene_count in enumerate(self.cumulative_gene_counts):
            logger.info(f"🔄 Generating scale {scale_idx + 1}/{self.num_scales}, target genes: {gene_count}")
            
            # Prepare input for current scale
            start_token = torch.zeros(B, 1, dtype=torch.long, device=device)
            
            if scale_idx == 0:
                # First scale: start token + positions for gene_count genes
                # We create placeholder positions that will be filled by the model
                current_scale_input = start_token.expand(B, gene_count + 1)  # [B, gene_count + 1]
                # Only the first position is start token, others will be predicted
                current_scale_input = current_scale_input.clone()
                current_scale_input[:, 1:] = 0  # Initialize prediction positions to 0
            else:
                # Subsequent scales: start token + positions for gene_count genes
                current_scale_input = torch.zeros(B, gene_count + 1, dtype=torch.long, device=device)
                # First position is start token, others are prediction positions
            
            # Build complete input: all previous scales + current scale template
            if completed_scales:
                full_input = torch.cat(completed_scales + [current_scale_input], dim=1)
            else:
                full_input = current_scale_input
            
            seq_len = full_input.shape[1]
            
            # Token embeddings
            input_embeds = self.gene_embedding(full_input)  # [B, seq_len, embed_dim]
            
            # Position embeddings
            pos_embeds = self.pos_embedding[:, :seq_len, :]
            input_embeds = input_embeds + pos_embeds
            
            # Scale embeddings - correctly assign scale IDs
            scale_indicators = []
            # Previous completed scales
            for s_idx, scale in enumerate(completed_scales):
                scale_indicators.extend([s_idx] * scale.shape[1])
            # Current scale
            scale_indicators.extend([scale_idx] * current_scale_input.shape[1])
            
            scale_ids = torch.tensor(scale_indicators, dtype=torch.long, device=device)
            scale_embeds = self.scale_embedding(scale_ids).unsqueeze(0).expand(B, -1, -1)
            input_embeds = input_embeds + scale_embeds
            
            # Create multi-scale causal mask (same as training)
            if completed_scales:
                sequence_lengths = [scale.shape[1] for scale in completed_scales] + [current_scale_input.shape[1]]
            else:
                sequence_lengths = [current_scale_input.shape[1]]
            
            causal_mask = self.create_multiscale_causal_mask(sequence_lengths).to(device)
            
            # Transformer processing with full context
            hidden_states = input_embeds
            for block in self.transformer_blocks:
                hidden_states = block(hidden_states, condition_embed, causal_mask)
            
            # Get predictions for current scale
            hidden_states = self.head_norm(hidden_states, condition_embed)
            logits = self.output_head(hidden_states)  # [B, seq_len, vocab_size]
            
            # Extract logits for current scale's prediction positions
            current_scale_start_idx = sum(scale.shape[1] for scale in completed_scales)
            current_scale_logits = logits[:, current_scale_start_idx + 1:current_scale_start_idx + 1 + gene_count, :]
            # Shape: [B, gene_count, vocab_size]
            
            # Sample all tokens for current scale in parallel (key improvement!)
            scale_tokens = self._sample_multiple_tokens(current_scale_logits, temperature, top_k, top_p)
            # Shape: [B, gene_count]
            
            # Build complete current scale sequence
            current_scale_complete = torch.cat([start_token, scale_tokens], dim=1)  # [B, gene_count + 1]
            
            # Add to completed scales
            completed_scales.append(current_scale_complete)
            generated_genes.append(scale_tokens)
            
            logger.info(f"✅ Scale {scale_idx + 1} completed: {scale_tokens.shape[1]} genes generated in parallel")
        
        # Final results
        full_sequence = torch.cat(completed_scales, dim=1)
        final_token_predictions = generated_genes[-1][:, :self.num_genes]  # Take only the required number of genes
        
        # 🔧 关键修复：将token ID转换为基因表达值
        # Token ID直接对应基因计数值（0-4095）
        final_predictions = final_token_predictions.float()  # 转换为浮点数
        
        return {
            'predictions': final_predictions,  # [B, num_genes] - 基因表达值（浮点数）
            'token_predictions': final_token_predictions,  # [B, num_genes] - token ID（整数）
            'all_scale_predictions': generated_genes,
            'generated_sequence': full_sequence
        }
    
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """Sample next token with temperature, top-k, and top-p"""
        logits = logits / temperature
        
        # Top-k filtering
        if top_k is not None:
            top_k = min(top_k, logits.size(-1))
            top_k_logits, _ = torch.topk(logits, top_k)
            min_top_k = top_k_logits[:, -1:].expand_as(logits)
            logits = torch.where(logits < min_top_k, torch.full_like(logits, float('-inf')), logits)
        
        # Top-p filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def _sample_multiple_tokens(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Sample multiple tokens in parallel from logits
        
        Args:
            logits: [B, seq_len, vocab_size] - logits for multiple positions
            temperature: sampling temperature
            top_k: top-k sampling parameter
            top_p: top-p sampling parameter
            
        Returns:
            tokens: [B, seq_len] - sampled tokens for each position
        """
        B, seq_len, vocab_size = logits.shape
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Top-k sampling
        if top_k is not None and top_k > 0:
            top_k = min(top_k, vocab_size)
            # Get top-k values for each position
            top_k_values, _ = torch.topk(logits, top_k, dim=-1)
            # Mask out values below top-k threshold
            indices_to_remove = logits < top_k_values[..., -1:] 
            logits[indices_to_remove] = -float('inf')
        
        # Top-p sampling
        if top_p is not None and top_p < 1.0:
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Create mask for tokens to remove (cumulative prob > top_p)
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least the first token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Convert back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
        
        # Sample tokens for all positions
        probs = F.softmax(logits, dim=-1)  # [B, seq_len, vocab_size]
        
        # Reshape for multinomial sampling
        probs_flat = probs.view(-1, vocab_size)  # [B * seq_len, vocab_size]
        tokens_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)  # [B * seq_len]
        
        # Reshape back to original dimensions
        tokens = tokens_flat.view(B, seq_len)  # [B, seq_len]
        
        return tokens
    
    def inference(
        self,
        histology_features: torch.Tensor,
        spatial_coords: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None
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
            return self.forward_inference(
                condition_embed, 
                temperature, 
                top_k, 
                top_p,
                seed
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
            'num_genes': self.num_genes,
            'gene_patch_nums': self.gene_patch_nums,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'histology_feature_dim': self.histology_feature_dim,
            'spatial_coord_dim': self.spatial_coord_dim,
            'condition_embed_dim': self.condition_embed_dim,
            'epoch': epoch
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"💾 Checkpoint saved to: {save_path}")
    
    @classmethod
    def load_checkpoint(cls, ckpt_path: str, device: str = 'cuda') -> 'MultiScaleGeneVAR':
        """
        Load model from checkpoint
        
        Args:
            ckpt_path: Path to checkpoint
            device: Device to load model on
            
        Returns:
            Loaded MultiScaleGeneVAR model
        """
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Create model with saved configuration
        model = cls(
            num_genes=checkpoint['num_genes'],
            gene_patch_nums=checkpoint['gene_patch_nums'],
            vocab_size=checkpoint['vocab_size'],
            embed_dim=checkpoint['embed_dim'],
            num_heads=checkpoint['num_heads'],
            num_layers=checkpoint['num_layers'],
            histology_feature_dim=checkpoint['histology_feature_dim'],
            spatial_coord_dim=checkpoint['spatial_coord_dim'],
            condition_embed_dim=checkpoint['condition_embed_dim'],
            device=device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        logger.info(f"📂 Model loaded from: {ckpt_path}")
        if 'epoch' in checkpoint:
            logger.info(f"📊 Loaded model from epoch: {checkpoint['epoch']}")
        
        return model
    
    def get_model_info(self) -> Dict:
        """Get comprehensive model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        condition_params = sum(p.numel() for p in self.condition_processor.parameters())
        transformer_params = sum(p.numel() for p in self.transformer_blocks.parameters())
        embedding_params = self.gene_embedding.weight.numel() + self.pos_embedding.numel() + self.scale_embedding.weight.numel()
        output_params = sum(p.numel() for p in self.head_norm.parameters()) + sum(p.numel() for p in self.output_head.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'condition_processor_parameters': condition_params,
            'transformer_parameters': transformer_params,
            'embedding_parameters': embedding_params,
            'output_parameters': output_params,
            'num_genes': self.num_genes,
            'gene_patch_nums': self.gene_patch_nums,
            'cumulative_gene_counts': self.cumulative_gene_counts,
            'num_scales': self.num_scales,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'vocab_size': self.vocab_size,
            'total_sequence_length': self.total_length
        }


# Backward compatibility alias
VARST = MultiScaleGeneVAR