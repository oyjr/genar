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
        # Gene-related parameters (必需参数在前)
        vocab_size: int,  # 必需参数，从配置中动态传入 (max_gene_count + 1)
        num_genes: int = 200,
        gene_patch_nums: Tuple[int, ...] = (1, 2, 4, 6, 8, 10, 15),
        
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
        """Count the number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
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
        Training forward pass with inter-scale teacher forcing
        
        每个尺度使用前一个尺度的真实基因作为输入，并行预测当前尺度的所有累积基因：
        - 尺度1: 输入[START] → 预测[g1] 
        - 尺度2: 输入[START, 真实g1] → 预测[g1, g2, g3, g4]
        - 尺度3: 输入[START, 真实g1, 真实g2, 真实g3, 真实g4] → 预测[g1, g2, ..., g16]
        """
        B = condition_embed.shape[0]
        device = condition_embed.device
        
        # Condition dropout for classifier-free guidance
        if self.training and self.cond_drop_rate > 0:
            mask = torch.rand(B, device=device) < self.cond_drop_rate
            condition_embed = condition_embed * (~mask).float().unsqueeze(1)
        
        # Build multi-scale input sequences with inter-scale teacher forcing
        input_sequences = []
        target_sequences = []
        scale_indicators = []
        
        for scale_idx, gene_count in enumerate(self.cumulative_gene_counts):
            # Current scale's target genes (cumulative)
            scale_target = target_genes[:, :gene_count]  # [B, gene_count]
            
            start_token = torch.zeros(B, 1, dtype=torch.long, device=device)
            
            if scale_idx == 0:
                # 第一个尺度：只用START token
                # 输入: [START] → 预测: [g1]
                scale_input = start_token  # [B, 1]
            else:
                # 其他尺度：使用前一个尺度的真实基因值 (Teacher Forcing)
                # 输入: [START, 真实g1, ..., 真实g_prev] → 预测: [g1, ..., g_curr]
                prev_gene_count = self.cumulative_gene_counts[scale_idx - 1]
                prev_genes = target_genes[:, :prev_gene_count]  # 前一个尺度的真实基因
                scale_input = torch.cat([start_token, prev_genes], dim=1)  # [B, prev_gene_count + 1]
            
            input_sequences.append(scale_input)
            target_sequences.append(scale_target)
            
            # Scale indicators for each token in input sequence
            scale_indicators.extend([scale_idx] * scale_input.shape[1])
        
        # Concatenate all sequences
        full_input = torch.cat(input_sequences, dim=1)  # [B, total_input_length]
        full_target = torch.cat(target_sequences, dim=1)  # [B, total_target_length]
        
        # Token embeddings
        input_embeds = self.gene_embedding(full_input)  # [B, total_input_length, embed_dim]
        
        # Add position embeddings
        seq_len = full_input.shape[1]
        pos_embeds = self.pos_embedding[:, :seq_len, :]
        input_embeds = input_embeds + pos_embeds
        
        # Add scale embeddings
        scale_ids = torch.tensor(scale_indicators, dtype=torch.long, device=device)
        scale_embeds = self.scale_embedding(scale_ids).unsqueeze(0).expand(B, -1, -1)
        input_embeds = input_embeds + scale_embeds
        
        # Create causal mask for multi-scale sequences
        sequence_lengths = [seq.shape[1] for seq in input_sequences]
        causal_mask = self.create_multiscale_causal_mask(sequence_lengths).to(device)
        
        # Transformer processing
        hidden_states = input_embeds
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, condition_embed, causal_mask)
        
        # Output predictions - 需要为每个尺度生成对应数量的预测
        hidden_states = self.head_norm(hidden_states, condition_embed)
        
        # 为每个尺度单独处理预测
        loss_logits = []
        loss_targets = []
        all_predictions = []
        
        input_start_idx = 0
        for scale_idx, (input_seq, target_seq) in enumerate(zip(input_sequences, target_sequences)):
            input_len = input_seq.shape[1]   # 输入序列长度
            target_len = target_seq.shape[1] # 目标序列长度
            
            # 获取当前尺度的hidden states
            scale_hidden = hidden_states[:, input_start_idx:input_start_idx+input_len, :]  # [B, input_len, embed_dim]
            
            if scale_idx == 0:
                # 第一个尺度：从START token预测1个基因
                # 输入: [START] → 预测: [g1]
                pred_hidden = scale_hidden[:, 0:1, :]  # [B, 1, embed_dim] - START token的hidden state
                pred_logits = self.output_head(pred_hidden)  # [B, 1, vocab_size]
                pred_targets = target_seq  # [B, 1]
            else:
                # 其他尺度：从输入序列预测更多基因
                # 需要扩展hidden states来预测target_len个基因
                
                # 方法：使用最后一个token的hidden state，通过位置编码扩展到target_len个预测
                last_hidden = scale_hidden[:, -1:, :]  # [B, 1, embed_dim] - 最后一个输入token
                
                # 扩展到target_len个位置
                expanded_hidden = last_hidden.expand(B, target_len, -1)  # [B, target_len, embed_dim]
                
                # 添加目标位置的位置编码
                target_pos_embeds = self.pos_embedding[:, :target_len, :]
                expanded_hidden = expanded_hidden + target_pos_embeds
                
                # 添加当前尺度的scale embedding
                scale_embed = self.scale_embedding(torch.tensor(scale_idx, device=device))
                scale_embeds = scale_embed.unsqueeze(0).unsqueeze(0).expand(B, target_len, -1)
                expanded_hidden = expanded_hidden + scale_embeds
                
                # 预测所有目标基因
                pred_logits = self.output_head(expanded_hidden)  # [B, target_len, vocab_size]
                pred_targets = target_seq  # [B, target_len]
            
            # 收集损失计算数据
            loss_logits.append(pred_logits.reshape(-1, self.vocab_size))
            loss_targets.append(pred_targets.reshape(-1))
            
            # 收集预测结果
            scale_predictions = pred_logits.argmax(dim=-1)  # [B, target_len]
            all_predictions.append(scale_predictions)
            
            input_start_idx += input_len
        
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
        
        # 提取最终尺度的基因预测（最后一个尺度包含所有200个基因）
        final_scale_predictions = all_predictions[-1]  # [B, 200]
        
        # 验证最后一个尺度确实包含200个基因
        assert final_scale_predictions.shape[1] == self.num_genes, f"最后尺度基因数量({final_scale_predictions.shape[1]})必须等于目标基因数量({self.num_genes})"
        
        return {
            'loss': loss,
            'logits': all_logits.view(B, -1, self.vocab_size),  # 确保输出logits用于期望值损失 [B, total_predictions, vocab_size]
            'predictions': final_scale_predictions,  # 最终200个基因预测 [B, 200]
            'all_scale_predictions': all_predictions,  # 所有尺度的预测结果
            'accuracy': accuracy,
            'perplexity': perplexity,
            'top5_accuracy': top5_accuracy,
            'full_target': all_targets.view(B, -1)  # 确保target维度匹配 [B, total_predictions]
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
        Inference with inter-scale autoregressive generation
        
        每个尺度使用前一个尺度的预测基因作为输入，并行预测当前尺度的所有累积基因：
        - 尺度1: 输入[START] → 预测[g1]
        - 尺度2: 输入[START, 预测g1] → 预测[g1, g2, g3, g4]  
        - 尺度3: 输入[START, 预测g1, 预测g2, 预测g3, 预测g4] → 预测[g1, g2, ..., g16]
        """
        B = condition_embed.shape[0]
        device = condition_embed.device
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # 逐尺度生成，尺度间自回归
        predicted_genes = None
        
        for scale_idx, gene_count in enumerate(self.cumulative_gene_counts):
            logger.info(f"🔄 推理尺度 {scale_idx + 1}/{self.num_scales}, 目标基因数: {gene_count}")
            
            start_token = torch.zeros(B, 1, dtype=torch.long, device=device)
            
            if scale_idx == 0:
                # 第一个尺度：输入[START] → 预测[g1]
                scale_input = start_token  # [B, 1]
                logger.info(f"📝 尺度{scale_idx + 1}: 输入形状 {scale_input.shape}")
            else:
                # 其他尺度：使用前一个尺度的预测结果
                # 输入: [START, 预测g1, ..., 预测g_prev] → 预测: [g1, ..., g_curr]
                prev_gene_count = self.cumulative_gene_counts[scale_idx - 1]
                
                if predicted_genes is None:
                    logger.error(f"❌ 尺度{scale_idx + 1}: predicted_genes为None!")
                    raise RuntimeError(f"predicted_genes为None在尺度{scale_idx + 1}")
                
                # 使用前一个尺度的预测基因
                prev_genes = predicted_genes[:, :prev_gene_count]  # [B, prev_gene_count]
                scale_input = torch.cat([start_token, prev_genes], dim=1)  # [B, prev_gene_count + 1]
                
                logger.info(f"📝 尺度{scale_idx + 1}: 输入形状 {scale_input.shape}, 使用前{prev_gene_count}个预测基因")
            
            # Token embeddings
            input_embeds = self.gene_embedding(scale_input)  # [B, seq_len, embed_dim]
            
            # Add position embeddings
            seq_len = scale_input.shape[1]
            pos_embeds = self.pos_embedding[:, :seq_len, :]
            input_embeds = input_embeds + pos_embeds
            
            # Add scale embeddings
            scale_embed = self.scale_embedding(torch.tensor(scale_idx, device=device))
            scale_embeds = scale_embed.unsqueeze(0).unsqueeze(0).expand(B, seq_len, -1)
            input_embeds = input_embeds + scale_embeds
            
            # Transformer处理（推理时不使用复杂的多尺度掩码）
            hidden_states = input_embeds
            for block in self.transformer_blocks:
                hidden_states = block(hidden_states, condition_embed, None)
            
            # 输出预测
            hidden_states = self.head_norm(hidden_states, condition_embed)
            
            # 并行预测当前尺度的所有基因
            if scale_idx == 0:
                # 第一个尺度：从START token预测1个基因
                pred_hidden = hidden_states[:, 0:1, :]  # [B, 1, embed_dim]
                gene_logits = self.output_head(pred_hidden)  # [B, 1, vocab_size]
                
                if temperature != 1.0 or top_k is not None or top_p is not None:
                    scale_predictions = self._sample_next_token(gene_logits.squeeze(1), temperature, top_k, top_p)
                    scale_predictions = scale_predictions.unsqueeze(1)  # [B, 1]
                else:
                    scale_predictions = gene_logits.argmax(dim=-1)  # [B, 1]
            else:
                # 其他尺度：从输入序列预测gene_count个基因
                # 使用最后一个token的hidden state，扩展到gene_count个预测
                
                last_hidden = hidden_states[:, -1:, :]  # [B, 1, embed_dim] - 最后一个输入token
                
                # 扩展到gene_count个位置
                expanded_hidden = last_hidden.expand(B, gene_count, -1)  # [B, gene_count, embed_dim]
                
                # 添加目标位置的位置编码
                target_pos_embeds = self.pos_embedding[:, :gene_count, :]
                expanded_hidden = expanded_hidden + target_pos_embeds
                
                # 添加当前尺度的scale embedding
                scale_embeds = scale_embed.unsqueeze(0).unsqueeze(0).expand(B, gene_count, -1)
                expanded_hidden = expanded_hidden + scale_embeds
                
                # 预测所有基因
                gene_logits = self.output_head(expanded_hidden)  # [B, gene_count, vocab_size]
                
                if temperature != 1.0 or top_k is not None or top_p is not None:
                    scale_predictions = self._sample_multiple_tokens(gene_logits, temperature, top_k, top_p)
                else:
                    scale_predictions = gene_logits.argmax(dim=-1)  # [B, gene_count]
            
            # 更新已预测的基因
            predicted_genes = scale_predictions
            
            logger.info(f"✅ 尺度 {scale_idx + 1} 完成: 预测了 {predicted_genes.shape[1]} 个基因")
        
        # 最终预测（最后一个尺度包含所有200个基因）
        final_predictions = predicted_genes[:, :self.num_genes].float()
        
        logger.info(f"🎯 最终预测形状: {final_predictions.shape}, 期望: [B, {self.num_genes}]")
        
        return {
            'predictions': final_predictions,  # [B, num_genes] - 最终基因表达预测
            'token_predictions': predicted_genes[:, :self.num_genes],  # [B, num_genes] - token IDs
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
            vocab_size=checkpoint['vocab_size'],
            num_genes=checkpoint['num_genes'],
            gene_patch_nums=checkpoint['gene_patch_nums'],
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

    def enable_kv_cache(self):
        """Enable KV caching for all transformer blocks during inference"""
        for block in self.transformer_blocks:
            block.enable_kv_cache(True)
    
    def disable_kv_cache(self):
        """Disable KV caching for all transformer blocks during training"""
        for block in self.transformer_blocks:
            block.enable_kv_cache(False)


# Backward compatibility alias
VARST = MultiScaleGeneVAR