"""
基因VAR Transformer模块

实现基于VAR架构的条件基因表达生成模型。

核心特性：
1. 条件处理器：处理组织学特征和空间坐标
2. VAR Transformer：自回归生成基因tokens
3. Next Token Prediction：标准的自回归语言模型训练

架构流程：
1. 条件信息：组织学特征[1024] + 空间坐标[2] → 条件嵌入[640]
2. VAR生成：条件嵌入 + 历史tokens → 下一个token预测
3. 损失计算：交叉熵损失 (next token prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List
import math
import os
from tqdm import tqdm


class ConditionProcessor(nn.Module):
    """
    条件处理器 - 处理组织学特征和空间坐标
    
    功能：
    1. 组织学特征处理：1024维 → 512维
    2. 空间坐标处理：2维 → 128维 (位置编码)
    3. 条件融合：512 + 128 = 640维条件嵌入
    """
    
    def __init__(
        self,
        histology_dim: int = 1024,
        spatial_dim: int = 2,
        histology_hidden_dim: int = 512,
        spatial_hidden_dim: int = 128,
        condition_embed_dim: int = 640
    ):
        super().__init__()
        
        self.histology_dim = histology_dim
        self.spatial_dim = spatial_dim
        self.condition_embed_dim = condition_embed_dim
        
        # 组织学特征处理器
        self.histology_processor = nn.Sequential(
            nn.LayerNorm(histology_dim),
            nn.Linear(histology_dim, histology_hidden_dim),
            nn.ReLU(),
            nn.Linear(histology_hidden_dim, histology_hidden_dim),
            nn.LayerNorm(histology_hidden_dim)
        )
        
        # 空间坐标处理器 (位置编码)
        self.spatial_processor = nn.Sequential(
            nn.Linear(spatial_dim, spatial_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(spatial_hidden_dim // 2, spatial_hidden_dim),
            nn.LayerNorm(spatial_hidden_dim)
        )
        
        # 正弦余弦位置编码 (可选)
        self.use_sincos_pos = True
        if self.use_sincos_pos:
            # 为2D坐标创建正弦余弦编码
            self.pos_encoding_dim = spatial_hidden_dim // 2
            div_term = torch.exp(torch.arange(0, self.pos_encoding_dim, 2).float() * 
                               (-math.log(10000.0) / self.pos_encoding_dim))
            self.register_buffer('div_term', div_term)
        
        # 最终投影层 (确保总维度为condition_embed_dim)
        total_dim = histology_hidden_dim + spatial_hidden_dim
        if total_dim != condition_embed_dim:
            self.final_projection = nn.Linear(total_dim, condition_embed_dim)
        else:
            self.final_projection = nn.Identity()
    
    def forward(
        self, 
        histology_features: torch.Tensor,  # [B, 1024]
        spatial_coords: torch.Tensor       # [B, 2]
    ) -> torch.Tensor:                     # [B, 640]
        """
        前向传播
        
        Args:
            histology_features: 组织学特征 [B, 1024]
            spatial_coords: 空间坐标 [B, 2]
            
        Returns:
            条件嵌入 [B, 640]
        """
        # 处理组织学特征
        histology_embed = self.histology_processor(histology_features)  # [B, 512]
        
        # 处理空间坐标
        if self.use_sincos_pos:
            # 应用正弦余弦位置编码
            B = spatial_coords.shape[0]
            x_coords = spatial_coords[:, 0:1]  # [B, 1]
            y_coords = spatial_coords[:, 1:2]  # [B, 1]
            
            # 为x和y坐标分别计算正弦余弦编码
            x_pe = torch.zeros(B, self.pos_encoding_dim, device=spatial_coords.device)
            y_pe = torch.zeros(B, self.pos_encoding_dim, device=spatial_coords.device)
            
            x_pe[:, 0::2] = torch.sin(x_coords * self.div_term[None, :])  # 偶数维度sin
            x_pe[:, 1::2] = torch.cos(x_coords * self.div_term[None, :])  # 奇数维度cos
            y_pe[:, 0::2] = torch.sin(y_coords * self.div_term[None, :])
            y_pe[:, 1::2] = torch.cos(y_coords * self.div_term[None, :])
            
            pos_encoding = torch.cat([x_pe, y_pe], dim=1)  # [B, spatial_hidden_dim]
            spatial_embed = self.spatial_processor(spatial_coords) + pos_encoding
        else:
            spatial_embed = self.spatial_processor(spatial_coords)  # [B, 128]
        
        # 融合特征
        condition_features = torch.cat([histology_embed, spatial_embed], dim=1)  # [B, 640]
        condition_embed = self.final_projection(condition_features)  # [B, 640]
        
        return condition_embed


class PositionalEncoding(nn.Module):
    """位置编码 - 为token序列添加位置信息"""
    
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, batch, d_model]
        Returns:
            位置编码后的张量 [seq_len, batch, d_model]
        """
        return x + self.pe[:x.size(0), :]


class GeneVARTransformer(nn.Module):
    """
    基因VAR Transformer
    
    架构：
    1. Token嵌入：将基因tokens转换为嵌入向量
    2. 位置编码：为token序列添加位置信息
    3. 条件融合：将条件信息融入每个Transformer层
    4. Transformer：多层自注意力机制
    5. 输出投影：预测下一个token的概率分布
    """
    
    def __init__(
        self,
        vocab_size: int = 4096,
        embed_dim: int = 640,
        num_heads: int = 8,
        num_layers: int = 12,
        feedforward_dim: int = 2560,
        dropout: float = 0.1,
        max_sequence_length: int = 1500,
        condition_embed_dim: int = 640
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_sequence_length = max_sequence_length
        
        # Token嵌入层
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(embed_dim, max_sequence_length)
        
        # 条件投影 (如果条件维度不等于embed_dim)
        if condition_embed_dim != embed_dim:
            self.condition_projection = nn.Linear(condition_embed_dim, embed_dim)
        else:
            self.condition_projection = nn.Identity()
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出投影层
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """创建因果注意力遮罩"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
    def forward(
        self,
        input_tokens: torch.Tensor,      # [B, seq_len]
        condition_embed: torch.Tensor,   # [B, condition_embed_dim]
        target_tokens: Optional[torch.Tensor] = None  # [B, seq_len] for training
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = input_tokens.shape
        device = input_tokens.device
        
        # Token嵌入
        token_embeds = self.token_embedding(input_tokens)  # [B, seq_len, embed_dim]
        
        # 位置编码 (转换为batch_first格式)
        token_embeds = token_embeds.transpose(0, 1)  # [seq_len, B, embed_dim]
        token_embeds = self.positional_encoding(token_embeds)
        token_embeds = token_embeds.transpose(0, 1)  # [B, seq_len, embed_dim]
        
        # 条件投影和融合
        condition_embed = self.condition_projection(condition_embed)  # [B, embed_dim]
        condition_broadcast = condition_embed.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq_len, embed_dim]
        
        # 融合token嵌入和条件信息
        fused_embeds = token_embeds + condition_broadcast  # [B, seq_len, embed_dim]
        
        # 创建因果遮罩
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Transformer处理
        transformer_output = self.transformer(
            fused_embeds, 
            mask=causal_mask
        )  # [B, seq_len, embed_dim]
        
        # 输出投影
        logits = self.output_projection(transformer_output)  # [B, seq_len, vocab_size]
        
        results = {
            'logits': logits,
            'token_embeds': token_embeds,
            'condition_embed': condition_embed
        }
        
        # 如果提供了目标tokens，计算损失和指标
        if target_tokens is not None:
            # 计算交叉熵损失
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), 
                target_tokens.view(-1),
                ignore_index=-1
            )
            
            # 计算准确率
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == target_tokens).float().mean()
            
            # 计算困惑度
            perplexity = torch.exp(loss)
            
            # 计算top-5准确率
            top5_predictions = logits.topk(5, dim=-1)[1]  # [B, seq_len, 5]
            top5_accuracy = (top5_predictions == target_tokens.unsqueeze(-1)).any(dim=-1).float().mean()
            
            results.update({
                'loss': loss,
                'accuracy': accuracy,
                'perplexity': perplexity,
                'top5_accuracy': top5_accuracy,
                'predictions': predictions,
                'target_tokens': target_tokens
            })
        
        return results
    
    @torch.no_grad()
    def generate(
        self,
        condition_embed: torch.Tensor,   # [B, condition_embed_dim]
        max_length: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:                   # [B, max_length]
        """
        自回归生成token序列
        
        Args:
            condition_embed: 条件嵌入 [B, condition_embed_dim]
            max_length: 生成的最大长度
            temperature: 采样温度
            top_k: Top-k采样
            top_p: Top-p采样
            
        Returns:
            生成的token序列 [B, max_length]
        """
        batch_size = condition_embed.size(0)
        device = condition_embed.device
        
        # 初始化序列 (使用0作为起始token)
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # 前向传播
            outputs = self.forward(generated, condition_embed)
            logits = outputs['logits']  # [B, current_len, vocab_size]
            
            # 获取最后一个位置的logits
            next_token_logits = logits[:, -1, :] / temperature  # [B, vocab_size]
            
            # Top-k filtering
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                top_k_logits, _ = torch.topk(next_token_logits, top_k)
                min_top_k = top_k_logits[:, -1:].expand_as(next_token_logits)
                next_token_logits = torch.where(
                    next_token_logits < min_top_k,
                    torch.full_like(next_token_logits, float('-inf')),
                    next_token_logits
                )
            
            # Top-p filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
            
            # 采样下一个token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            # 拼接到生成序列
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated