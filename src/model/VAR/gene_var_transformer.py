"""
基因VAR Transformer模块 - Stage 2训练

实现基于VAR架构的条件基因表达生成模型。

核心特性：
1. 条件处理器：处理组织学特征和空间坐标
2. VAR Transformer：自回归生成基因tokens
3. 两阶段训练：Stage 2冻结VQVAE，只训练Transformer
4. Next Token Prediction：标准的自回归语言模型训练

架构流程：
1. 条件信息：组织学特征[1024] + 空间坐标[2] → 条件嵌入[640]
2. Token序列：基因tokens[B, 1446] (来自冻结的VQVAE编码)
3. VAR生成：条件嵌入 + 历史tokens → 下一个token预测
4. 损失计算：交叉熵损失 (next token prediction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List
import math
import os
from tqdm import tqdm

from .shared_components import SharedVectorQuantizer
from .multi_scale_gene_vqvae import MultiScaleGeneVQVAE


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
    基因VAR Transformer - Stage 2的核心模型
    
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
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='relu',
            batch_first=False  # 使用(seq_len, batch, embed_dim)格式
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 输出投影层
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """创建因果注意力掩码"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        input_tokens: torch.Tensor,      # [B, seq_len]
        condition_embed: torch.Tensor,   # [B, condition_embed_dim]
        target_tokens: Optional[torch.Tensor] = None  # [B, seq_len] for training
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_tokens: 输入token序列 [B, seq_len]
            condition_embed: 条件嵌入 [B, condition_embed_dim] 
            target_tokens: 目标token序列 [B, seq_len] (训练时使用)
            
        Returns:
            包含logits和loss的字典
        """
        B, seq_len = input_tokens.shape
        device = input_tokens.device
        
        # Token嵌入
        token_embeds = self.token_embedding(input_tokens)  # [B, seq_len, embed_dim]
        token_embeds = token_embeds.transpose(0, 1)        # [seq_len, B, embed_dim]
        
        # 位置编码
        token_embeds = self.positional_encoding(token_embeds)  # [seq_len, B, embed_dim]
        
        # 处理条件信息
        condition_proj = self.condition_projection(condition_embed)  # [B, embed_dim]
        # 扩展条件为记忆序列，用作Transformer的memory
        memory = condition_proj.unsqueeze(0)  # [1, B, embed_dim]
        
        # 创建因果掩码
        tgt_mask = self.create_causal_mask(seq_len, device)  # [seq_len, seq_len]
        
        # Transformer解码
        transformer_output = self.transformer_decoder(
            tgt=token_embeds,           # [seq_len, B, embed_dim]
            memory=memory,              # [1, B, embed_dim]
            tgt_mask=tgt_mask          # [seq_len, seq_len]
        )  # [seq_len, B, embed_dim]
        
        # 输出投影
        logits = self.output_projection(transformer_output)  # [seq_len, B, vocab_size]
        logits = logits.transpose(0, 1)  # [B, seq_len, vocab_size]
        
        result = {'logits': logits}
        
        # 如果提供了目标tokens，计算损失和指标
        if target_tokens is not None:
            # 计算交叉熵损失 (next token prediction)
            # 输入: input_tokens[:-1], 目标: target_tokens[1:]
            shift_logits = logits[:, :-1, :].contiguous()  # [B, seq_len-1, vocab_size]
            shift_labels = target_tokens[:, 1:].contiguous()  # [B, seq_len-1]
            
            # 🔧 Stage 2只使用交叉熵损失，移除其他损失组件
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-1,  # 忽略padding tokens
                reduction='mean'
            )
            
            result['loss'] = loss
            
            # 计算准确率和困惑度
            with torch.no_grad():
                # Token预测准确率
                predictions = torch.argmax(shift_logits, dim=-1)
                valid_mask = (shift_labels != -1)  # 忽略padding
                accuracy = (predictions == shift_labels)[valid_mask].float().mean()
                result['accuracy'] = accuracy
                
                # 🔧 困惑度计算：perplexity = exp(loss)
                # 困惑度衡量模型预测的不确定性，越低越好
                perplexity = torch.exp(loss)
                result['perplexity'] = perplexity
                
                # 🔧 额外指标：top-5准确率
                top5_predictions = torch.topk(shift_logits, k=5, dim=-1)[1]  # [B, seq_len-1, 5]
                shift_labels_expanded = shift_labels.unsqueeze(-1).expand_as(top5_predictions)
                top5_accuracy = (top5_predictions == shift_labels_expanded).any(dim=-1)[valid_mask].float().mean()
                result['top5_accuracy'] = top5_accuracy
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        condition_embed: torch.Tensor,   # [B, condition_embed_dim]
        max_length: int = 1446,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:                   # [B, max_length]
        """
        自回归生成基因tokens
        
        Args:
            condition_embed: 条件嵌入 [B, condition_embed_dim]
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: top-k采样
            top_p: nucleus采样
            
        Returns:
            生成的token序列 [B, max_length]
        """
        B = condition_embed.shape[0]
        device = condition_embed.device
        
        # 初始化序列 (使用特殊的开始token，这里用0)
        generated = torch.zeros(B, 1, dtype=torch.long, device=device)
        
        for step in range(max_length - 1):
            # 前向传播
            outputs = self.forward(generated, condition_embed)
            logits = outputs['logits']  # [B, current_length, vocab_size]
            
            # 获取最后一个位置的logits
            next_token_logits = logits[:, -1, :] / temperature  # [B, vocab_size]
            
            # 应用top-k采样
            if top_k is not None:
                values, indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, indices, values)
            
            # 应用nucleus (top-p)采样
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的tokens
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                for i in range(B):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i][indices_to_remove] = float('-inf')
            
            # 采样下一个token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)  # [B, 1]
            
            # 添加到序列中
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


class Stage2Trainer:
    """
    Stage 2训练器 - 训练基因VAR Transformer
    
    功能：
    1. 冻结Stage 1的VQVAE模型
    2. 训练VAR Transformer进行条件生成
    3. 管理训练循环和验证
    4. 保存和加载checkpoint
    """
    
    def __init__(
        self,
        vqvae_model: MultiScaleGeneVQVAE,
        var_transformer: GeneVARTransformer,
        condition_processor: ConditionProcessor,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        print_freq: int = 100
    ):
        self.device = device
        self.print_freq = print_freq
        
        # 模型组件
        self.vqvae_model = vqvae_model.to(device)
        self.var_transformer = var_transformer.to(device)
        self.condition_processor = condition_processor.to(device)
        
        # 冻结VQVAE参数
        for param in self.vqvae_model.parameters():
            param.requires_grad = False
        self.vqvae_model.eval()
        
        # 优化器 (只优化VAR Transformer和条件处理器)
        trainable_params = list(self.var_transformer.parameters()) + list(self.condition_processor.parameters())
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        
        # 训练统计
        self.epoch_losses = []
        self.epoch_accuracies = []
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器 (包含基因表达、组织学特征、空间坐标)
            epoch: 当前epoch
            
        Returns:
            平均损失和准确率
        """
        self.var_transformer.train()
        self.condition_processor.train()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # 严格验证数据格式
            if isinstance(batch, (list, tuple)):
                if len(batch) < 3:
                    raise ValueError(f"Batch must contain [gene_expressions, histology_features, spatial_coords], "
                                   f"but got {len(batch)} elements")
                gene_expressions = batch[0]
                histology_features = batch[1]
                spatial_coords = batch[2]
            else:
                raise ValueError("Batch must be a tuple/list containing [gene_expressions, histology_features, spatial_coords]. "
                               "Single tensor batches are not supported for Stage 2 training.")
            
            # 移动到设备
            gene_expressions = gene_expressions.to(self.device)
            histology_features = histology_features.to(self.device)
            spatial_coords = spatial_coords.to(self.device)
            
            # 使用冻结的VQVAE编码基因表达为tokens
            with torch.no_grad():
                vqvae_result = self.vqvae_model(gene_expressions)
                tokens = vqvae_result['tokens']  # Dict of tokens for each scale
                
                # 将多尺度tokens展平为序列
                token_sequence = []
                for scale in ['global', 'pathway', 'module', 'individual']:
                    scale_tokens = tokens[scale].view(tokens[scale].shape[0], -1)  # [B, num_tokens]
                    token_sequence.append(scale_tokens)
                
                full_token_sequence = torch.cat(token_sequence, dim=1)  # [B, total_seq_len]
            
            # 处理条件信息
            condition_embed = self.condition_processor(histology_features, spatial_coords)
            
            # 准备输入和目标 (teacher forcing)
            input_tokens = full_token_sequence  # [B, seq_len]
            target_tokens = full_token_sequence  # [B, seq_len] (same for autoregressive training)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.var_transformer(input_tokens, condition_embed, target_tokens)
            
            loss = outputs['loss']
            accuracy = outputs['accuracy']
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.var_transformer.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.condition_processor.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 累积统计
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            
            # 打印进度
            if batch_idx % self.print_freq == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}: "
                      f"Loss={loss.item():.4f}, Accuracy={accuracy.item():.4f}")
        
        # 计算平均值
        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches
        
        # 保存统计
        self.epoch_losses.append(avg_loss)
        self.epoch_accuracies.append(avg_accuracy)
        
        return {'loss': avg_loss, 'accuracy': avg_accuracy}
    
    @torch.no_grad()
    def validate_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """
        验证一个epoch
        
        Args:
            dataloader: 验证数据加载器
            epoch: 当前epoch
            
        Returns:
            验证损失和准确率
        """
        self.var_transformer.eval()
        self.condition_processor.eval()
        
        val_loss = 0.0
        val_accuracy = 0.0
        num_batches = len(dataloader)
        
        for batch in dataloader:
            # 严格验证数据格式 (与训练相同)
            if isinstance(batch, (list, tuple)):
                if len(batch) < 3:
                    raise ValueError(f"Validation batch must contain [gene_expressions, histology_features, spatial_coords], "
                                   f"but got {len(batch)} elements")
                gene_expressions = batch[0]
                histology_features = batch[1]
                spatial_coords = batch[2]
            else:
                raise ValueError("Validation batch must be a tuple/list containing [gene_expressions, histology_features, spatial_coords]. "
                               "Single tensor batches are not supported for Stage 2 validation.")
            
            gene_expressions = gene_expressions.to(self.device)
            histology_features = histology_features.to(self.device)
            spatial_coords = spatial_coords.to(self.device)
            
            # 编码基因tokens
            vqvae_result = self.vqvae_model(gene_expressions)
            tokens = vqvae_result['tokens']
            
            token_sequence = []
            for scale in ['global', 'pathway', 'module', 'individual']:
                scale_tokens = tokens[scale].view(tokens[scale].shape[0], -1)
                token_sequence.append(scale_tokens)
            
            full_token_sequence = torch.cat(token_sequence, dim=1)
            
            # 处理条件
            condition_embed = self.condition_processor(histology_features, spatial_coords)
            
            # 前向传播
            outputs = self.var_transformer(full_token_sequence, condition_embed, full_token_sequence)
            
            val_loss += outputs['loss'].item()
            val_accuracy += outputs['accuracy'].item()
        
        avg_val_loss = val_loss / num_batches
        avg_val_accuracy = val_accuracy / num_batches
        
        return {'loss': avg_val_loss, 'accuracy': avg_val_accuracy}
    
    def save_checkpoint(self, filepath: str, epoch: int, metadata: Optional[Dict] = None):
        """保存Stage 2 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'var_transformer_state_dict': self.var_transformer.state_dict(),
            'condition_processor_state_dict': self.condition_processor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch_losses': self.epoch_losses,
            'epoch_accuracies': self.epoch_accuracies,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, filepath)
        print(f"Stage 2 checkpoint保存至: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict:
        """加载Stage 2 checkpoint - 严格验证"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 严格验证checkpoint完整性
        required_keys = ['var_transformer_state_dict', 'condition_processor_state_dict', 
                        'optimizer_state_dict', 'epoch', 'epoch_losses', 'epoch_accuracies']
        
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"Missing required key in Stage 2 checkpoint: {key}")
        
        self.var_transformer.load_state_dict(checkpoint['var_transformer_state_dict'])
        self.condition_processor.load_state_dict(checkpoint['condition_processor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch_losses = checkpoint['epoch_losses']
        self.epoch_accuracies = checkpoint['epoch_accuracies']
        
        print(f"Stage 2 checkpoint加载: {filepath}, epoch: {checkpoint['epoch']}")
        return checkpoint