"""
多尺度基因VQVAE模块 - Stage 1训练

实现基于生物学多尺度的基因表达Vector-Quantized Variational AutoEncoder。

核心特性：
1. 生物学多尺度分解：Global(1) → Pathway(8) → Module(32) → Individual(200)
2. 共享量化器：符合VAR原始设计，单一codebook，词汇表大小4096
3. 残差重建策略：逐层累积重建，确保信息完整性
4. 独立训练：Stage 1只需要基因表达数据，无需组织学特征

训练目标：
- 学习基因表达的多尺度离散表示
- 为Stage 2提供稳定的量化特征
- 生成用于VAR Transformer的tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any

from .shared_components import (
    SharedVectorQuantizer,
    GlobalEncoder, PathwayEncoder, ModuleEncoder, IndividualEncoder,
    GlobalDecoder, PathwayDecoder, ModuleDecoder, IndividualDecoder,
    ResidualReconstructor,
    MultiScaleDecomposer
)


class MultiScaleGeneVQVAE(nn.Module):
    """
    多尺度基因VQVAE - Stage 1训练的核心模型
    
    架构流程：
    1. 输入: 基因表达 [B, 200]
    2. 多尺度分解: → Global[1], Pathway[8], Module[32], Individual[200]
    3. 分层编码: → 统一128维特征表示
    4. 共享量化: → 离散tokens (从同一codebook)
    5. 分层解码: → 重建各尺度特征
    6. 残差重建: → 最终基因表达 [B, 200]
    
    损失函数：
    - 总重建损失：MSE(final_reconstruction, original_gene_expression)
    - 分层重建损失：各尺度重建的MSE损失
    - VQ损失：所有尺度的Vector Quantization损失
    """
    
    def __init__(
        self,
        vocab_size: int = 4096,
        embed_dim: int = 128,
        beta: float = 0.25,
        hierarchical_loss_weight: float = 0.1,
        vq_loss_weight: float = 0.25
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.beta = beta
        self.hierarchical_loss_weight = hierarchical_loss_weight
        self.vq_loss_weight = vq_loss_weight
        
        # 1. 多尺度分解器
        self.decomposer = MultiScaleDecomposer()
        
        # 2. 多尺度编码器
        self.encoders = nn.ModuleDict({
            'global': GlobalEncoder(embed_dim=embed_dim),
            'pathway': PathwayEncoder(embed_dim=embed_dim),
            'module': ModuleEncoder(embed_dim=embed_dim),
            'individual': IndividualEncoder(embed_dim=embed_dim)
        })
        
        # 3. 共享量化器 (VAR核心设计)
        self.shared_quantizer = SharedVectorQuantizer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            beta=beta
        )
        
        # 4. 多尺度解码器
        self.decoders = nn.ModuleDict({
            'global': GlobalDecoder(embed_dim=embed_dim),
            'pathway': PathwayDecoder(embed_dim=embed_dim),
            'module': ModuleDecoder(embed_dim=embed_dim),
            'individual': IndividualDecoder(embed_dim=embed_dim)
        })
        
        # 5. 残差重建器
        self.reconstructor = ResidualReconstructor()
        
        print(f"🧬 MultiScaleGeneVQVAE初始化:")
        print(f"   词汇表大小: {vocab_size}")
        print(f"   嵌入维度: {embed_dim}")
        print(f"   β参数: {beta}")
        print(f"   分层损失权重: {hierarchical_loss_weight}")
        print(f"   VQ损失权重: {vq_loss_weight}")
    
    def encode(self, gene_expression: torch.Tensor) -> Dict[str, Any]:
        """
        编码阶段：基因表达 → 多尺度tokens
        
        Args:
            gene_expression: [B, 200] - 输入基因表达
            
        Returns:
            包含tokens、量化特征、VQ损失的字典
        """
        # 1. 多尺度分解
        decomposed = self.decomposer(gene_expression)
        
        # 2. 多尺度编码
        encoded = {}
        for scale in ['global', 'pathway', 'module', 'individual']:
            encoded[scale] = self.encoders[scale](decomposed[scale])
        
        # 3. 共享量化
        tokens = {}
        quantized = {}
        vq_losses = []
        
        for scale in ['global', 'pathway', 'module', 'individual']:
            scale_tokens, scale_quantized, scale_vq_loss = self.shared_quantizer(encoded[scale])
            tokens[scale] = scale_tokens
            quantized[scale] = scale_quantized
            vq_losses.append(scale_vq_loss)
        
        total_vq_loss = sum(vq_losses)
        
        return {
            'decomposed': decomposed,
            'encoded': encoded,
            'tokens': tokens,
            'quantized': quantized,
            'vq_loss': total_vq_loss,
            'scale_vq_losses': vq_losses
        }
    
    def decode(self, quantized: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        解码阶段：量化特征 → 重建基因表达
        
        Args:
            quantized: 各尺度的量化特征字典
            
        Returns:
            包含重建结果的字典
        """
        # 1. 多尺度解码
        decoded = {}
        for scale in ['global', 'pathway', 'module', 'individual']:
            decoded[scale] = self.decoders[scale](quantized[scale])
        
        # 2. 残差重建
        reconstruction_result = self.reconstructor(
            decoded['global'], decoded['pathway'],
            decoded['module'], decoded['individual']
        )
        
        return {
            'decoded': decoded,
            'reconstruction_result': reconstruction_result,
            'final_reconstruction': reconstruction_result['final_reconstruction']
        }
    
    def decode_from_tokens(self, tokens: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        从tokens解码：tokens → 重建基因表达
        
        Args:
            tokens: 各尺度的token字典
            
        Returns:
            包含重建结果的字典
        """
        # 1. 从tokens获取量化特征
        quantized = {}
        for scale in ['global', 'pathway', 'module', 'individual']:
            quantized[scale] = self.shared_quantizer.decode(tokens[scale])
        
        # 2. 解码
        return self.decode(quantized)
    
    def forward(self, gene_expression: torch.Tensor) -> Dict[str, Any]:
        """
        前向传播：基因表达 → 重建基因表达
        
        Args:
            gene_expression: [B, 200] - 输入基因表达
            
        Returns:
            包含所有中间结果和损失的字典
        """
        # 1. 编码
        encode_result = self.encode(gene_expression)
        
        # 2. 解码
        decode_result = self.decode(encode_result['quantized'])
        
        # 3. 计算损失
        loss_result = self.compute_losses(
            original=gene_expression,
            encode_result=encode_result,
            decode_result=decode_result
        )
        
        # 合并结果
        result = {
            **encode_result,
            **decode_result,
            **loss_result
        }
        
        return result
    
    def compute_losses(
        self,
        original: torch.Tensor,
        encode_result: Dict[str, Any],
        decode_result: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        计算各种损失函数
        
        Args:
            original: [B, 200] - 原始基因表达
            encode_result: 编码结果
            decode_result: 解码结果
            
        Returns:
            包含各种损失的字典
        """
        # 1. 总重建损失 (最重要)
        final_reconstruction = decode_result['final_reconstruction']
        total_reconstruction_loss = F.mse_loss(final_reconstruction, original)
        
        # 2. 分层重建损失
        decomposed = encode_result['decomposed']
        decoded = decode_result['decoded']
        
        hierarchical_losses = {}
        for scale in ['global', 'pathway', 'module', 'individual']:
            hierarchical_losses[f'{scale}_recon_loss'] = F.mse_loss(
                decoded[scale], decomposed[scale]
            )
        
        total_hierarchical_loss = sum(hierarchical_losses.values())
        
        # 3. VQ损失
        total_vq_loss = encode_result['vq_loss']
        
        # 4. 总损失
        total_loss = (total_reconstruction_loss + 
                     self.hierarchical_loss_weight * total_hierarchical_loss +
                     self.vq_loss_weight * total_vq_loss)
        
        return {
            'total_loss': total_loss,
            'total_reconstruction_loss': total_reconstruction_loss,
            'total_hierarchical_loss': total_hierarchical_loss,
            'total_vq_loss': total_vq_loss,
            **hierarchical_losses
        }
    
    def get_quantization_info(self) -> Dict[str, Any]:
        """获取量化信息，用于监控训练过程"""
        codebook_usage = torch.zeros(self.vocab_size)
        return {
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'codebook_usage': codebook_usage,
            'utilization_rate': 0.0  # 需要在训练过程中更新
        }
    
    def update_codebook_usage(self, tokens: Dict[str, torch.Tensor]) -> float:
        """
        更新codebook使用情况
        
        Args:
            tokens: 各尺度的token字典
            
        Returns:
            codebook利用率
        """
        # 收集所有tokens
        all_tokens = torch.cat([
            tokens[scale].flatten() 
            for scale in ['global', 'pathway', 'module', 'individual']
        ])
        
        # 统计使用的token数量
        unique_tokens = torch.unique(all_tokens)
        utilization_rate = len(unique_tokens) / self.vocab_size
        
        return utilization_rate
    
    @torch.no_grad()
    def generate_random_tokens(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        生成随机tokens，用于测试和验证
        
        Args:
            batch_size: 批次大小
            device: 设备
            
        Returns:
            随机token字典
        """
        tokens = {}
        
        # 生成各尺度的随机tokens
        scale_shapes = {
            'global': (batch_size, 1),
            'pathway': (batch_size, 8),
            'module': (batch_size, 32),
            'individual': (batch_size, 200)
        }
        
        for scale, shape in scale_shapes.items():
            tokens[scale] = torch.randint(
                low=0, high=self.vocab_size,
                size=shape, device=device, dtype=torch.long
            )
        
        return tokens
    
    def save_stage1_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[dict] = None) -> None:
        """
        保存Stage 1训练checkpoint
        
        Args:
            path: 保存路径
            epoch: 当前epoch
            optimizer_state: 优化器状态
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'beta': self.beta,
                'hierarchical_loss_weight': self.hierarchical_loss_weight,
                'vq_loss_weight': self.vq_loss_weight
            },
            'epoch': epoch,
            'stage': 'stage1_vqvae'
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
        
        torch.save(checkpoint, path)
        print(f"💾 Stage 1 checkpoint保存到: {path}")
    
    @classmethod
    def load_stage1_checkpoint(cls, path: str, device: torch.device) -> Tuple['MultiScaleGeneVQVAE', dict]:
        """
        加载Stage 1训练checkpoint
        
        Args:
            path: checkpoint路径
            device: 目标设备
            
        Returns:
            (model, checkpoint_info)
        """
        checkpoint = torch.load(path, map_location=device)
        
        # 重建模型
        model_config = checkpoint['model_config']
        model = cls(**model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        checkpoint_info = {
            'epoch': checkpoint['epoch'],
            'stage': checkpoint['stage'],
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict', None)
        }
        
        print(f"📂 Stage 1 checkpoint从 {path} 加载完成")
        return model, checkpoint_info


class Stage1Trainer:
    """
    Stage 1训练器 - 专门用于训练多尺度基因VQVAE
    
    训练特点：
    1. 只需要基因表达数据
    2. 批次处理spot级别样本
    3. 监控VQ损失和重建精度
    4. 检查codebook利用率
    """
    
    def __init__(
        self,
        model: MultiScaleGeneVQVAE,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        print_freq: int = 100
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.print_freq = print_freq
        
        # 训练统计
        self.epoch_losses = []
        self.codebook_utilizations = []
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器 (只包含基因表达数据)
            epoch: 当前epoch
            
        Returns:
            平均损失字典
        """
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'hierarchical_loss': 0.0,
            'vq_loss': 0.0
        }
        
        epoch_utilizations = []
        num_batches = len(dataloader)
        
        for batch_idx, batch in enumerate(dataloader):
            # 处理不同的数据格式
            if isinstance(batch, (list, tuple)):
                gene_expressions = batch[0]  # 取第一个元素作为基因表达数据
            else:
                gene_expressions = batch
            
            # 移动到设备
            gene_expressions = gene_expressions.to(self.device)  # [B, 200]
            
            # 前向传播
            self.optimizer.zero_grad()
            result = self.model(gene_expressions)
            
            # 反向传播
            total_loss = result['total_loss']
            total_loss.backward()
            self.optimizer.step()
            
            # 累积损失
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['reconstruction_loss'] += result['total_reconstruction_loss'].item()
            epoch_losses['hierarchical_loss'] += result['total_hierarchical_loss'].item()
            epoch_losses['vq_loss'] += result['total_vq_loss'].item()
            
            # 计算codebook利用率
            utilization = self.model.update_codebook_usage(result['tokens'])
            epoch_utilizations.append(utilization)
            
            # 打印进度
            if batch_idx % self.print_freq == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}: "
                      f"Loss={total_loss.item():.4f}, "
                      f"Recon={result['total_reconstruction_loss'].item():.4f}, "
                      f"VQ={result['total_vq_loss'].item():.4f}, "
                      f"Util={utilization:.3f}")
        
        # 计算平均值
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        avg_utilization = sum(epoch_utilizations) / len(epoch_utilizations)
        epoch_losses['codebook_utilization'] = avg_utilization
        
        # 保存统计
        self.epoch_losses.append(epoch_losses)
        self.codebook_utilizations.append(avg_utilization)
        
        return epoch_losses
    
    @torch.no_grad()
    def validate_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """
        验证一个epoch
        
        Args:
            dataloader: 验证数据加载器
            epoch: 当前epoch
            
        Returns:
            验证损失字典
        """
        self.model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'hierarchical_loss': 0.0,
            'vq_loss': 0.0
        }
        
        val_utilizations = []
        num_batches = len(dataloader)
        
        for batch in dataloader:
            # 处理不同的数据格式
            if isinstance(batch, (list, tuple)):
                gene_expressions = batch[0]  # 取第一个元素作为基因表达数据
            else:
                gene_expressions = batch
            
            gene_expressions = gene_expressions.to(self.device)
            
            # 前向传播
            result = self.model(gene_expressions)
            
            # 累积损失
            val_losses['total_loss'] += result['total_loss'].item()
            val_losses['reconstruction_loss'] += result['total_reconstruction_loss'].item()
            val_losses['hierarchical_loss'] += result['total_hierarchical_loss'].item()
            val_losses['vq_loss'] += result['total_vq_loss'].item()
            
            # 计算利用率
            utilization = self.model.update_codebook_usage(result['tokens'])
            val_utilizations.append(utilization)
        
        # 计算平均值
        for key in val_losses:
            val_losses[key] /= num_batches
        
        avg_utilization = sum(val_utilizations) / len(val_utilizations)
        val_losses['codebook_utilization'] = avg_utilization
        
        return val_losses
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'epoch_losses': self.epoch_losses,
            'codebook_utilizations': self.codebook_utilizations,
            'num_epochs_trained': len(self.epoch_losses)
        } 