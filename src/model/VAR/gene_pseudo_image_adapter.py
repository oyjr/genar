"""
基因伪图像适配器

负责基因表达向量与VAR期望的图像格式之间的转换
核心功能：
1. 基因表达向量 [B, 196] -> 14×14 -> 插值上采样 -> [B, 1, 224, 224] (VQVAE标准输入)
2. 伪图像 [B, 1, 224, 224] -> 下采样 -> 14×14 -> 基因表达向量 [B, 196]
3. 数据验证和标准化

🔧 新解决方案：14×14→224×224插值上采样
- 196基因完美匹配14×14（100%空间利用率）
- 使用最近邻插值扩展到224×224（VQVAE标准输入）
- 每个基因值复制到16×16区域，保持信息密度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import math


class GenePseudoImageAdapter(nn.Module):
    """
    基因伪图像适配器 - 插值上采样版本
    
    将196基因表达向量通过14×14中间表示插值上采样到224×224，满足VQVAE需求
    
    关键改进：
    - 196基因 → 14×14（完美匹配，100%空间利用率）
    - 14×14 → 插值上采样 → 224×224（VQVAE标准输入）
    - 每个基因值扩散到16×16区域，保持信息连续性
    - 反向：224×224 → 下采样 → 14×14 → 196基因
    """
    
    def __init__(
        self,
        num_genes: int = 196,
        intermediate_size: int = 14,  # 🔧 中间14×14表示
        target_image_size: int = 64,  # 🔧 改回64×64输出
        normalize_method: str = 'layer_norm',
        eps: float = 1e-6
    ):
        """
        初始化基因伪图像适配器 - 插值上采样版本
        
        Args:
            num_genes: 基因数量（固定196）
            intermediate_size: 中间表示大小（14×14=196）
            target_image_size: 目标图像大小（64×64，VQVAE标准）
            normalize_method: 标准化方法 ('layer_norm', 'batch_norm', 'none')
            eps: 数值稳定性参数
        """
        super().__init__()
        
        self.num_genes = num_genes
        self.intermediate_size = intermediate_size
        self.target_image_size = target_image_size
        self.normalize_method = normalize_method
        self.eps = eps
        
        # 验证196基因与14×14的完美匹配
        intermediate_positions = intermediate_size * intermediate_size
        if num_genes != intermediate_positions:
            raise ValueError(
                f"基因数量 {num_genes} 必须等于中间表示位置数 {intermediate_size}^2 = {intermediate_positions}"
            )
        
        # 计算上采样倍数
        self.upsampling_factor = target_image_size // intermediate_size
        
        print(f"🧬 初始化基因伪图像适配器 (插值上采样版本):")
        print(f"   - 基因数量: {num_genes}")
        print(f"   - 中间表示: {intermediate_size}×{intermediate_size}")
        print(f"   - 目标图像尺寸: {target_image_size}×{target_image_size}")
        print(f"   - 上采样倍数: {self.upsampling_factor}×")
        print(f"   - 空间利用率: 100% (完美匹配)")
        print(f"   - 标准化方法: {normalize_method}")
        if normalize_method == 'none':
            print(f"   - ⚠️ LayerNorm已禁用，保持log2转换基因表达的原始数值范围")
        print(f"   - ✅ 196基因模式：14×14 → 插值上采样 → {target_image_size}×{target_image_size}")
        
        # 初始化标准化层（只对实际基因数量进行标准化）
        if normalize_method == 'layer_norm':
            self.norm_layer = nn.LayerNorm(num_genes, eps=eps)
        elif normalize_method == 'batch_norm':
            self.norm_layer = nn.BatchNorm1d(num_genes, eps=eps)
        else:
            self.norm_layer = nn.Identity()
    
    def _apply_normalization(self, gene_expression: torch.Tensor) -> torch.Tensor:
        """应用标准化到基因表达向量"""
        if self.normalize_method == 'none':
            return gene_expression
        elif self.normalize_method == 'batch_norm':
            # BatchNorm1d 期望 [B, C] 或 [B, C, L]
            return self.norm_layer(gene_expression)
        else:  # layer_norm
            # LayerNorm 期望 [..., normalized_shape]
            return self.norm_layer(gene_expression)
    
    def _apply_denormalization(self, gene_expression: torch.Tensor) -> torch.Tensor:
        """应用反标准化到基因表达向量"""
        # 🔧 修复：在验证和推理阶段需要正确的反标准化
        # 但由于LayerNorm的参数在训练过程中会变化，这里暂时直接返回
        # 正确的做法是保存标准化的统计信息或使用可逆的标准化
        return gene_expression
    
    def genes_to_pseudo_image(self, gene_expression: torch.Tensor) -> torch.Tensor:
        """
        将基因表达向量转换为伪图像 (插值上采样版本)
        
        流程: [B, 196] → reshape → [B, 1, 14, 14] → 插值上采样 → [B, 1, 224, 224]
        
        Args:
            gene_expression: [B, num_genes] - 基因表达向量 (196个基因)
            
        Returns:
            torch.Tensor: [B, 1, target_image_size, target_image_size] - 伪图像 (单通道，224×224)
        """
        B, num_genes = gene_expression.shape
        
        # 验证基因数量
        if num_genes != self.num_genes:
            raise ValueError(f"期望基因数量: {self.num_genes}, 得到: {num_genes}")
        
        # 🔧 修复：LayerNorm已禁用，保持log2基因表达的原始数值范围
        # 不再区分训练/验证模式，始终保持数据的原始分布
        normalized_genes = gene_expression
        
        # Step 1: 将196基因重塑为14×14中间表示
        # [B, 196] → [B, 1, 14, 14]
        intermediate_image = normalized_genes.view(
            B, 1, self.intermediate_size, self.intermediate_size
        ).contiguous()
        
        # Step 2: 插值上采样到224×224
        # [B, 1, 14, 14] → [B, 1, 224, 224]
        upsampled_image = F.interpolate(
            intermediate_image,
            size=(self.target_image_size, self.target_image_size),
            mode='nearest',  # 最近邻插值，每个基因值复制到16×16区域
            align_corners=None
        )
        
        return upsampled_image
    
    def pseudo_image_to_genes(self, pseudo_image: torch.Tensor) -> torch.Tensor:
        """
        将伪图像转换回基因表达向量 (插值上采样版本)
        
        流程: [B, 1, 224, 224] → 下采样 → [B, 1, 14, 14] → reshape → [B, 196]
        
        Args:
            pseudo_image: [B, 1, target_image_size, target_image_size] - 伪图像 (单通道，224×224)
            
        Returns:
            torch.Tensor: [B, num_genes] - 基因表达向量 (196个基因)
        """
        B, C, H, W = pseudo_image.shape
        
        # 验证输入形状
        if C != 1:
            raise ValueError(f"期望单通道输入，得到: {C}")
        if H != self.target_image_size or W != self.target_image_size:
            raise ValueError(f"期望图像尺寸: {self.target_image_size}x{self.target_image_size}, 得到: {H}x{W}")
        
        # Step 1: 下采样到14×14中间表示
        # [B, 1, 224, 224] → [B, 1, 14, 14]
        downsampled_image = F.interpolate(
            pseudo_image,
            size=(self.intermediate_size, self.intermediate_size),
            mode='bilinear',  # 双线性插值进行下采样，保持平滑性
            align_corners=False
        )
        
        # Step 2: 重塑为基因表达向量
        # [B, 1, 14, 14] → [B, 196]
        gene_expression = downsampled_image.view(B, self.num_genes).contiguous()
        
        # 🔧 修复：LayerNorm已禁用，不需要反标准化
        # 直接返回原始基因表达数值范围
        denormalized_genes = gene_expression
        
        return denormalized_genes
    
    def validate_conversion(
        self, 
        test_batch_size: int = 4,
        tolerance: float = 1e-5
    ) -> Dict[str, Any]:
        """
        验证基因表达与伪图像之间的转换准确性（包含padding处理）
        
        Args:
            test_batch_size: 测试batch大小
            tolerance: 数值容差
            
        Returns:
            包含验证结果的字典
        """
        # 🔧 确保在eval模式下进行验证（禁用标准化）
        original_training_mode = self.training
        self.eval()
        
        try:
            # 创建测试数据
            original_genes = torch.randn(test_batch_size, self.num_genes)
            
            with torch.no_grad():
                # 正向转换: 基因 → 单通道伪图像 (with padding)
                pseudo_image = self.genes_to_pseudo_image(original_genes)
                
                # 反向转换: 单通道伪图像 → 基因 (remove padding)
                reconstructed_genes = self.pseudo_image_to_genes(pseudo_image)
                
                # 计算重建误差
                reconstruction_error = torch.abs(original_genes - reconstructed_genes)
                max_error = reconstruction_error.max().item()
                mean_error = reconstruction_error.mean().item()
                
                # 验证形状
                pseudo_shape_correct = pseudo_image.shape == (test_batch_size, 1, self.target_image_size, self.target_image_size)
                gene_shape_correct = reconstructed_genes.shape == (test_batch_size, self.num_genes)
                
                # 验证padding区域
                padding_region = pseudo_image.view(test_batch_size, -1)[:, self.num_genes:]
                padding_zeros = torch.allclose(padding_region, torch.zeros_like(padding_region), atol=tolerance)
                
                return {
                    'conversion_successful': max_error < tolerance,
                    'max_reconstruction_error': max_error,
                    'mean_reconstruction_error': mean_error,
                    'pseudo_image_shape_correct': pseudo_shape_correct,
                    'gene_shape_correct': gene_shape_correct,
                    'padding_preserved': padding_zeros,
                    'original_genes_shape': original_genes.shape,
                    'pseudo_image_shape': pseudo_image.shape,
                    'reconstructed_genes_shape': reconstructed_genes.shape,
                    'num_genes': self.num_genes,
                    'target_image_size': self.target_image_size,
                    'upsampling_factor': self.upsampling_factor,
                    'space_utilization': 1.0
                }
        finally:
            # 恢复原始训练状态
            self.train(original_training_mode)

    def get_conversion_info(self) -> dict:
        """获取转换配置信息"""
        return {
            'num_genes': self.num_genes,
            'target_image_size': self.target_image_size,
            'intermediate_size': self.intermediate_size,
            'upsampling_factor': self.upsampling_factor,
            'space_utilization': 1.0,
            'normalize_method': self.normalize_method,
            'use_padding': False
        }
    
    def forward(self, x: torch.Tensor, direction: str = 'gene_to_image') -> torch.Tensor:
        """
        前向传播 (支持双向转换)
        
        Args:
            x: 输入张量
            direction: 转换方向 ('gene_to_image' 或 'image_to_gene')
        
        Returns:
            转换后的张量
        """
        if direction == 'gene_to_image':
            return self.genes_to_pseudo_image(x)
        elif direction == 'image_to_gene':
            return self.pseudo_image_to_genes(x)
        else:
            raise ValueError(f"不支持的转换方向: {direction}")

    def reverse_transform(self, pseudo_images: torch.Tensor) -> torch.Tensor:
        """
        从伪图像反向转换为基因表达
        
        Args:
            pseudo_images: 伪图像 [B, 1, target_size, target_size]
            
        Returns:
            基因表达 [B, num_genes]
        """
        B = pseudo_images.shape[0]
        
        # 确保输入格式正确
        if pseudo_images.dim() != 4 or pseudo_images.shape[1] != 1:
            raise ValueError(f"期望输入形状为 [B, 1, H, W]，得到 {pseudo_images.shape}")
        
        # 使用现有的pseudo_image_to_genes方法
        return self.pseudo_image_to_genes(pseudo_images) 