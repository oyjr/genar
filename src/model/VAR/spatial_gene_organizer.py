"""
Spatial Gene Expression Organizer for VAR-ST

This module converts scattered spot gene expressions into spatial "images" 
that VAR can process, maintaining the original VAR architecture without modifications.

Author: VAR-ST Team
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class SpatialGeneOrganizer:
    """
    将散乱的spots基因表达重组为规则的空间"图像"
    这是适配VAR的关键组件 - 完全保留原始输入格式
    
    核心功能：
    1. spots基因表达 -> 空间基因表达"图像" (类似RGB图像)
    2. 空间基因表达"图像" -> spots基因表达 (逆向操作)
    
    设计理念：
    - 将基因表达视为"颜色通道"
    - 将空间位置视为"像素位置"
    - 使VAR能够像处理图像一样处理基因表达数据
    """
    
    def __init__(self, target_spatial_size: int = 16, num_genes: int = 200):
        """
        Initialize spatial gene organizer
        
        Args:
            target_spatial_size: Target spatial size for gene expression "image"
                                Creates target_spatial_size × target_spatial_size grid
            num_genes: Number of genes, acts as "color channels"
        """
        self.target_size = target_spatial_size
        self.num_genes = num_genes
        
        print(f"🔧 初始化空间基因组织器:")
        print(f"  - 目标空间尺寸: {target_spatial_size}×{target_spatial_size}")
        print(f"  - 基因数量: {num_genes} (作为颜色通道)")
    
    def spots_to_spatial_gene_image(
        self, 
        gene_expression: torch.Tensor, 
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        核心功能：将spots转换为空间基因"图像"
        
        这个函数实现了从离散spots到连续空间表示的转换，
        使得VAR可以像处理图像一样处理基因表达数据。
        
        Args:
            gene_expression: [B, N, num_genes] - spots的基因表达
            positions: [B, N, 2] - spots的空间位置，范围[0,1]
        
        Returns:
            spatial_gene_image: [B, num_genes, H, W] - 空间基因表达"图像"
            spatial_mask: [B, 1, H, W] - 有效区域掩码
        """
        B, N, G = gene_expression.shape
        H, W = self.target_size, self.target_size
        
        # 初始化空间网格
        spatial_gene_image = torch.zeros(B, G, H, W, device=gene_expression.device, dtype=gene_expression.dtype)
        spatial_mask = torch.zeros(B, 1, H, W, device=gene_expression.device, dtype=gene_expression.dtype)
        spot_count_grid = torch.zeros(B, H, W, device=gene_expression.device, dtype=gene_expression.dtype)
        
        for b in range(B):
            # 将连续位置[0,1]映射到网格坐标[0, H-1], [0, W-1]
            grid_x = (positions[b, :, 0] * (W - 1)).long().clamp(0, W - 1)  # [N]
            grid_y = (positions[b, :, 1] * (H - 1)).long().clamp(0, H - 1)  # [N]
            
            # 聚合到网格：处理多个spots映射到同一网格的情况
            for n in range(N):
                x, y = grid_x[n].item(), grid_y[n].item()
                spatial_gene_image[b, :, y, x] += gene_expression[b, n]  # [G]
                spot_count_grid[b, y, x] += 1
                spatial_mask[b, 0, y, x] = 1
        
        # 归一化：平均同一网格内的多个spots
        # 这确保了空间密度不同的区域具有合理的基因表达值
        for b in range(B):
            for y in range(H):
                for x in range(W):
                    if spot_count_grid[b, y, x] > 1:
                        spatial_gene_image[b, :, y, x] /= spot_count_grid[b, y, x]
        
        return spatial_gene_image, spatial_mask
    
    def spatial_gene_image_to_spots(
        self, 
        spatial_gene_image: torch.Tensor, 
        target_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        逆向操作：从空间基因"图像"提取spots的基因表达
        
        使用双线性插值从连续的空间基因表达图像中提取任意位置的基因表达值。
        这使得模型可以预测任意空间位置的基因表达，而不仅仅是训练时的网格位置。
        
        Args:
            spatial_gene_image: [B, num_genes, H, W] - 空间基因表达"图像"
            target_positions: [B, N, 2] - 目标spots位置，范围[0,1]
        
        Returns:
            gene_expression: [B, N, num_genes] - spots基因表达
        """
        B, G, H, W = spatial_gene_image.shape
        N = target_positions.shape[1]
        
        gene_expression = torch.zeros(B, N, G, device=spatial_gene_image.device, dtype=spatial_gene_image.dtype)
        
        for b in range(B):
            # 对每个目标spot位置进行双线性插值
            for n in range(N):
                x_pos, y_pos = target_positions[b, n]  # [0, 1]范围
                
                # 转换到图像坐标
                x_img = x_pos * (W - 1)
                y_img = y_pos * (H - 1)
                
                # 双线性插值的四个邻近点
                x0, x1 = int(x_img), min(int(x_img) + 1, W - 1)
                y0, y1 = int(y_img), min(int(y_img) + 1, H - 1)
                
                # 插值权重
                wx = x_img - x0
                wy = y_img - y0
                
                # 双线性插值计算
                # gene_expression[b, n] = ∑(weight * spatial_gene_image[b, :, yi, xi])
                gene_expression[b, n] = (
                    (1 - wy) * (1 - wx) * spatial_gene_image[b, :, y0, x0] +
                    (1 - wy) * wx * spatial_gene_image[b, :, y0, x1] +
                    wy * (1 - wx) * spatial_gene_image[b, :, y1, x0] +
                    wy * wx * spatial_gene_image[b, :, y1, x1]
                )
        
        return gene_expression
    
    def generate_default_positions(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        """
        生成默认的规则网格位置
        
        当输入数据没有提供空间位置信息时，生成规则的网格位置。
        这确保了模型即使在没有真实空间信息的情况下也能正常工作。
        
        Args:
            B: Batch size
            N: Number of spots
            device: Target device
        
        Returns:
            positions: [B, N, 2] - 规则网格位置，范围[0,1]
        """
        # 生成√N × √N的规则网格
        side_length = int(np.ceil(np.sqrt(N)))
        positions = []
        
        for i in range(N):
            row = i // side_length
            col = i % side_length
            # 归一化到[0,1]，并添加小偏移避免边界问题
            x = (col + 0.5) / side_length
            y = (row + 0.5) / side_length
            positions.append([x, y])
        
        positions = torch.tensor(positions, device=device, dtype=torch.float32)
        return positions.unsqueeze(0).expand(B, -1, -1)
    
    def visualize_spatial_mapping(
        self, 
        gene_expression: torch.Tensor, 
        positions: torch.Tensor, 
        gene_idx: int = 0
    ) -> dict:
        """
        可视化空间映射过程（用于调试和分析）
        
        Args:
            gene_expression: [B, N, num_genes] - spots基因表达
            positions: [B, N, 2] - spots位置
            gene_idx: 要可视化的基因索引
        
        Returns:
            dict: 包含可视化信息的字典
        """
        spatial_gene_image, spatial_mask = self.spots_to_spatial_gene_image(gene_expression, positions)
        reconstructed_expr = self.spatial_gene_image_to_spots(spatial_gene_image, positions)
        
        # 计算重建误差
        reconstruction_error = torch.mean(torch.abs(gene_expression - reconstructed_expr))
        
        return {
            'original_spots': gene_expression[0, :, gene_idx].cpu().numpy(),
            'spatial_image': spatial_gene_image[0, gene_idx].cpu().numpy(),
            'reconstructed_spots': reconstructed_expr[0, :, gene_idx].cpu().numpy(),
            'reconstruction_error': reconstruction_error.item(),
            'spatial_mask': spatial_mask[0, 0].cpu().numpy()
        }