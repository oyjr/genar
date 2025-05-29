"""
空间多尺度基因表达组织器

实现VAR原始设计理念的空间多尺度概念：
- 将空间转录组学数据组织为不同分辨率的空间网格
- 每个网格cell聚合其内部spots的基因表达
- 支持从粗粒度到细粒度的渐进式生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import cdist


class SpatialMultiscaleOrganizer:
    """
    空间多尺度基因表达组织器
    
    将spots的坐标和基因表达组织为多个空间分辨率的网格表示，
    每个网格cell包含聚合的基因表达向量。
    """
    
    def __init__(
        self,
        scales: List[int] = [1, 2, 4, 8],
        aggregation_method: str = 'mean',
        spatial_smoothing: bool = True,
        normalize_coordinates: bool = True
    ):
        """
        初始化空间多尺度组织器
        
        Args:
            scales: 空间分辨率列表 [1, 2, 4, 8] 表示 1×1, 2×2, 4×4, 8×8 网格
            aggregation_method: 聚合方法 ('mean', 'sum', 'max', 'weighted_mean')
            spatial_smoothing: 是否应用空间平滑
            normalize_coordinates: 是否标准化坐标到[0,1]范围
        """
        self.scales = scales
        self.aggregation_method = aggregation_method
        self.spatial_smoothing = spatial_smoothing
        self.normalize_coordinates = normalize_coordinates
        
        print(f"🗂️ 初始化空间多尺度组织器:")
        print(f"   - 分辨率层级: {scales}")
        print(f"   - 聚合方法: {aggregation_method}")
        print(f"   - 空间平滑: {spatial_smoothing}")
        print(f"   - 坐标标准化: {normalize_coordinates}")
    
    def organize_multiscale(
        self,
        gene_expression: torch.Tensor,
        positions: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        将spots组织为多尺度空间网格
        
        Args:
            gene_expression: [B, N, num_genes] - spots的基因表达
            positions: [B, N, 2] - spots的空间坐标 (x, y)
        
        Returns:
            List[torch.Tensor]: 每个尺度的网格表示
            - scale 1×1: [B, 1, num_genes]
            - scale 2×2: [B, 4, num_genes] 
            - scale 4×4: [B, 16, num_genes]
            - scale 8×8: [B, 64, num_genes]
        """
        B, N, num_genes = gene_expression.shape
        device = gene_expression.device
        
        print(f"🗂️ 组织多尺度空间数据:")
        print(f"   - 输入: {gene_expression.shape} 基因表达, {positions.shape} 位置")
        
        multiscale_expressions = []
        
        for scale_idx, scale in enumerate(self.scales):
            print(f"   - 处理尺度 {scale}×{scale}...")
            
            # 为每个batch样本处理
            batch_scale_expressions = []
            
            for b in range(B):
                batch_gene_expr = gene_expression[b]  # [N, num_genes]
                batch_positions = positions[b]        # [N, 2]
                
                # 组织为当前尺度的网格
                grid_expression = self._organize_single_scale(
                    batch_gene_expr, batch_positions, scale
                )  # [scale*scale, num_genes]
                
                batch_scale_expressions.append(grid_expression)
            
            # 合并batch维度
            scale_expressions = torch.stack(batch_scale_expressions, dim=0)  # [B, scale*scale, num_genes]
            multiscale_expressions.append(scale_expressions)
            
            print(f"     -> 输出: {scale_expressions.shape}")
        
        print(f"✅ 多尺度组织完成，共{len(multiscale_expressions)}个尺度")
        return multiscale_expressions
    
    def _organize_single_scale(
        self,
        gene_expression: torch.Tensor,
        positions: torch.Tensor,
        scale: int
    ) -> torch.Tensor:
        """
        将单个样本组织为指定尺度的网格
        
        Args:
            gene_expression: [N, num_genes] - spots基因表达
            positions: [N, 2] - spots空间坐标
            scale: 网格分辨率 (scale×scale)
        
        Returns:
            torch.Tensor: [scale*scale, num_genes] - 网格化的基因表达
        """
        N, num_genes = gene_expression.shape
        device = gene_expression.device
        
        # 标准化坐标到[0, 1]范围
        if self.normalize_coordinates:
            positions_norm = self._normalize_positions(positions)
        else:
            positions_norm = positions
        
        # 创建网格
        grid_coords = self._create_grid_coordinates(scale, device)  # [scale*scale, 2]
        
        # 为每个网格cell聚合基因表达
        grid_expressions = []
        
        for grid_idx in range(scale * scale):
            grid_center = grid_coords[grid_idx]  # [2]
            
            # 计算每个spot到当前grid center的距离权重
            distances = torch.norm(positions_norm - grid_center.unsqueeze(0), dim=1)  # [N]
            
            # 使用距离权重聚合基因表达
            if self.aggregation_method == 'mean':
                # 简单平均 (在网格cell内的spots)
                cell_size = 1.0 / scale
                in_cell_mask = (
                    (torch.abs(positions_norm[:, 0] - grid_center[0]) < cell_size / 2) &
                    (torch.abs(positions_norm[:, 1] - grid_center[1]) < cell_size / 2)
                )
                
                if in_cell_mask.sum() > 0:
                    cell_expression = gene_expression[in_cell_mask].mean(dim=0)
                else:
                    # 如果cell内没有spots，使用最近邻
                    nearest_idx = distances.argmin()
                    cell_expression = gene_expression[nearest_idx]
                    
            elif self.aggregation_method == 'weighted_mean':
                # 高斯权重聚合
                sigma = 1.0 / (scale * 2)  # 自适应标准差
                weights = torch.exp(-distances**2 / (2 * sigma**2))
                weights = weights / (weights.sum() + 1e-8)  # 归一化权重
                
                cell_expression = torch.sum(
                    weights.unsqueeze(1) * gene_expression, dim=0
                )  # [num_genes]
                
            elif self.aggregation_method == 'sum':
                # 区域内求和
                cell_size = 1.0 / scale
                in_cell_mask = (
                    (torch.abs(positions_norm[:, 0] - grid_center[0]) < cell_size / 2) &
                    (torch.abs(positions_norm[:, 1] - grid_center[1]) < cell_size / 2)
                )
                
                if in_cell_mask.sum() > 0:
                    cell_expression = gene_expression[in_cell_mask].sum(dim=0)
                else:
                    nearest_idx = distances.argmin()
                    cell_expression = gene_expression[nearest_idx]
                    
            else:
                raise ValueError(f"不支持的聚合方法: {self.aggregation_method}")
            
            grid_expressions.append(cell_expression)
        
        # 合并所有grid cells
        grid_result = torch.stack(grid_expressions, dim=0)  # [scale*scale, num_genes]
        
        # 可选的空间平滑
        if self.spatial_smoothing and scale > 1:
            grid_result = self._apply_spatial_smoothing(grid_result, scale)
        
        return grid_result
    
    def _normalize_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """标准化坐标到[0, 1]范围"""
        pos_min = positions.min(dim=0, keepdim=True)[0]
        pos_max = positions.max(dim=0, keepdim=True)[0]
        pos_range = pos_max - pos_min
        
        # 避免除零
        pos_range = torch.where(pos_range > 1e-8, pos_range, torch.ones_like(pos_range))
        
        normalized = (positions - pos_min) / pos_range
        return normalized
    
    def _create_grid_coordinates(self, scale: int, device: torch.device) -> torch.Tensor:
        """创建网格中心坐标"""
        # 创建均匀分布的网格中心点
        step = 1.0 / scale
        coords = torch.linspace(step/2, 1-step/2, scale, device=device)
        
        # 创建网格坐标 [scale*scale, 2]
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        grid_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        return grid_coords
    
    def _apply_spatial_smoothing(self, grid_expression: torch.Tensor, scale: int) -> torch.Tensor:
        """对网格应用空间平滑"""
        # 重塑为空间网格 [scale, scale, num_genes]
        num_genes = grid_expression.shape[1]
        spatial_grid = grid_expression.view(scale, scale, num_genes)
        
        # 应用2D高斯滤波进行平滑
        # 转置为 [num_genes, scale, scale] 以便批量处理
        spatial_grid = spatial_grid.permute(2, 0, 1).unsqueeze(0)  # [1, num_genes, scale, scale]
        
        # 创建高斯核
        kernel_size = 3 if scale >= 4 else 1
        if kernel_size > 1:
            smoothed = F.avg_pool2d(
                spatial_grid, 
                kernel_size=kernel_size, 
                stride=1, 
                padding=kernel_size//2
            )
            smoothed = smoothed.squeeze(0).permute(1, 2, 0)  # [scale, scale, num_genes]
            smoothed = smoothed.view(-1, num_genes)  # [scale*scale, num_genes]
        else:
            smoothed = grid_expression
        
        return smoothed
    
    def reconstruct_from_multiscale(
        self, 
        multiscale_expressions: List[torch.Tensor],
        target_positions: torch.Tensor,
        reconstruction_method: str = 'finest_scale'
    ) -> torch.Tensor:
        """
        从多尺度表示重建原始spots的基因表达
        
        Args:
            multiscale_expressions: 多尺度基因表达列表
            target_positions: [B, N, 2] - 目标spots的位置
            reconstruction_method: 重建方法 ('finest_scale', 'hierarchical', 'weighted_combination')
        
        Returns:
            torch.Tensor: [B, N, num_genes] - 重建的基因表达
        """
        B, N, _ = target_positions.shape
        num_genes = multiscale_expressions[0].shape[-1]
        device = target_positions.device
        
        print(f"🔄 从多尺度重建基因表达:")
        print(f"   - 目标位置: {target_positions.shape}")
        print(f"   - 重建方法: {reconstruction_method}")
        
        if reconstruction_method == 'finest_scale':
            # 使用最细尺度进行插值重建
            finest_scale_expr = multiscale_expressions[-1]  # [B, scale*scale, num_genes]
            finest_scale = self.scales[-1]
            
            reconstructed = self._interpolate_from_grid(
                finest_scale_expr, target_positions, finest_scale
            )
            
        elif reconstruction_method == 'hierarchical':
            # 分层重建：从粗到细逐步细化
            reconstructed = None
            
            for scale_idx, scale_expr in enumerate(multiscale_expressions):
                scale = self.scales[scale_idx]
                
                scale_contribution = self._interpolate_from_grid(
                    scale_expr, target_positions, scale
                )
                
                if reconstructed is None:
                    reconstructed = scale_contribution
                else:
                    # 加权融合
                    weight = 0.5 ** (len(multiscale_expressions) - scale_idx - 1)
                    reconstructed = reconstructed * (1 - weight) + scale_contribution * weight
                    
        elif reconstruction_method == 'weighted_combination':
            # 加权组合所有尺度
            all_contributions = []
            weights = []
            
            for scale_idx, scale_expr in enumerate(multiscale_expressions):
                scale = self.scales[scale_idx]
                contribution = self._interpolate_from_grid(
                    scale_expr, target_positions, scale
                )
                all_contributions.append(contribution)
                
                # 更细的尺度有更高权重
                weight = (scale_idx + 1) / len(multiscale_expressions)
                weights.append(weight)
            
            # 标准化权重
            weights = torch.tensor(weights, device=device)
            weights = weights / weights.sum()
            
            # 加权组合
            reconstructed = torch.zeros_like(all_contributions[0])
            for contrib, weight in zip(all_contributions, weights):
                reconstructed += contrib * weight
                
        else:
            raise ValueError(f"不支持的重建方法: {reconstruction_method}")
        
        print(f"   - 重建结果: {reconstructed.shape}")
        return reconstructed
    
    def _interpolate_from_grid(
        self,
        grid_expression: torch.Tensor,
        target_positions: torch.Tensor,
        scale: int
    ) -> torch.Tensor:
        """
        从网格插值到目标位置
        
        Args:
            grid_expression: [B, scale*scale, num_genes] - 网格基因表达
            target_positions: [B, N, 2] - 目标位置
            scale: 网格分辨率
        
        Returns:
            torch.Tensor: [B, N, num_genes] - 插值后的基因表达
        """
        B, N, _ = target_positions.shape
        num_genes = grid_expression.shape[-1]
        device = target_positions.device
        
        # 创建网格坐标
        grid_coords = self._create_grid_coordinates(scale, device)  # [scale*scale, 2]
        
        interpolated_expressions = []
        
        for b in range(B):
            batch_grid_expr = grid_expression[b]      # [scale*scale, num_genes]
            batch_target_pos = target_positions[b]    # [N, 2]
            
            # 标准化目标位置
            if self.normalize_coordinates:
                batch_target_pos_norm = self._normalize_positions(batch_target_pos)
            else:
                batch_target_pos_norm = batch_target_pos
            
            # 计算距离权重进行插值
            distances = torch.cdist(batch_target_pos_norm, grid_coords)  # [N, scale*scale]
            
            # 使用反距离权重插值
            eps = 1e-8
            weights = 1.0 / (distances + eps)  # [N, scale*scale]
            weights = weights / weights.sum(dim=1, keepdim=True)  # 归一化
            
            # 加权平均
            interpolated = torch.mm(weights, batch_grid_expr)  # [N, num_genes]
            interpolated_expressions.append(interpolated)
        
        result = torch.stack(interpolated_expressions, dim=0)  # [B, N, num_genes]
        return result 