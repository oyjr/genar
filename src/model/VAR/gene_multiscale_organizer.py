"""
基因多尺度组织器

实现VAR原始设计理念的基因维度多尺度概念：
- 将基因表达向量组织为不同粒度的表示
- 每个尺度包含不同数量的基因特征
- 支持从粗粒度到细粒度的渐进式生成

这是VAR架构的核心：多尺度自回归建模，只是应用在基因维度而不是空间维度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Any


class GeneMultiscaleOrganizer(nn.Module):
    """
    基因多尺度组织器
    
    将基因表达向量组织为多个尺度的表示，
    每个尺度包含不同粒度的基因信息。
    
    核心思想：
    - 尺度1: 1个特征 (全局基因表达模式)
    - 尺度2: 4个特征 (主要功能模块)  
    - 尺度3: 16个特征 (功能通路级别)
    - 尺度4: 64个特征 (基因簇级别)
    - 尺度5: 200个特征 (完整基因表达)
    
    这与VAR在图像上的1×1→2×2→4×4→8×8→16×16完全对应
    """
    
    def __init__(
        self,
        num_genes: int = 200,
        scales: List[int] = [1, 4, 16, 64, 200],
        projection_method: str = 'learned',
        preserve_variance: bool = True,
        normalize_features: bool = True
    ):
        """
        初始化基因多尺度组织器
        
        Args:
            num_genes: 总基因数量
            scales: 多尺度特征数量列表 [1, 4, 16, 64, 200]
            projection_method: 投影方法 ('learned', 'pca', 'importance')
            preserve_variance: 是否保持方差信息
            normalize_features: 是否标准化特征
        """
        super().__init__()  # 继承nn.Module
        
        self.num_genes = num_genes
        self.scales = scales
        self.projection_method = projection_method
        self.preserve_variance = preserve_variance
        self.normalize_features = normalize_features
        
        print(f"🧬 初始化基因多尺度组织器:")
        print(f"   - 基因数量: {num_genes}")
        print(f"   - 尺度层级: {scales}")
        print(f"   - 投影方法: {projection_method}")
        print(f"   - 保持方差: {preserve_variance}")
        
        # 验证尺度设置
        assert scales[-1] == num_genes, f"最后一个尺度必须等于基因数量: {scales[-1]} != {num_genes}"
        assert all(scales[i] <= scales[i+1] for i in range(len(scales)-1)), "尺度必须递增"
        
        # 初始化投影层（如果使用学习投影）
        if projection_method == 'learned':
            self.projection_layers = nn.ModuleList()
            for scale in scales[:-1]:  # 最后一个尺度就是原始数据
                self.projection_layers.append(
                    nn.Linear(num_genes, scale, bias=False)
                )
            
            # 初始化投影矩阵
            self._initialize_projections()
        
        # 存储重建层（用于验证信息保持）
        if preserve_variance:
            self.reconstruction_layers = nn.ModuleList()
            for scale in scales[:-1]:
                self.reconstruction_layers.append(
                    nn.Linear(scale, num_genes, bias=False)
                )
    
    def _initialize_projections(self):
        """初始化投影矩阵以保持重要信息"""
        print("🔧 初始化基因投影矩阵...")
        
        for i, proj_layer in enumerate(self.projection_layers):
            # 使用正交初始化保持信息
            nn.init.orthogonal_(proj_layer.weight)
            
            # 缩放权重以保持方差
            scale_factor = np.sqrt(self.scales[i] / self.num_genes)
            proj_layer.weight.data *= scale_factor
            
            print(f"   - 尺度{i+1}: {self.num_genes} → {self.scales[i]} (正交投影)")
    
    def organize_multiscale(
        self,
        gene_expression: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        将基因表达向量组织为多尺度表示
        
        Args:
            gene_expression: [B, num_genes] - 基因表达向量
        
        Returns:
            List[torch.Tensor]: 多尺度基因表达
            - scale_0: [B, 1] - 全局模式
            - scale_1: [B, 4] - 主要模块  
            - scale_2: [B, 16] - 功能通路
            - scale_3: [B, 64] - 基因簇
            - scale_4: [B, 200] - 完整表达
        """
        B, num_genes = gene_expression.shape
        device = gene_expression.device
        
        if num_genes != self.num_genes:
            raise ValueError(f"输入基因数量 {num_genes} 与配置不符 {self.num_genes}")
        
        print(f"🧬 组织基因多尺度表示:")
        print(f"   - 输入: {gene_expression.shape}")
        
        # 可选的特征标准化
        if self.normalize_features:
            gene_expression_norm = F.layer_norm(gene_expression, [num_genes])
        else:
            gene_expression_norm = gene_expression
        
        multiscale_expressions = []
        
        # 生成每个尺度的表示
        for scale_idx, scale in enumerate(self.scales):
            if scale == self.num_genes:
                # 最后一个尺度：使用原始数据
                scale_expression = gene_expression_norm
                print(f"   - 尺度{scale_idx+1}: {scale_expression.shape} (原始)")
                
            else:
                # 其他尺度：使用投影
                if self.projection_method == 'learned':
                    proj_layer = self.projection_layers[scale_idx]
                    scale_expression = proj_layer(gene_expression_norm)  # [B, scale]
                    
                elif self.projection_method == 'pca':
                    scale_expression = self._pca_projection(gene_expression_norm, scale)
                    
                elif self.projection_method == 'importance':
                    scale_expression = self._importance_projection(gene_expression_norm, scale)
                    
                else:
                    raise ValueError(f"不支持的投影方法: {self.projection_method}")
                
                print(f"   - 尺度{scale_idx+1}: {scale_expression.shape} (投影)")
            
            # 确保张量连续性
            scale_expression = scale_expression.contiguous()
            multiscale_expressions.append(scale_expression)
        
        print(f"✅ 基因多尺度组织完成，共{len(multiscale_expressions)}个尺度")
        return multiscale_expressions
    
    def _pca_projection(self, gene_expression: torch.Tensor, target_dim: int) -> torch.Tensor:
        """使用PCA进行降维投影"""
        # 简化的PCA投影（实际使用中可以预计算主成分）
        B, num_genes = gene_expression.shape
        
        # 计算协方差矩阵的主成分（简化版本）
        centered = gene_expression - gene_expression.mean(dim=0, keepdim=True)
        
        # SVD分解
        U, S, V = torch.svd(centered.t())  # 转置后进行SVD
        
        # 取前target_dim个主成分
        principal_components = V[:, :target_dim]  # [num_genes, target_dim]
        
        # 投影
        projected = torch.mm(gene_expression, principal_components)  # [B, target_dim]
        
        return projected
    
    def _importance_projection(self, gene_expression: torch.Tensor, target_dim: int) -> torch.Tensor:
        """基于基因重要性的投影"""
        B, num_genes = gene_expression.shape
        
        # 计算基因的重要性分数（方差作为简单的重要性度量）
        gene_variance = gene_expression.var(dim=0)  # [num_genes]
        
        # 选择方差最大的基因
        _, top_indices = torch.topk(gene_variance, target_dim)
        
        # 投影到重要基因子空间
        projected = gene_expression[:, top_indices]  # [B, target_dim]
        
        return projected
    
    def reconstruct_from_multiscale(
        self, 
        multiscale_expressions: List[torch.Tensor],
        reconstruction_method: str = 'finest_scale'
    ) -> torch.Tensor:
        """
        从多尺度表示重建完整的基因表达
        
        Args:
            multiscale_expressions: 多尺度基因表达列表
            reconstruction_method: 重建方法
                - 'finest_scale': 直接使用最细粒度尺度
                - 'learned_combination': 学习多尺度组合
                - 'progressive': 渐进式重建
        
        Returns:
            torch.Tensor: [B, num_genes] - 重建的基因表达
        """
        if reconstruction_method == 'finest_scale':
            # 最简单：直接返回最后一个尺度（完整基因表达）
            return multiscale_expressions[-1]
        
        elif reconstruction_method == 'learned_combination':
            # 使用学习到的重建层组合多尺度信息
            B = multiscale_expressions[0].shape[0]
            device = multiscale_expressions[0].device
            
            reconstructed = torch.zeros(B, self.num_genes, device=device)
            
            for scale_idx, scale_expr in enumerate(multiscale_expressions[:-1]):
                if hasattr(self, 'reconstruction_layers'):
                    recon_layer = self.reconstruction_layers[scale_idx]
                    scale_contribution = recon_layer(scale_expr)
                    reconstructed += scale_contribution
            
            # 添加最细尺度
            reconstructed += multiscale_expressions[-1]
            
            return reconstructed
        
        elif reconstruction_method == 'progressive':
            # 渐进式重建：从粗粒度开始逐步细化
            current_reconstruction = multiscale_expressions[0]  # 从最粗尺度开始
            
            for scale_idx in range(1, len(multiscale_expressions)):
                current_scale = multiscale_expressions[scale_idx]
                
                if current_scale.shape[1] == self.num_genes:
                    # 最后一个尺度：直接使用
                    current_reconstruction = current_scale
                else:
                    # 中间尺度：组合当前重建和新尺度信息
                    if hasattr(self, 'reconstruction_layers') and scale_idx-1 < len(self.reconstruction_layers):
                        recon_layer = self.reconstruction_layers[scale_idx-1]
                        upsampled = recon_layer(current_scale)
                        
                        # 如果维度匹配，进行残差连接
                        if hasattr(current_reconstruction, 'shape') and current_reconstruction.shape[1] == upsampled.shape[1]:
                            current_reconstruction = current_reconstruction + upsampled
                        else:
                            current_reconstruction = upsampled
                    else:
                        # 简单策略：使用当前尺度
                        current_reconstruction = current_scale
            
            return current_reconstruction
        
        else:
            raise ValueError(f"不支持的重建方法: {reconstruction_method}")
    
    def validate_information_preservation(
        self,
        original: torch.Tensor,
        multiscale: List[torch.Tensor],
        tolerance: float = 0.1
    ) -> Dict[str, float]:
        """
        验证多尺度分解是否保持了重要信息
        
        Args:
            original: [B, num_genes] - 原始基因表达
            multiscale: 多尺度表示列表
            tolerance: 容忍的信息损失比例
        
        Returns:
            Dict: 验证结果
        """
        results = {}
        
        # 重建验证
        reconstructed = self.reconstruct_from_multiscale(multiscale, 'finest_scale')
        
        # 计算重建误差
        mse_loss = F.mse_loss(reconstructed, original)
        relative_error = mse_loss / (original.var() + 1e-8)
        
        results['mse_loss'] = mse_loss.item()
        results['relative_error'] = relative_error.item()
        results['information_preserved'] = relative_error.item() < tolerance
        
        # 每个尺度的信息量
        for i, scale_expr in enumerate(multiscale):
            scale_variance = scale_expr.var()
            results[f'scale_{i+1}_variance'] = scale_variance.item()
        
        # 相关性验证
        correlation = F.cosine_similarity(
            original.view(-1), 
            reconstructed.view(-1), 
            dim=0
        )
        results['cosine_similarity'] = correlation.item()
        
        print(f"📊 信息保持验证:")
        print(f"   - MSE损失: {results['mse_loss']:.6f}")
        print(f"   - 相对误差: {results['relative_error']:.4f}")
        print(f"   - 余弦相似度: {results['cosine_similarity']:.4f}")
        print(f"   - 信息保持: {'✅' if results['information_preserved'] else '❌'}")
        
        return results
    
    def get_scale_info(self) -> Dict[str, Any]:
        """获取尺度配置信息"""
        return {
            'num_genes': self.num_genes,
            'scales': self.scales,
            'projection_method': self.projection_method,
            'num_scale_levels': len(self.scales),
            'scale_ratios': [s / self.num_genes for s in self.scales],
            'compression_ratios': [self.num_genes / s for s in self.scales]
        } 