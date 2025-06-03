"""
VAR-ST Complete: 基于原始VAR架构的基因表达预测模型

核心设计理念：
- 完全保持原始VAR架构不变
- 基因表达向量 → 单通道伪图像 → VAR处理
- 使用数据适配而非架构修改的方式
- 正确理解VAR的next-scale prediction机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import math
import os

# 导入VAR基因包装器
from .var_gene_wrapper import VARGeneWrapper


class VAR_ST_Complete(nn.Module):
    """
    VAR-ST Complete: 基于原始VAR的基因表达预测模型
    
    🔧 修复核心问题：使用padding策略解决VAR的2D尺寸限制
    
    关键改进：
    - 使用16×16而不是14×14图像尺寸
    - 196基因 + 60个零padding = 256位置 (16×16)
    - 为VAR提供足够的空间进行多层卷积处理
    - patch_nums使用标准序列：(1, 2, 4, 8, 16)
    
    VAR的核心原理：
    - 不是传统的patch-based处理
    - 而是multi-scale autoregressive generation
    - patch_nums = [1, 2, 4, 8, 16] 表示生成序列：1x1 → 2x2 → 4x4 → 8x8 → 16x16
    - 每个scale基于前一个scale进行autoregressive prediction
    """
    
    def __init__(
        self,
        num_genes: int = 196,  # 🔧 固定196基因
        spatial_size: int = 64,  # 🔧 改为64×64，解决VQVAE下采样问题
        histology_feature_dim: Optional[int] = None,  # 🔧 改为可选参数
        feature_dim: Optional[int] = None,  # 🆕 新增从config.MODEL.feature_dim传入的参数
        var_config: Optional[Dict] = None,
        vqvae_config: Optional[Dict] = None,
        adapter_config: Optional[Dict] = None,
        **kwargs
    ):
        """
        完整版VAR-ST模型：空间转录组学基因表达预测 - Padding版本
        
        关键变化：
        - 🔧 固定196基因 → 16×16伪图像 (padding策略)
        - 🔧 使用标准VAR序列(1,2,4,8,16)，避免尺寸过小问题
        - 🔧 从数据配置正确获取组织学特征维度
        
        Args:
            num_genes: 基因数量，固定196
            spatial_size: 空间尺寸，改为16×16 (padding到256位置)
            histology_feature_dim: 组织学特征维度（优先级最高）
            feature_dim: 从config.MODEL.feature_dim传入的特征维度（作为fallback）
            var_config: VAR模型配置
            vqvae_config: VQVAE模型配置
            adapter_config: 适配器配置
        """
        super().__init__()
        
        # 🔧 确定正确的组织学特征维度
        if histology_feature_dim is not None:
            self.histology_feature_dim = histology_feature_dim
        elif feature_dim is not None:
            self.histology_feature_dim = feature_dim
            print(f"🔧 使用config.MODEL.feature_dim作为histology_feature_dim: {feature_dim}")
        else:
            # 默认值保持512，但会在下面警告
            self.histology_feature_dim = 512
            print("⚠️ 未指定histology_feature_dim，使用默认值512，可能导致维度不匹配")
        
        # 固定配置验证
        if num_genes != 196:
            raise ValueError(f"VAR_ST_Complete只支持196基因，got {num_genes}")
        
        if spatial_size < 16:
            raise ValueError(f"padding策略要求spatial_size至少为16，got {spatial_size}")
        
        self.num_genes = num_genes
        self.spatial_size = spatial_size
        
        print(f"🏗️ 初始化VAR_ST_Complete (196基因 + Padding策略):")
        print(f"   - 基因数量: {self.num_genes}")
        print(f"   - 目标图像尺寸: {self.spatial_size}×{self.spatial_size} (padding策略)")
        print(f"   - 总位置数: {self.spatial_size * self.spatial_size}")
        print(f"   - Padding大小: {self.spatial_size * self.spatial_size - self.num_genes}")
        print(f"   - 空间利用率: {self.num_genes / (self.spatial_size * self.spatial_size):.1%}")
        print(f"   - 组织学特征维度: {self.histology_feature_dim}")
        
        # 🔧 使用标准VAR兼容的patch_nums序列
        if spatial_size == 64:
            var_patch_nums = (1, 2, 4)  # 64×64输入经16倍下采样后为4×4
        elif spatial_size == 16:
            var_patch_nums = (1, 2, 4, 8, 16)  # 标准VAR序列，经过验证
        elif spatial_size == 20:
            var_patch_nums = (1, 2, 4, 5, 10, 20)  # 20×20的因子序列
        else:
            # 通用策略：生成合适的因子序列
            var_patch_nums = self._generate_patch_nums_for_size(spatial_size)
        
        print(f"🧬 VAR多尺度配置:")
        print(f"   - Patch序列: {var_patch_nums}")
        print(f"   - 分辨率演进: {' → '.join([f'{p}×{p}' for p in var_patch_nums])}")
        print(f"   - Token数量: {' + '.join([f'{p*p}' for p in var_patch_nums])} = {sum(p*p for p in var_patch_nums)}")
        
        # 🔧 使用基因包装器 - 传入16×16配置
        self.var_gene_wrapper = VARGeneWrapper(
            num_genes=num_genes,
            image_size=spatial_size,  # 使用16×16
            histology_feature_dim=self.histology_feature_dim,  # 🔧 使用正确的维度
            patch_nums=var_patch_nums,  # 🔧 传入标准VAR兼容序列
            var_config=var_config,
            vqvae_config=vqvae_config,
            adapter_config=adapter_config
        )
        
        # 训练状态管理
        self._step_count = 0
        self._verbose_logging = True
        
        # 显示padding策略优势
        self._show_padding_strategy(var_patch_nums)
        
        # 验证配置正确性
        self._validate_var_config(spatial_size, var_patch_nums)  # 验证VAR兼容序列
        
        print(f"✅ VAR-ST Complete初始化完成 (Padding策略)")
    
    def _show_padding_strategy(self, var_patch_nums: Tuple[int, ...]):
        """显示padding策略的优势"""
        print(f"   📦 Padding策略优势:")
        print(f"     * 解决尺寸限制: 14×14太小 → 16×16足够大")
        print(f"     * 支持标准VAR: 使用经过验证的patch序列")
        print(f"     * 保持架构不变: 无需修改VAR/VQVAE核心代码")
        print(f"     * 信息保持完整: 196基因信息完全保留")
        print(f"     * 计算开销小: 仅增加{(self.spatial_size**2 - self.num_genes) / self.num_genes:.1%}的存储")
        
        total_tokens = sum(p*p for p in var_patch_nums)
        print(f"   🎯 多尺度token分布:")
        for i, p in enumerate(var_patch_nums):
            tokens = p * p
            percentage = tokens / total_tokens * 100
            print(f"     * 尺度{i+1}: {p}×{p}={tokens} tokens ({percentage:.1f}%)")
    
    def _map_biological_to_var_scales(self, image_size: int) -> Tuple[int, ...]:
        """为padding策略生成VAR兼容的空间多尺度"""
        if image_size == 16:
            # 16×16图像的标准VAR兼容序列
            return (1, 2, 4, 8, 16)
        elif image_size == 20:
            # 20×20图像的VAR兼容序列  
            return (1, 2, 4, 5, 10, 20)
        elif image_size == 24:
            # 24×24图像的VAR兼容序列
            return (1, 2, 3, 4, 6, 8, 12, 24)
        else:
            # 通用映射：生成合适的因子序列
            factors = []
            for i in range(1, image_size + 1):
                if image_size % i == 0:
                    factors.append(i)
            # 选择合理的子集，确保不超过6-8个尺度
            if len(factors) > 8:
                step = len(factors) // 6
                selected_factors = factors[::step] + [factors[-1]]
                return tuple(sorted(set(selected_factors)))
            return tuple(factors)
    
    def _calculate_var_config(self, num_genes: int, patch_nums: Optional[Tuple[int, ...]]) -> Tuple[int, Tuple[int, ...]]:
        """
        🔧 解决需求与VAR架构兼容性：生物学多尺度映射到VAR空间多尺度
        
        核心问题：
        - 需求：patch_nums = (1, 2, 3, 4, 5) - 生物学语义层次
        - VAR：要求patch_nums[-1] == image_size - 空间分辨率约束
        
        解决方案：
        - 保持需求的生物学语义概念
        - 将生物学多尺度映射到VAR兼容的空间序列
        - 196基因 → 14×14 → VAR空间序列：(1, 2, 7, 14)
        
        Args:
            num_genes: 基因数量
            patch_nums: 可选的patch序列（将被解释为生物学语义）
            
        Returns:
            (image_size, var_patch_nums): 图像尺寸和VAR兼容的patch序列
        """
        # 🔧 严格按照需求：196基因必须对应14×14图像
        if num_genes == 196:
            image_size = 14  # 14×14 = 196，完美匹配
            
            # 🔧 关键修复：解决生物学多尺度与VAR空间约束的冲突
            # 需求的生物学多尺度：(1, 2, 3, 4, 5) 
            # VAR兼容的空间多尺度：(1, 2, 7, 14) - 确保最后值=14
            biological_scales = (1, 2, 3, 4, 5)  # 需求中的生物学语义
            var_spatial_scales = (1, 2, 7, 14)   # VAR兼容的空间序列
            
            print(f"🧬 生物学多尺度语义映射:")
            print(f"   - 生物学尺度: {biological_scales}")
            print(f"     * 1: 全局基因表达模式")
            print(f"     * 2: 基因功能组级别")
            print(f"     * 3: 基因通路级别")
            print(f"     * 4: 基因家族级别")
            print(f"     * 5: 单基因级别")
            print(f"   - VAR空间序列: {var_spatial_scales} (确保最后值=14)")
            print(f"   - 映射策略: 保持生物学语义，适配VAR空间约束")
            
            # 存储生物学语义信息，供后续使用
            default_patch_nums = var_spatial_scales
            self._biological_semantics = {
                'original_scales': biological_scales,
                'spatial_scales': var_spatial_scales,
                'semantic_names': [
                    "全局基因表达模式",
                    "基因功能组级别", 
                    "基因通路级别",
                    "单基因级别"  # 映射后只有4层
                ]
            }
            
        else:
            # 其他基因数量的备用映射
            gene_to_size_map = {
                225: 15,  # 225基因 → 15×15
                256: 16,  # 256基因 → 16×16
                144: 12,  # 144基因 → 12×12
                169: 13,  # 169基因 → 13×13
                64: 8,    # 64基因 → 8×8
                100: 10,  # 100基因 → 10×10
            }
            
            if num_genes in gene_to_size_map:
                image_size = gene_to_size_map[num_genes]
            else:
                # 自动计算最接近的平方数
                sqrt_genes = math.sqrt(num_genes)
                if sqrt_genes == int(sqrt_genes):
                    image_size = int(sqrt_genes)
                else:
                    image_size = math.ceil(sqrt_genes)
            
            # 对于非196基因的情况，生成VAR兼容的patch_nums
            default_patch_nums = self._generate_patch_nums_for_size(image_size)
            self._biological_semantics = None
        
        # 🔧 如果用户指定了patch_nums，进行智能处理
        if patch_nums is not None:
            # 对于196基因，用户的patch_nums被理解为生物学意图
            if num_genes == 196:
                print(f"⚠️ 检测到196基因用户配置: {patch_nums}")
                print(f"   解释为生物学语义意图，但使用VAR兼容序列: {default_patch_nums}")
                patch_nums = default_patch_nums
            # 对于其他基因，验证VAR兼容性
            elif patch_nums[-1] > image_size:
                print(f"⚠️ 用户指定的patch_nums最大值 {patch_nums[-1]} 超过图像尺寸 {image_size}")
                print(f"   自动修正为VAR兼容配置: {default_patch_nums}")
                patch_nums = default_patch_nums
        else:
            patch_nums = default_patch_nums
        
        return image_size, patch_nums
    
    def _generate_patch_nums_for_size(self, image_size: int) -> Tuple[int, ...]:
        """
        为指定图像尺寸生成合适的patch_nums序列
        
        注意：patch_nums的最后一个值应该等于VQVAE编码器的输出特征图尺寸，
        而不是输入图像尺寸。VAR的VQVAE下采样16倍。
        
        Args:
            image_size: 图像尺寸
            
        Returns:
            合适的patch_nums序列
        """
        # 根据图像尺寸生成合适的分辨率序列
        if image_size == 64:
            return (1, 2, 4)  # 64经16倍下采样后为4
        elif image_size == 16:
            return (1, 2, 4, 8, 16)
        elif image_size == 15:
            return (1, 3, 5, 15)
        elif image_size == 14:
            return (1, 2, 7, 14)
        elif image_size == 12:
            return (1, 2, 3, 4, 6, 12)
        elif image_size == 13:
            return (1, 13)
        elif image_size == 10:
            return (1, 2, 5, 10)
        elif image_size == 8:
            return (1, 2, 4, 8)
        else:
            # 通用策略：计算下采样后的特征图尺寸
            feature_map_size = max(1, image_size // 16)  # VAR VQVAE下采样16倍
            factors = []
            for i in range(1, feature_map_size + 1):
                if feature_map_size % i == 0:
                    factors.append(i)
            
            # 构建合理的序列
            if len(factors) >= 3:
                return tuple(factors)
            else:
                return (1, feature_map_size)
    
    def _generate_valid_patch_sequence(self, image_size: int) -> Tuple[int, ...]:
        """
        🔧 彻底修复：VAR的patch_nums实际上是分辨率序列！
        
        VAR的真正原理（从原始代码分析得出）：
        - v_patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16) 表示分辨率序列
        - 从1×1分辨率开始，逐步提升到最终的H×W分辨率
        - 关键断言：patch_hws[-1][0] == H and patch_hws[-1][1] == W
        - 这意味着最后一个值必须等于图像的高度和宽度
        
        Args:
            image_size: 最终图像尺寸（正方形图像）
            
        Returns:
            正确的分辨率序列
        """
        print(f"🔍 生成VAR分辨率序列，目标尺寸：{image_size}×{image_size}")
        
        # 根据图像尺寸生成合适的分辨率序列
        if image_size == 16:
            # 16×16: 经典VAR配置，从1×1到16×16
            patch_nums = (1, 2, 4, 8, 16)
        elif image_size == 15:
            # 15×15: 需要能被15整除的序列
            patch_nums = (1, 3, 5, 15)
        elif image_size == 12:
            # 12×12: 更多中间步骤
            patch_nums = (1, 2, 3, 4, 6, 12)
        elif image_size == 8:
            # 8×8: 2的幂次序列
            patch_nums = (1, 2, 4, 8)
        elif image_size == 10:
            # 10×10: 因子序列
            patch_nums = (1, 2, 5, 10)
        elif image_size == 14:
            # 14×14: 因子序列
            patch_nums = (1, 2, 7, 14)
        else:
            # 通用策略：找到合理的递增序列，确保最后等于image_size
            factors = []
            for i in range(1, image_size + 1):
                if image_size % i == 0:
                    factors.append(i)
            
            print(f"   - 可用因子: {factors}")
            
            # 构建递增序列
            if len(factors) >= 4:
                # 选择几个关键因子：开始、中间几个、结束
                indices = [0, 1, len(factors)//2, len(factors)-2, len(factors)-1]
                patch_nums = tuple(factors[i] for i in indices if factors[i] <= image_size)
            elif len(factors) >= 3:
                patch_nums = (factors[0], factors[1], factors[-1])
            else:
                patch_nums = (1, image_size)
            
            # 确保序列严格递增且不重复
            patch_nums = tuple(sorted(set(patch_nums)))
        
        print(f"✅ 生成的分辨率序列: {patch_nums}")
        print(f"   - 分辨率演进: {' → '.join([f'{p}×{p}' for p in patch_nums])}")
        
        return patch_nums
    
    def _validate_var_config(self, image_size: int, patch_nums: Tuple[int, ...]):
        """
        🔧 验证VAR配置的正确性
        
        VAR的两层结构：
        1. VQVAE层：将输入图像编码到特征图，要求patch_nums[-1]等于特征图尺寸
        2. VAR层：从特征图生成multi-scale tokens序列
        """
        print(f"🔍 验证VAR分辨率配置:")
        print(f"   - 输入图像尺寸: {image_size}×{image_size}")
        print(f"   - 分辨率序列: {patch_nums}")
        
        # VAR VQVAE下采样16倍
        feature_map_size = image_size // 16
        print(f"   - VQVAE特征图尺寸: {feature_map_size}×{feature_map_size} (下采样16倍)")
        
        # 验证1：最后一个分辨率必须等于VQVAE特征图尺寸
        if patch_nums[-1] != feature_map_size:
            raise ValueError(f"❌ VQVAE要求：最后一个分辨率 ({patch_nums[-1]}) 必须等于特征图尺寸 ({feature_map_size})")
        
        # 验证2：序列必须严格递增
        for i in range(1, len(patch_nums)):
            if patch_nums[i] <= patch_nums[i-1]:
                raise ValueError(f"❌ 分辨率序列必须严格递增，但 {patch_nums[i]} <= {patch_nums[i-1]}")
        
        # 验证3：第一个分辨率应该是1（VAR标准）
        if patch_nums[0] != 1:
            print(f"⚠️ 警告: VAR标准是从1×1开始，但当前从{patch_nums[0]}×{patch_nums[0]}开始")
        
        # 验证4：计算token数量
        total_tokens = sum(pn * pn for pn in patch_nums)
        print(f"   - 总token数量: {total_tokens}")
        print(f"   - Token分布: {' + '.join([f'{pn}²={pn*pn}' for pn in patch_nums])}")
        print(f"   - 分辨率演进: {' → '.join([f'{p}×{p}' for p in patch_nums])}")
        
        print(f"✅ VAR分辨率配置验证通过")

    def set_verbose_logging(self, verbose: bool):
        """设置详细日志输出"""
        self._verbose_logging = verbose
        if verbose:
            print("🔊 启用详细训练日志")
        else:
            print("🔇 切换到简洁训练模式")
    
    def forward_training(
        self,
        gene_expression: torch.Tensor,
        histology_features: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """训练阶段前向传播"""
        self._step_count += 1
        # 🔇 大幅减少详细输出：只在前3步和每1000步显示详情
        # 🔧 在分布式训练中，只在主进程显示详细信息
        is_main_process = int(os.environ.get('LOCAL_RANK', 0)) == 0
        show_details = (self._verbose_logging and is_main_process and 
                       (self._step_count <= 3 or self._step_count % 1000 == 0))
        
        if show_details:
            print(f"\n📊 VAR-ST Step {self._step_count} - 修复版VAR架构训练:")
        
        # 处理多spot输入情况
        if gene_expression.dim() == 3 and gene_expression.shape[1] > 1:
            if show_details:
                print(f"🔄 检测到多spot输入 {gene_expression.shape}，切换到多spot模式")
            return self.forward_multi_spot(gene_expression, histology_features, positions, class_labels)
        
        # 标准化输入格式
        gene_expression = self._normalize_gene_input(gene_expression, show_details)
        histology_features = self._normalize_histology_input(histology_features, show_details)
        
        B, num_genes = gene_expression.shape
        
        if show_details:
            print(f"   - 基因表达: {gene_expression.shape}")
            print(f"   - 组织学特征: {histology_features.shape}")
            print(f"🚀 执行: 基因→伪图像→VAR多尺度编码→自回归训练→重建")
        
        # 调用VAR基因包装器进行训练
        results = self.var_gene_wrapper.forward_training(
            gene_expression=gene_expression,
            histology_features=histology_features,
            class_labels=class_labels,
            show_details=show_details
        )
        
        if show_details:
            print(f"📊 损失: VAR={results['var_loss'].item():.4f}, 重建={results['recon_loss'].item():.4f}, 总计={results['loss'].item():.4f}")
        
        # 添加兼容性字段
        results.update({
            'predicted_expression': results['predictions'],
            'logits': results['predictions'],
            'targets': gene_expression
        })
        
        return results
    
    def forward_multi_spot(
        self,
        gene_expression: torch.Tensor,
        histology_features: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """多spot前向传播"""
        B, N, num_genes = gene_expression.shape
        
        print(f"🌟 VAR-ST多spot前向传播 (修复版VAR架构):")
        print(f"   - 输入shape: {gene_expression.shape}")
        print(f"   - Batch size: {B}, Spots per sample: {N}")
        
        # 重塑输入
        gene_expr_flat = gene_expression.view(B * N, num_genes).contiguous()
        
        if histology_features.dim() == 3:
            hist_feat_flat = histology_features.view(B * N, -1).contiguous()
        else:
            hist_feat_flat = histology_features.unsqueeze(1).expand(-1, N, -1).view(B * N, -1).contiguous()
        
        if class_labels is not None:
            if class_labels.dim() == 1 and class_labels.shape[0] == B:
                class_labels_flat = class_labels.unsqueeze(1).expand(-1, N).view(B * N).contiguous()
            else:
                class_labels_flat = class_labels
        else:
            class_labels_flat = None
        
        # 调用单spot训练方法
        spot_results = self.forward_training(
            gene_expression=gene_expr_flat,
            histology_features=hist_feat_flat,
            positions=None,
            class_labels=class_labels_flat
        )
        
        # 重塑输出
        predictions = spot_results['predictions'].view(B, N, num_genes).contiguous()
        targets = gene_expression
        
        return {
            'loss': spot_results['loss'],
            'var_loss': spot_results['var_loss'],
            'recon_loss': spot_results['recon_loss'],
            'predictions': predictions,
            'targets': targets,
            'predicted_expression': predictions,
            'logits': predictions,
            'class_labels': spot_results.get('class_labels'),
            'pseudo_images': spot_results.get('pseudo_images'),
            'tokens': spot_results.get('tokens')
        }
    
    def forward_inference(
        self,
        histology_features: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.5,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0,
        num_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """推理阶段：从组织学特征生成基因表达预测"""
        histology_features = self._normalize_histology_input(histology_features, show_details=False)
        
        # 🔇 简化推理输出
        print(f"🔮 VAR-ST推理: {histology_features.shape[0]} samples, CFG={cfg_scale}")
        
        # 调用VAR基因包装器进行推理
        results = self.var_gene_wrapper.forward_inference(
            histology_features=histology_features,
            class_labels=class_labels,
            cfg_scale=cfg_scale,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_samples=num_samples
        )
        
        print(f"✅ 推理完成: {results['predictions'].shape}")
        
        # 添加兼容性字段
        results.update({
            'predicted_expression': results['predictions'],
            'logits': results['predictions']
        })
        
        return results
    
    def _normalize_gene_input(self, gene_expression: torch.Tensor, show_details: bool = False) -> torch.Tensor:
        """标准化基因表达输入格式"""
        if gene_expression.dim() == 3:
            B, N, num_genes = gene_expression.shape
            if N == 1:
                gene_expression = gene_expression.squeeze(1)
                if show_details:
                    print(f"🔧 压缩单spot输入: [B, N=1, num_genes] → [B, num_genes]")
            else:
                raise ValueError(f"多spot输入应由forward_multi_spot处理: {gene_expression.shape}")
        
        return gene_expression.contiguous()
    
    def _normalize_histology_input(self, histology_features: torch.Tensor, show_details: bool = False) -> torch.Tensor:
        """标准化组织学特征输入格式"""
        if histology_features.dim() == 3:
            B, N, feature_dim = histology_features.shape
            if N == 1:
                histology_features = histology_features.squeeze(1)
                if show_details:
                    print(f"🔧 压缩单spot特征: [B, N=1, feature_dim] → [B, feature_dim]")
            else:
                histology_features = histology_features.mean(dim=1)
                if show_details:
                    print(f"🔧 平均多spot特征: [B, N={N}, feature_dim] → [B, feature_dim]")
        
        return histology_features.contiguous()
    
    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """统一前向传播接口"""
        mode = inputs.get('mode', 'training')
        
        gene_expression = inputs.get('gene_expression')
        histology_features = inputs.get('histology_features')
        
        if mode == 'training':
            return self.forward_training(
                gene_expression=gene_expression,
                histology_features=histology_features,
                positions=inputs.get('positions'),
                class_labels=inputs.get('class_labels')
            )
        else:
            return self.forward_inference(
                histology_features=histology_features,
                positions=inputs.get('positions'),
                class_labels=inputs.get('class_labels'),
                cfg_scale=inputs.get('cfg_scale', 1.5),
                top_k=inputs.get('top_k', 50),
                top_p=inputs.get('top_p', 0.9),
                temperature=inputs.get('temperature', 1.0),
                num_samples=inputs.get('num_samples', 1)
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': 'VAR_ST_Complete',
            'architecture': 'VAR + Gene Pseudo Image Adapter (Fixed)',
            'num_genes': self.num_genes,
            'pseudo_image_size': f"{self.spatial_size}x{self.spatial_size}",
            'patch_nums': (1, 2, 4, 8, 16),
            'histology_feature_dim': self.histology_feature_dim,
            'patch_sequence': ' → '.join([f'{p}×{p}' for p in (1, 2, 4, 8, 16)]),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        } 