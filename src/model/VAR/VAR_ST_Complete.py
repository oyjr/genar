"""
VAR-ST Complete: 空间转录组学的视觉自回归模型

基于VAR原始设计理念的真正空间多尺度实现：
- 将空间转录组学数据组织为不同分辨率的空间网格
- 使用VQVAE对每个尺度的基因表达网格进行编码
- VAR自回归生成：从粗粒度全局模式到细粒度局部细节
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .vqvae_st import VQVAE
from .var_st import VAR
from .spatial_multiscale_organizer import SpatialMultiscaleOrganizer


class VAR_ST_Complete(nn.Module):
    """
    VAR-ST Complete: 空间转录组学的完整VAR实现
    
    实现真正的空间多尺度VAR：
    1. 空间多尺度组织：将spots组织为不同分辨率的空间网格
    2. 多尺度VQVAE编码：每个尺度使用专门的VQVAE编码器
    3. VAR自回归生成：从粗粒度到细粒度渐进式生成
    4. 空间重建：从多尺度网格重建到原始spots
    """
    
    def __init__(
        self,
        num_genes: int = 200,
        histology_feature_dim: int = 512,
        spatial_scales: List[int] = [1, 2, 4, 8],
        vqvae_configs: Optional[List[Dict]] = None,
        var_config: Optional[Dict] = None,
        spatial_config: Optional[Dict] = None,
        **kwargs
    ):
        """
        初始化VAR-ST Complete模型
        
        Args:
            num_genes: 基因数量
            histology_feature_dim: 组织学特征维度 
            spatial_scales: 空间分辨率列表 [1, 2, 4, 8]
            vqvae_configs: 每个尺度的VQVAE配置列表
            var_config: VAR模型配置
            spatial_config: 空间组织器配置
        """
        super().__init__()
        
        self.num_genes = num_genes
        self.histology_feature_dim = histology_feature_dim
        self.spatial_scales = spatial_scales
        self.num_scales = len(spatial_scales)
        
        print(f"🧬 初始化VAR_ST_Complete (真正的空间多尺度模式)")
        print(f"   - 目标基因数: {num_genes}")
        print(f"   - 组织学特征维度: {histology_feature_dim}")
        print(f"   - 空间分辨率: {spatial_scales}")
        
        # 初始化空间多尺度组织器
        spatial_config = spatial_config or {}
        self.spatial_organizer = SpatialMultiscaleOrganizer(
            scales=spatial_scales,
            aggregation_method=spatial_config.get('aggregation_method', 'weighted_mean'),
            spatial_smoothing=spatial_config.get('spatial_smoothing', True),
            normalize_coordinates=spatial_config.get('normalize_coordinates', True)
        )
        
        # 为每个空间尺度创建专门的VQVAE编码器
        self.vqvaes = nn.ModuleList()
        self.codebook_sizes = []
        
        if vqvae_configs is None:
            vqvae_configs = [self._get_default_vqvae_config(scale) for scale in spatial_scales]
        
        for scale_idx, scale in enumerate(spatial_scales):
            print(f"🧬 初始化尺度 {scale}×{scale} 的VQVAE:")
            
            vqvae_config = vqvae_configs[scale_idx] if scale_idx < len(vqvae_configs) else vqvae_configs[-1]
            
            # 每个尺度的VQVAE处理相同维度的基因向量 [num_genes]
            vqvae = VQVAE(
                input_dim=num_genes,
                hidden_dim=vqvae_config.get('hidden_dim', 256),
                latent_dim=vqvae_config.get('latent_dim', 32),
                num_embeddings=vqvae_config.get('num_embeddings', 2048),  # 不同尺度可以有不同码本大小
                commitment_cost=vqvae_config.get('commitment_cost', 0.25)
            )
            
            self.vqvaes.append(vqvae)
            self.codebook_sizes.append(vqvae_config.get('num_embeddings', 2048))
            
            print(f"  - 隐藏维度: {vqvae_config.get('hidden_dim', 256)}")
            print(f"  - 潜在维度: {vqvae_config.get('latent_dim', 32)}")
            print(f"  - 码本大小: {vqvae_config.get('num_embeddings', 2048)}")
        
        # 计算每个尺度的token数量
        self.tokens_per_scale = [scale * scale for scale in spatial_scales]
        self.total_tokens = sum(self.tokens_per_scale)
        
        print(f"   - 每尺度token数: {self.tokens_per_scale}")
        print(f"   - 总token数: {self.total_tokens}")
        
        # 初始化VAR模型
        if var_config is None:
            var_config = self._get_default_var_config()
        
        # VAR需要处理所有尺度的组合token序列
        self.var_model = VAR(
            vocab_size=max(self.codebook_sizes),  # 使用最大的词汇表
            embed_dim=var_config.get('embed_dim', 1024),
            depth=var_config.get('depth', 16),
            num_heads=var_config.get('num_heads', 16),
            sequence_length=self.total_tokens,
            class_dropout_prob=var_config.get('class_dropout_prob', 0.1)
        )
        
        print(f"🚀 VAR模型初始化:")
        print(f"  - 词汇表大小: {max(self.codebook_sizes)}")
        print(f"  - 嵌入维度: {var_config.get('embed_dim', 1024)}")
        print(f"  - 序列长度: {self.total_tokens}")
        
        # 组织学特征处理器（支持动态维度适配）
        self.base_histology_dim = histology_feature_dim
        self.histology_processors = nn.ModuleDict()
        
        print(f"✅ VAR_ST_Complete初始化完成")
    
    def _get_default_vqvae_config(self, scale: int) -> Dict:
        """为不同尺度生成默认VQVAE配置"""
        # 较大尺度使用更大的码本和更复杂的模型
        base_size = 1024
        scale_factor = scale  # 尺度越大，模型越复杂
        
        return {
            'hidden_dim': min(512, 128 + scale * 32),  # 尺度越大，隐藏层越大
            'latent_dim': 32,
            'num_embeddings': min(8192, base_size * scale_factor),  # 尺度越大，码本越大
            'commitment_cost': 0.25
        }
    
    def _get_default_var_config(self) -> Dict:
        """生成默认VAR配置"""
        return {
            'embed_dim': 1024,
            'depth': 16,
            'num_heads': 16,
            'class_dropout_prob': 0.1
        }
    
    def _get_histology_processor(self, input_dim: int) -> nn.Module:
        """获取或创建对应维度的组织学特征处理器"""
        key = str(input_dim)
        
        if key not in self.histology_processors:
            # 创建新的处理器
            if input_dim == self.base_histology_dim:
                # 维度匹配，使用恒等映射
                processor = nn.Identity()
            else:
                # 维度不匹配，使用线性变换
                processor = nn.Sequential(
                    nn.Linear(input_dim, self.base_histology_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
            
            # 确保处理器移动到正确的设备
            device = next(self.parameters()).device
            processor = processor.to(device)
            
            self.histology_processors[key] = processor
            print(f"🔧 创建组织学特征处理器: {input_dim} → {self.base_histology_dim} (设备: {device})")
        
        return self.histology_processors[key]
    
    def forward_training(
        self,
        gene_expression: torch.Tensor,
        histology_features: torch.Tensor,
        positions: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        训练阶段前向传播 - 真正的空间多尺度模式
        
        Args:
            gene_expression: [B, N, num_genes] - spots的基因表达
            histology_features: [B, N, feature_dim] - spots的组织学特征
            positions: [B, N, 2] - spots的空间坐标
            class_labels: [B] - 条件类别标签(可选)
        
        Returns:
            Dict包含所有损失和预测结果
        """
        B, N, num_genes = gene_expression.shape
        device = gene_expression.device
        
        print(f"📊 VAR-ST训练前向传播:")
        print(f"   - 基因表达: {gene_expression.shape}")
        print(f"   - 组织学特征: {histology_features.shape}")
        print(f"   - 空间位置: {positions.shape}")
        
        # 动态适配组织学特征维度
        actual_hist_dim = histology_features.shape[-1]
        if actual_hist_dim != self.base_histology_dim:
            print(f"🔧 检测到组织学特征维度不匹配: 期望{self.base_histology_dim}, 实际{actual_hist_dim}")
            print(f"   - 自动适配: {'UNI编码器(1024维)' if actual_hist_dim == 1024 else 'CONCH编码器(512维)'}")
        
        histology_processor = self._get_histology_processor(actual_hist_dim)
        processed_hist = histology_processor(histology_features)  # [B, N, base_hist_dim]
        
        # 生成类别标签 (如果没有提供)
        if class_labels is None:
            # 使用组织学特征的统计量作为类别标签
            hist_stats = torch.mean(processed_hist.view(B, -1), dim=1) * 1000
            class_labels = hist_stats.long() % 1000  # [B]
        
        print(f"   - 类别标签: {class_labels.shape}")
        
        # Stage 1: 空间多尺度组织
        print(f"🗂️ Stage 1: 空间多尺度组织")
        multiscale_expressions = self.spatial_organizer.organize_multiscale(
            gene_expression, positions
        )
        
        # Stage 2: 多尺度VQVAE编码
        print(f"🔧 Stage 2: 多尺度VQVAE编码")
        all_tokens = []
        all_vqvae_losses = []
        
        for scale_idx, scale_expression in enumerate(multiscale_expressions):
            scale = self.spatial_scales[scale_idx]
            scale_vqvae = self.vqvaes[scale_idx]
            
            print(f"   - 编码尺度 {scale}×{scale}: {scale_expression.shape}")
            
            # 重塑为批量处理格式
            B, num_cells, num_genes = scale_expression.shape
            scale_expression_flat = scale_expression.view(-1, num_genes)  # [B*num_cells, num_genes]
            
            # VQVAE编码
            vq_result = scale_vqvae.encode_to_tokens(scale_expression_flat)
            tokens = vq_result['tokens']  # [B*num_cells, 1] or [B*num_cells]
            vq_loss = vq_result['loss']
            
            # 重塑回原始批次格式
            if tokens.dim() == 2 and tokens.shape[1] == 1:
                tokens = tokens.squeeze(1)  # [B*num_cells]
            tokens = tokens.view(B, num_cells)  # [B, num_cells]
            
            all_tokens.append(tokens)
            all_vqvae_losses.append(vq_loss)
            
            print(f"     -> tokens: {tokens.shape}, loss: {vq_loss.item():.4f}")
        
        # Stage 3: 组合tokens序列
        print(f"🔗 Stage 3: 组合tokens序列")
        combined_tokens = torch.cat(all_tokens, dim=1)  # [B, total_tokens]
        print(f"   - 组合tokens: {combined_tokens.shape}")
        
        # Stage 4: VAR自回归训练
        print(f"🚀 Stage 4: VAR自回归训练")
        var_result = self.var_model.forward_training(
            tokens=combined_tokens,
            class_labels=class_labels,
            cfg=1.0,  # 训练时不使用CFG
            cond_drop_prob=0.1
        )
        
        # Stage 5: 重建验证
        print(f"🔄 Stage 5: 重建验证")
        with torch.no_grad():
            # 从tokens重建多尺度表达
            reconstructed_multiscale = self._decode_multiscale_from_tokens(all_tokens)
            
            # 从多尺度重建原始spots表达
            reconstructed_expression = self.spatial_organizer.reconstruct_from_multiscale(
                reconstructed_multiscale, positions, reconstruction_method='finest_scale'
            )
        
        # 计算总损失
        total_vqvae_loss = sum(all_vqvae_losses) / len(all_vqvae_losses)
        var_loss = var_result['loss']
        
        # 空间重建损失
        spatial_recon_loss = F.mse_loss(reconstructed_expression, gene_expression)
        
        # 组合损失
        total_loss = var_loss + 0.1 * total_vqvae_loss + 0.1 * spatial_recon_loss
        
        print(f"📊 损失统计:")
        print(f"   - VAR损失: {var_loss.item():.4f}")
        print(f"   - VQVAE损失: {total_vqvae_loss.item():.4f}")
        print(f"   - 空间重建损失: {spatial_recon_loss.item():.4f}")
        print(f"   - 总损失: {total_loss.item():.4f}")
        
        return {
            'loss': total_loss,
            'var_loss': var_loss,
            'vqvae_loss': total_vqvae_loss,
            'spatial_recon_loss': spatial_recon_loss,
            'predictions': reconstructed_expression,
            'targets': gene_expression,
            'tokens': combined_tokens,
            'multiscale_expressions': multiscale_expressions
        }
    
    def forward_inference(
        self,
        histology_features: torch.Tensor,
        positions: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.5,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0,
        num_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        推理阶段：从组织学特征和空间位置生成基因表达预测
        
        Args:
            histology_features: [B, N, feature_dim] - 组织学特征
            positions: [B, N, 2] - 空间坐标
            class_labels: [B] - 条件类别(可选)
            cfg_scale: Classifier-free guidance缩放因子
            top_k, top_p, temperature: 采样参数
            num_samples: 生成样本数量
        
        Returns:
            Dict包含生成的基因表达预测
        """
        B, N, feature_dim = histology_features.shape
        device = histology_features.device
        
        print(f"🔮 VAR-ST推理生成:")
        print(f"   - 输入特征: {histology_features.shape}")
        print(f"   - 空间位置: {positions.shape}")
        print(f"   - CFG scale: {cfg_scale}")
        
        # 处理组织学特征
        actual_hist_dim = histology_features.shape[-1]
        histology_processor = self._get_histology_processor(actual_hist_dim)
        processed_hist = histology_processor(histology_features)
        
        # 生成类别标签
        if class_labels is None:
            hist_stats = torch.mean(processed_hist.view(B, -1), dim=1) * 1000
            class_labels = hist_stats.long() % 1000
        
        print(f"   - 类别标签: {class_labels.shape}")
        
        # VAR生成tokens序列
        print(f"🚀 VAR自回归生成tokens...")
        generated_tokens = self.var_model.autoregressive_infer_cfg(
            B=B * num_samples,
            class_labels=class_labels.repeat(num_samples) if class_labels is not None else None,
            cfg=cfg_scale,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            generator=torch.Generator(device=device).manual_seed(42)
        )
        
        print(f"   - 生成tokens: {generated_tokens.shape}")
        
        # 分解tokens到不同尺度
        split_tokens = self._split_tokens_by_scale(generated_tokens)
        
        # 从tokens解码多尺度基因表达
        print(f"🔧 从tokens解码多尺度基因表达...")
        decoded_multiscale = self._decode_multiscale_from_tokens(split_tokens)
        
        # 从多尺度重建最终基因表达
        print(f"🔄 从多尺度重建最终基因表达...")
        final_expression = self.spatial_organizer.reconstruct_from_multiscale(
            decoded_multiscale, positions, reconstruction_method='hierarchical'
        )
        
        print(f"   - 最终预测: {final_expression.shape}")
        
        return {
            'predictions': final_expression,
            'tokens': generated_tokens,
            'multiscale_expressions': decoded_multiscale
        }
    
    def _decode_multiscale_from_tokens(
        self, 
        split_tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """从分割的tokens解码多尺度基因表达"""
        decoded_expressions = []
        
        for scale_idx, scale_tokens in enumerate(split_tokens):
            scale = self.spatial_scales[scale_idx]
            scale_vqvae = self.vqvaes[scale_idx]
            
            # 重塑为VQVAE期望的格式
            B, num_cells = scale_tokens.shape
            scale_tokens_flat = scale_tokens.view(-1)  # [B*num_cells]
            
            # VQVAE解码
            decoded_flat = scale_vqvae.decode_from_tokens(scale_tokens_flat)  # [B*num_cells, num_genes]
            
            # 重塑回多尺度格式
            num_genes = decoded_flat.shape[-1]
            decoded = decoded_flat.view(B, num_cells, num_genes)  # [B, num_cells, num_genes]
            
            decoded_expressions.append(decoded)
        
        return decoded_expressions
    
    def _split_tokens_by_scale(self, combined_tokens: torch.Tensor) -> List[torch.Tensor]:
        """将组合的tokens序列分割回各个尺度"""
        B = combined_tokens.shape[0]
        split_tokens = []
        start_idx = 0
        
        for scale_idx, tokens_count in enumerate(self.tokens_per_scale):
            end_idx = start_idx + tokens_count
            scale_tokens = combined_tokens[:, start_idx:end_idx]  # [B, tokens_count]
            split_tokens.append(scale_tokens)
            start_idx = end_idx
        
        return split_tokens
    
    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """统一前向传播接口"""
        mode = inputs.get('mode', 'training')
        
        if mode == 'training':
            return self.forward_training(
                gene_expression=inputs['gene_expression'],
                histology_features=inputs['histology_features'],
                positions=inputs['positions'],
                class_labels=inputs.get('class_labels')
            )
        else:
            return self.forward_inference(
                histology_features=inputs['histology_features'],
                positions=inputs['positions'],
                class_labels=inputs.get('class_labels'),
                cfg_scale=inputs.get('cfg_scale', 1.5),
                top_k=inputs.get('top_k', 50),
                top_p=inputs.get('top_p', 0.9),
                temperature=inputs.get('temperature', 1.0),
                num_samples=inputs.get('num_samples', 1)
            ) 