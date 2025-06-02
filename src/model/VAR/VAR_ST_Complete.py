"""
VAR-ST Complete: 基因表达向量的视觉自回归模型

基于VAR原始设计理念的基因维度多尺度实现：
- 将基因表达向量组织为不同粒度的特征表示
- 使用VQVAE对每个尺度的基因特征进行编码
- VAR自回归生成：从粗粒度全局模式到细粒度基因表达
- 完整保留VAR的所有组件和功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .vqvae_st import VQVAE
from .var_st import VAR
from .gene_multiscale_organizer import GeneMultiscaleOrganizer


class VAR_ST_Complete(nn.Module):
    """
    VAR-ST Complete: 基因表达向量的完整VAR实现
    
    实现真正的基因维度多尺度VAR：
    1. 基因多尺度组织：将基因向量组织为不同粒度的特征表示
    2. 多尺度VQVAE编码：每个尺度使用专门的VQVAE编码器
    3. VAR自回归生成：从粗粒度到细粒度渐进式生成
    4. 基因重建：从多尺度特征重建到完整基因表达
    """
    
    def __init__(
        self,
        num_genes: int = 200,
        histology_feature_dim: int = 512,
        gene_scales: List[int] = [1, 4, 16, 64, 200],
        vqvae_configs: Optional[List[Dict]] = None,
        var_config: Optional[Dict] = None,
        gene_config: Optional[Dict] = None,
        **kwargs
    ):
        """
        初始化VAR-ST Complete模型
        
        Args:
            num_genes: 基因数量
            histology_feature_dim: 组织学特征维度 
            gene_scales: 基因多尺度特征数量列表 [1, 4, 16, 64, 200]
            vqvae_configs: 每个尺度的VQVAE配置列表
            var_config: VAR模型配置
            gene_config: 基因组织器配置
        """
        super().__init__()
        
        self.num_genes = num_genes
        self.histology_feature_dim = histology_feature_dim
        self.gene_scales = gene_scales
        self.num_scales = len(gene_scales)
        
        print(f"🧬 初始化VAR_ST_Complete (基因维度多尺度模式)")
        print(f"   - 目标基因数: {num_genes}")
        print(f"   - 组织学特征维度: {histology_feature_dim}")
        print(f"   - 基因多尺度: {gene_scales}")
        
        # 初始化基因多尺度组织器
        gene_config = gene_config or {}
        self.gene_organizer = GeneMultiscaleOrganizer(
            num_genes=num_genes,
            scales=gene_scales,
            projection_method=gene_config.get('projection_method', 'learned'),
            preserve_variance=gene_config.get('preserve_variance', True),
            normalize_features=gene_config.get('normalize_features', True)
        )
        
        # 为每个基因尺度创建专门的VQVAE编码器
        self.vqvaes = nn.ModuleList()
        self.codebook_sizes = []
        
        if vqvae_configs is None:
            vqvae_configs = [self._get_default_vqvae_config(scale) for scale in gene_scales]
        
        for scale_idx, scale in enumerate(gene_scales):
            print(f"🧬 初始化尺度 {scale} 特征的VQVAE:")
            
            vqvae_config = vqvae_configs[scale_idx] if scale_idx < len(vqvae_configs) else vqvae_configs[-1]
            
            # 每个尺度的VQVAE处理对应维度的基因特征向量 [scale]
            vqvae = VQVAE(
                input_dim=scale,
                hidden_dim=vqvae_config.get('hidden_dim', max(32, scale // 2)),
                latent_dim=vqvae_config.get('latent_dim', min(32, scale)),
                num_embeddings=vqvae_config.get('num_embeddings', min(8192, 512 * scale)),
                commitment_cost=vqvae_config.get('commitment_cost', 0.25)
            )
            
            self.vqvaes.append(vqvae)
            self.codebook_sizes.append(vqvae_config.get('num_embeddings', min(8192, 512 * scale)))
            
            print(f"  - 输入维度: {scale}")
            print(f"  - 隐藏维度: {vqvae_config.get('hidden_dim', max(32, scale // 2))}")
            print(f"  - 潜在维度: {vqvae_config.get('latent_dim', min(32, scale))}")
            print(f"  - 码本大小: {vqvae_config.get('num_embeddings', min(8192, 512 * scale))}")
        
        # 计算每个尺度的token数量（基因维度多尺度每个batch样本产生1个token）
        self.tokens_per_scale = [1 for _ in gene_scales]  # 每个尺度每个样本产生1个token
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
        # 根据特征维度调整模型复杂度
        return {
            'hidden_dim': max(32, min(512, scale * 2)),  # 隐藏层大小与特征维度成正比
            'latent_dim': min(32, max(8, scale // 2)),   # 潜在维度适配特征维度
            'num_embeddings': min(8192, max(256, 512 * scale)),  # 码本大小与特征维度成正比
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
        positions: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        训练阶段前向传播 - 基因维度多尺度模式
        
        Args:
            gene_expression: [B, num_genes] or [B, N, num_genes] - 基因表达向量
            histology_features: [B, feature_dim] or [B, N, feature_dim] - 组织学特征
            positions: Optional[torch.Tensor] - 空间坐标(基因模式下不使用)
            class_labels: [B] - 条件类别标签(可选)
        
        Returns:
            Dict包含所有损失和预测结果
        """
        # 检查是否为多spot模式（验证/测试时可能出现）
        if gene_expression.dim() == 3 and gene_expression.shape[1] > 1:
            # 多spot模式：使用专门的多spot处理方法
            print(f"🔄 检测到多spot输入 {gene_expression.shape}，切换到多spot模式")
            return self.forward_multi_spot(gene_expression, histology_features, positions, class_labels)
        
        # 单spot模式：原有的训练逻辑
        if gene_expression.dim() == 3:
            # 如果输入是[B, 1, num_genes]，压缩维度
            B, N, num_genes = gene_expression.shape
            if N == 1:
                gene_expression = gene_expression.squeeze(1)  # [B, num_genes]
                print(f"🔧 压缩单spot输入: [B, N=1, num_genes] -> [B, num_genes]")
            else:
                # 这种情况现在由forward_multi_spot处理
                raise ValueError(f"意外的多spot输入在单spot模式中: {gene_expression.shape}")
        
        if histology_features.dim() == 3:
            # 如果输入是[B, 1, feature_dim]，压缩维度
            B, N, feature_dim = histology_features.shape
            if N == 1:
                histology_features = histology_features.squeeze(1)  # [B, feature_dim]
                print(f"🔧 压缩单spot特征: [B, N=1, feature_dim] -> [B, feature_dim]")
            else:
                # 如果是多spot，取平均（兼容性处理）
                histology_features = histology_features.mean(dim=1)  # [B, feature_dim]
                print(f"🔧 平均多spot特征: [B, N={N}, feature_dim] -> [B, feature_dim]")
        
        B, num_genes = gene_expression.shape
        device = gene_expression.device
        
        print(f"📊 VAR-ST训练前向传播 (基因多尺度模式):")
        print(f"   - 基因表达: {gene_expression.shape}")
        print(f"   - 组织学特征: {histology_features.shape}")
        
        # 确保输入张量连续性
        gene_expression = gene_expression.contiguous()
        histology_features = histology_features.contiguous()
        
        # 动态适配组织学特征维度
        actual_hist_dim = histology_features.shape[-1]
        if actual_hist_dim != self.base_histology_dim:
            print(f"🔧 检测到组织学特征维度不匹配: 期望{self.base_histology_dim}, 实际{actual_hist_dim}")
            print(f"   - 自动适配: {'UNI编码器(1024维)' if actual_hist_dim == 1024 else 'CONCH编码器(512维)'}")
        
        histology_processor = self._get_histology_processor(actual_hist_dim)
        processed_hist = histology_processor(histology_features).contiguous()  # [B, base_hist_dim]
        
        # 生成类别标签 (如果没有提供)
        if class_labels is None:
            # 使用组织学特征的统计量作为类别标签
            hist_stats = torch.mean(processed_hist, dim=1) * 1000
            class_labels = hist_stats.long() % 1000  # [B]
            class_labels = class_labels.contiguous()
        
        print(f"   - 类别标签: {class_labels.shape}")
        
        # Stage 1: 基因多尺度组织
        print(f"🧬 Stage 1: 基因多尺度组织")
        multiscale_expressions = self.gene_organizer.organize_multiscale(
            gene_expression  # [B, num_genes] -> List[[B, scale_i]]
        )
        
        # Stage 2: 多尺度VQVAE编码
        print(f"🔧 Stage 2: 多尺度VQVAE编码")
        all_tokens = []
        all_vqvae_losses = []
        
        for scale_idx, scale_expression in enumerate(multiscale_expressions):
            scale = self.gene_scales[scale_idx]
            scale_vqvae = self.vqvaes[scale_idx]
            
            print(f"   - 编码尺度 {scale} 特征: {scale_expression.shape}")
            
            # 确保scale_expression连续性
            scale_expression = scale_expression.contiguous()  # [B, scale]
            
            # VQVAE编码 - 直接处理[B, scale]格式
            vq_result = scale_vqvae.encode_to_tokens(scale_expression)
            tokens = vq_result['tokens']  # [B, 1] or [B]
            vq_loss = vq_result['loss']
            
            # 确保tokens连续性并统一格式
            tokens = tokens.contiguous()
            if tokens.dim() == 2 and tokens.shape[1] == 1:
                tokens = tokens.squeeze(1).contiguous()  # [B]
            # 为VAR序列准备：每个样本每个尺度贡献1个token
            tokens = tokens.unsqueeze(1).contiguous()  # [B, 1] - 每个尺度1个token
            
            all_tokens.append(tokens)
            all_vqvae_losses.append(vq_loss)
            
            print(f"     -> tokens: {tokens.shape}, loss: {vq_loss.item():.4f}")
        
        # Stage 3: 组合tokens序列
        print(f"🔗 Stage 3: 组合tokens序列")
        combined_tokens = torch.cat(all_tokens, dim=1).contiguous()  # [B, total_tokens]
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
            split_tokens = self._split_tokens_by_scale(combined_tokens)
            reconstructed_multiscale = self._decode_multiscale_from_tokens(split_tokens)
            
            # 从多尺度重建原始基因表达
            reconstructed_expression = self.gene_organizer.reconstruct_from_multiscale(
                reconstructed_multiscale, reconstruction_method='finest_scale'
            )
            reconstructed_expression = reconstructed_expression.contiguous()
        
        # 计算总损失
        total_vqvae_loss = sum(all_vqvae_losses) / len(all_vqvae_losses)
        var_loss = var_result['loss']
        
        # 基因重建损失
        gene_recon_loss = F.mse_loss(reconstructed_expression, gene_expression)
        
        # 组合损失
        total_loss = var_loss + 0.1 * total_vqvae_loss + 0.1 * gene_recon_loss
        
        print(f"📊 损失统计:")
        print(f"   - VAR损失: {var_loss.item():.4f}")
        print(f"   - VQVAE损失: {total_vqvae_loss.item():.4f}")
        print(f"   - 基因重建损失: {gene_recon_loss.item():.4f}")
        print(f"   - 总损失: {total_loss.item():.4f}")
        
        return {
            'loss': total_loss,
            'var_loss': var_loss,
            'vqvae_loss': total_vqvae_loss,
            'gene_recon_loss': gene_recon_loss,
            'predictions': reconstructed_expression,
            'targets': gene_expression,
            'tokens': combined_tokens,
            'multiscale_expressions': multiscale_expressions,
            'predicted_expression': reconstructed_expression,
            'logits': reconstructed_expression
        }

    def forward_multi_spot(
        self,
        gene_expression: torch.Tensor,  # [B, N, num_genes]
        histology_features: torch.Tensor,  # [B, N, feature_dim] 
        positions: Optional[torch.Tensor] = None,  # [B, N, 2]
        class_labels: Optional[torch.Tensor] = None  # [B]
    ) -> Dict[str, torch.Tensor]:
        """
        多spot前向传播 - 独立预测每个spot的基因表达
        
        这个方法专门处理验证/测试时的多spot输入，
        为每个spot独立进行基因多尺度预测。
        
        Args:
            gene_expression: [B, N, num_genes] - 多个spots的基因表达
            histology_features: [B, N, feature_dim] - 多个spots的组织学特征
            positions: Optional[B, N, 2] - 空间位置(基因模式下不使用)
            class_labels: Optional[B] - 条件类别(扩展到所有spots)
        
        Returns:
            Dict包含多spot预测结果
        """
        B, N, num_genes = gene_expression.shape
        device = gene_expression.device
        
        print(f"🌟 VAR-ST多spot前向传播:")
        print(f"   - 输入shape: {gene_expression.shape}")
        print(f"   - Batch size: {B}, Spots per sample: {N}")
        print(f"   - 组织学特征: {histology_features.shape}")
        
        # 重塑输入：[B, N, *] -> [B*N, *] 以便独立处理每个spot
        gene_expr_flat = gene_expression.view(B * N, num_genes).contiguous()  # [B*N, num_genes]
        
        # 处理组织学特征
        if histology_features.dim() == 3:
            hist_feat_flat = histology_features.view(B * N, -1).contiguous()  # [B*N, feature_dim]
        else:
            # 如果组织学特征是[B, feature_dim]，需要扩展到[B*N, feature_dim]
            hist_feat_flat = histology_features.unsqueeze(1).expand(-1, N, -1).view(B * N, -1).contiguous()
        
        # 处理类别标签
        if class_labels is not None:
            if class_labels.dim() == 1 and class_labels.shape[0] == B:
                # [B] -> [B*N]
                class_labels_flat = class_labels.unsqueeze(1).expand(-1, N).view(B * N).contiguous()
            else:
                class_labels_flat = class_labels
        else:
            class_labels_flat = None
        
        print(f"   - 重塑后基因表达: {gene_expr_flat.shape}")
        print(f"   - 重塑后组织学特征: {hist_feat_flat.shape}")
        
        # 调用单spot训练方法处理每个spot
        spot_results = self.forward_training(
            gene_expression=gene_expr_flat,
            histology_features=hist_feat_flat,
            positions=None,  # 基因模式下不使用空间位置
            class_labels=class_labels_flat
        )
        
        # 重塑输出：[B*N, *] -> [B, N, *]
        predictions = spot_results['predictions']  # [B*N, num_genes]
        predictions = predictions.view(B, N, num_genes).contiguous()  # [B, N, num_genes]
        
        targets = gene_expression  # 保持原始目标格式 [B, N, num_genes]
        
        print(f"   - 输出预测shape: {predictions.shape}")
        print(f"   - 输出目标shape: {targets.shape}")
        
        # 返回多spot格式的结果
        return {
            'loss': spot_results['loss'],  # 标量损失
            'var_loss': spot_results['var_loss'],
            'vqvae_loss': spot_results['vqvae_loss'], 
            'gene_recon_loss': spot_results['gene_recon_loss'],
            'predictions': predictions,  # [B, N, num_genes]
            'targets': targets,  # [B, N, num_genes]
            'tokens': spot_results['tokens'],  # [B*N, total_tokens]
            'multiscale_expressions': spot_results['multiscale_expressions'],
            'predicted_expression': predictions,  # [B, N, num_genes]
            'logits': predictions  # [B, N, num_genes]
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
        """
        推理阶段：从组织学特征生成基因表达预测
        
        Args:
            histology_features: [B, feature_dim] - 组织学特征
            positions: Optional[torch.Tensor] - 空间坐标(基因模式下不使用)
            class_labels: [B] - 条件类别(可选)
            cfg_scale: Classifier-free guidance缩放因子
            top_k, top_p, temperature: 采样参数
            num_samples: 生成样本数量
        
        Returns:
            Dict包含生成的基因表达预测
        """
        # 处理输入维度
        if histology_features.dim() == 3:
            B, N, feature_dim = histology_features.shape
            if N == 1:
                histology_features = histology_features.squeeze(1)  # [B, feature_dim]
            else:
                histology_features = histology_features.mean(dim=1)  # [B, feature_dim]
            print(f"🔧 转换多spot特征: [B, N={N}, feature_dim] -> [B, feature_dim]")
        
        B, feature_dim = histology_features.shape
        device = histology_features.device
        
        print(f"🔮 VAR-ST推理生成 (基因多尺度模式):")
        print(f"   - 输入特征: {histology_features.shape}")
        print(f"   - CFG scale: {cfg_scale}")
        
        # 处理组织学特征
        actual_hist_dim = histology_features.shape[-1]
        histology_processor = self._get_histology_processor(actual_hist_dim)
        processed_hist = histology_processor(histology_features)
        
        # 生成类别标签
        if class_labels is None:
            hist_stats = torch.mean(processed_hist, dim=1) * 1000
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
            temperature=temperature
        )
        
        if isinstance(generated_tokens, list):
            # 如果VAR返回list，转换为tensor
            generated_tokens = torch.cat(generated_tokens, dim=1)  # [B*num_samples, total_tokens]
        
        print(f"   - 生成tokens: {generated_tokens.shape}")
        
        # 从tokens解码多尺度基因表达
        print(f"🔄 从tokens解码基因表达...")
        split_tokens = self._split_tokens_by_scale(generated_tokens)
        decoded_multiscale = self._decode_multiscale_from_tokens(split_tokens)
        
        # 从多尺度重建最终基因表达
        print(f"🔄 从多尺度重建最终基因表达...")
        final_expression = self.gene_organizer.reconstruct_from_multiscale(
            decoded_multiscale, reconstruction_method='finest_scale'
        )
        
        # 重塑为原始批次大小
        if num_samples > 1:
            final_expression = final_expression.view(B, num_samples, -1)
        
        print(f"✅ 推理完成: {final_expression.shape}")
        
        return {
            'predictions': final_expression,
            'generated_tokens': generated_tokens,
            'multiscale_expressions': decoded_multiscale,
            'predicted_expression': final_expression,
            'logits': final_expression
        }
    
    def _decode_multiscale_from_tokens(
        self, 
        split_tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """从分割的tokens解码多尺度基因表达"""
        decoded_expressions = []
        
        for scale_idx, scale_tokens in enumerate(split_tokens):
            scale = self.gene_scales[scale_idx]
            scale_vqvae = self.vqvaes[scale_idx]
            
            # 确保scale_tokens连续性
            scale_tokens = scale_tokens.contiguous()
            
            # 处理tokens格式：[B, 1] -> [B]
            if scale_tokens.dim() == 2 and scale_tokens.shape[1] == 1:
                scale_tokens = scale_tokens.squeeze(1).contiguous()  # [B]
            
            # VQVAE解码
            decoded = scale_vqvae.decode_from_tokens(scale_tokens)  # [B, scale]
            decoded = decoded.contiguous()
            
            decoded_expressions.append(decoded)
            
            print(f"   - 解码尺度{scale_idx+1}: tokens{scale_tokens.shape} -> 表达{decoded.shape}")
        
        return decoded_expressions
    
    def _split_tokens_by_scale(self, combined_tokens: torch.Tensor) -> List[torch.Tensor]:
        """将组合的tokens序列分割回各个尺度"""
        # 确保输入tokens连续性
        combined_tokens = combined_tokens.contiguous()
        
        B = combined_tokens.shape[0]
        split_tokens = []
        start_idx = 0
        
        for scale_idx, tokens_count in enumerate(self.tokens_per_scale):
            end_idx = start_idx + tokens_count
            scale_tokens = combined_tokens[:, start_idx:end_idx].contiguous()  # [B, tokens_count]
            split_tokens.append(scale_tokens)
            start_idx = end_idx
        
        return split_tokens
    
    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """统一前向传播接口"""
        mode = inputs.get('mode', 'training')
        
        # 检查输入数据格式
        gene_expression = inputs.get('gene_expression')
        histology_features = inputs.get('histology_features')
        
        # 智能模式检测
        if mode == 'training':
            # 训练模式：优先使用forward_training
            # 但如果检测到多spot输入，自动切换到多spot模式
            if (gene_expression is not None and 
                gene_expression.dim() == 3 and 
                gene_expression.shape[1] > 1):
                print(f"🔍 训练模式检测到多spot输入，自动使用多spot处理")
                return self.forward_training(
                    gene_expression=gene_expression,
                    histology_features=histology_features,
                    positions=inputs.get('positions'),
                    class_labels=inputs.get('class_labels')
                )
            else:
                return self.forward_training(
                    gene_expression=gene_expression,
                    histology_features=histology_features,
                    positions=inputs.get('positions'),
                    class_labels=inputs.get('class_labels')
                )
        else:
            # 推理模式：使用forward_inference
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