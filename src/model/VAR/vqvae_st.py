#!/usr/bin/env python3
"""
Multi-scale VQVAE for Spatial Transcriptomics Gene Expression
完全适配VAR架构的多尺度VQVAE实现

关键设计:
- 5个独立的VQVAE编码器处理不同尺度: 1x1, 2x2, 3x3, 4x4, 5x5
- 输入: 基因伪图像 [B, 1, 15, 15] (225个基因)
- 输出: 多尺度离散tokens [1+4+9+16+25 = 55个tokens]
- 完全保持VAR原始VQVAE架构不变
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import math

# 导入VAR基础组件
try:
    from .var_basic_components import (
        Encoder, Decoder, VectorQuantizer2, 
        ResnetBlock, AttnBlock, Downsample, Upsample,
        nonlinearity, normalize
    )
except ImportError:
    from var_basic_components import (
        Encoder, Decoder, VectorQuantizer2,
        ResnetBlock, AttnBlock, Downsample, Upsample, 
        nonlinearity, normalize
    )


class MultiScaleVQVAE_ST(nn.Module):
    """
    多尺度VQVAE用于空间转录组学基因表达数据
    
    架构设计:
    - 5个独立的VQVAE编码器/解码器对
    - 每个尺度处理不同分辨率的基因表达模式
    - 从全局基因模式(1x1)到单基因级别(5x5)
    - 总共生成55个离散tokens供VAR使用
    """
    
    def __init__(
        self,
        gene_count: int = 225,                    # 基因数量 (15x15 = 225)
        patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5),  # 多尺度patch数量
        vocab_size: int = 8192,                   # 每个尺度的词汇表大小
        embed_dim: int = 256,                     # VQVAE嵌入维度
        hidden_dim: int = 128,                    # 编码器隐藏维度
        num_res_blocks: int = 2,                  # ResNet块数量
        dropout: float = 0.0,                     # Dropout率
        beta: float = 0.25,                       # Commitment loss权重
        using_znorm: bool = False,                # 是否使用Z-normalization
        share_decoder: bool = True,               # 是否共享解码器
        test_mode: bool = False,                  # 测试模式(冻结参数)
    ):
        super().__init__()
        
        self.gene_count = gene_count
        self.patch_nums = patch_nums
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_scales = len(patch_nums)
        self.share_decoder = share_decoder
        self.test_mode = test_mode
        
        # 计算总token数量
        self.total_tokens = sum(pn * pn for pn in patch_nums)  # 1+4+9+16+25 = 55
        
        # 基因空间维度
        self.gene_spatial_dim = int(math.sqrt(gene_count))  # 15 for 225 genes
        if self.gene_spatial_dim * self.gene_spatial_dim != gene_count:
            raise ValueError(f"基因数量{gene_count}必须是完全平方数")
        
        print(f"🧬 初始化多尺度VQVAE_ST:")
        print(f"  - 基因数量: {gene_count} ({self.gene_spatial_dim}x{self.gene_spatial_dim})")
        print(f"  - 多尺度patch: {patch_nums}")
        print(f"  - 总token数: {self.total_tokens}")
        print(f"  - 词汇表大小: {vocab_size}")
        print(f"  - 嵌入维度: {embed_dim}")
        print(f"  - 共享解码器: {share_decoder}")
        
        # 为每个尺度创建独立的编码器
        self.encoders = nn.ModuleList()
        for i, pn in enumerate(patch_nums):
            encoder = self._create_encoder(
                input_size=pn,  # 输入尺寸
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                scale_idx=i
            )
            self.encoders.append(encoder)
        
        # 为每个尺度创建量化器
        self.quantizers = nn.ModuleList()
        for i, pn in enumerate(patch_nums):
            quantizer = VectorQuantizer2(
                vocab_size=vocab_size,
                Cvae=embed_dim,
                using_znorm=using_znorm,
                beta=beta,
                v_patch_nums=(pn,),  # 每个量化器只处理单一尺度
                default_qresi_counts=1,
                quant_resi=0.5,
                share_quant_resi=1,  # 单尺度共享
            )
            self.quantizers.append(quantizer)
        
        # 解码器 - 可选择共享或独立
        if share_decoder:
            # 共享解码器 - 从所有尺度的特征重建到完整基因空间
            self.decoder = self._create_shared_decoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_size=self.gene_spatial_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout
            )
            self.decoders = None
        else:
            # 独立解码器 - 每个尺度独立解码
            self.decoders = nn.ModuleList()
            for i, pn in enumerate(patch_nums):
                decoder = self._create_decoder(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    output_size=pn,
                    target_size=self.gene_spatial_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    scale_idx=i
                )
                self.decoders.append(decoder)
            self.decoder = None
        
        # 量化前后的卷积层
        self.quant_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, 1) for _ in patch_nums
        ])
        self.post_quant_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, 1) for _ in patch_nums
        ])
        
        # 初始化权重
        self.init_weights()
        
        # 测试模式设置
        if test_mode:
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
        
    def _create_encoder(
        self, 
        input_size: int, 
        hidden_dim: int, 
        embed_dim: int,
        num_res_blocks: int,
        dropout: float,
        scale_idx: int
    ) -> nn.Module:
        """为指定尺度创建编码器"""
        
        layers = []
        
        # 输入投影层 - 从基因伪图像到隐藏维度
        layers.append(nn.Conv2d(1, hidden_dim, 3, stride=1, padding=1))
        layers.append(nn.GroupNorm(32, hidden_dim))
        layers.append(nn.SiLU())
        
        # 自适应到目标尺寸
        if input_size != self.gene_spatial_dim:
            # 下采样到目标尺寸
            while layers[-3].out_channels < embed_dim // 2:
                in_ch = layers[-3].out_channels if hasattr(layers[-3], 'out_channels') else hidden_dim
                out_ch = min(in_ch * 2, embed_dim // 2)
                
                layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
                layers.append(nn.GroupNorm(32, out_ch))
                layers.append(nn.SiLU())
        
        # ResNet块
        current_dim = layers[-3].out_channels if hasattr(layers[-3], 'out_channels') else hidden_dim
        for _ in range(num_res_blocks):
            layers.append(ResnetBlock(
                in_channels=current_dim,
                out_channels=current_dim,
                dropout=dropout
            ))
        
        # 最终投影到嵌入维度
        layers.append(nn.Conv2d(current_dim, embed_dim, 3, stride=1, padding=1))
        layers.append(nn.GroupNorm(32, embed_dim))
        layers.append(nn.SiLU())
        
        # 自适应池化到目标尺寸
        layers.append(nn.AdaptiveAvgPool2d((input_size, input_size)))
        
        return nn.Sequential(*layers)
    
    def _create_decoder(
        self,
        embed_dim: int,
        hidden_dim: int, 
        output_size: int,
        target_size: int,
        num_res_blocks: int,
        dropout: float,
        scale_idx: int
    ) -> nn.Module:
        """为指定尺度创建解码器"""
        
        layers = []
        
        # 输入投影
        layers.append(nn.Conv2d(embed_dim, hidden_dim, 3, stride=1, padding=1))
        layers.append(nn.GroupNorm(32, hidden_dim))
        layers.append(nn.SiLU())
        
        # ResNet块
        for _ in range(num_res_blocks):
            layers.append(ResnetBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                dropout=dropout
            ))
        
        # 上采样到目标尺寸
        layers.append(nn.Upsample(size=(target_size, target_size), mode='bilinear', align_corners=False))
        
        # 最终输出层
        layers.append(nn.Conv2d(hidden_dim, 1, 3, stride=1, padding=1))
        
        return nn.Sequential(*layers)
    
    def _create_shared_decoder(
        self,
        embed_dim: int,
        hidden_dim: int,
        output_size: int,
        num_res_blocks: int,
        dropout: float
    ) -> nn.Module:
        """创建共享解码器，从融合的多尺度特征重建完整基因表达"""
        
        # 特征融合层
        fusion_layers = []
        fusion_layers.append(nn.Conv2d(embed_dim * self.num_scales, hidden_dim * 2, 1))
        fusion_layers.append(nn.GroupNorm(32, hidden_dim * 2))
        fusion_layers.append(nn.SiLU())
        
        # ResNet处理
        layers = []
        layers.extend(fusion_layers)
        
        current_dim = hidden_dim * 2
        for _ in range(num_res_blocks + 1):  # 多一层ResNet处理融合特征
            layers.append(ResnetBlock(
                in_channels=current_dim,
                out_channels=hidden_dim,
                dropout=dropout
            ))
            current_dim = hidden_dim
        
        # 上采样到目标分辨率
        layers.append(nn.Upsample(size=(output_size, output_size), mode='bilinear', align_corners=False))
        
        # 最终输出
        layers.append(nn.Conv2d(hidden_dim, 1, 3, stride=1, padding=1))
        
        return nn.Sequential(*layers)
    
    def init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def encode(self, gene_pseudo_img: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        编码基因伪图像到多尺度离散tokens
        
        Args:
            gene_pseudo_img: [B, 1, H, W] 基因伪图像 (H=W=15 for 225 genes)
            
        Returns:
            ms_tokens: List[torch.Tensor] - 每个尺度的离散token indices
            ms_embeddings: List[torch.Tensor] - 每个尺度的连续嵌入
        """
        B = gene_pseudo_img.shape[0]
        
        # 调整输入到正确的基因空间维度
        if gene_pseudo_img.shape[-1] != self.gene_spatial_dim:
            gene_pseudo_img = F.interpolate(
                gene_pseudo_img, 
                size=(self.gene_spatial_dim, self.gene_spatial_dim),
                mode='bilinear', 
                align_corners=False
            )
        
        ms_tokens = []
        ms_embeddings = []
        
        for i, pn in enumerate(self.patch_nums):
            # 为当前尺度调整输入尺寸
            scale_input = F.adaptive_avg_pool2d(gene_pseudo_img, (pn, pn))
            
            # 编码
            encoded = self.encoders[i](scale_input)  # [B, embed_dim, pn, pn]
            
            # 量化前卷积
            quant_input = self.quant_convs[i](encoded)
            
            # 量化
            quantized, _, vq_loss = self.quantizers[i](quant_input)
            
            # 提取token indices
            tokens = self.quantizers[i].f_to_idxBl_or_fhat(quant_input, to_fhat=False, v_patch_nums=(pn,))
            if isinstance(tokens, list):
                tokens = tokens[0]  # [B, pn*pn]
            
            ms_tokens.append(tokens)
            ms_embeddings.append(quantized)
        
        return ms_tokens, ms_embeddings
    
    def decode(self, ms_tokens: List[torch.Tensor]) -> torch.Tensor:
        """
        从多尺度tokens解码回基因伪图像
        
        Args:
            ms_tokens: List[torch.Tensor] - 每个尺度的token indices
            
        Returns:
            reconstructed: [B, 1, H, W] 重建的基因伪图像
        """
        B = ms_tokens[0].shape[0]
        
        # 将tokens转换为embeddings
        ms_embeddings = []
        for i, tokens in enumerate(ms_tokens):
            pn = self.patch_nums[i]
            
            # 重塑tokens
            if tokens.dim() == 2:
                # [B, pn*pn] -> [B, pn, pn]
                tokens_2d = tokens.view(B, pn, pn)
            else:
                tokens_2d = tokens
            
            # 查找embeddings
            embeddings = self.quantizers[i].embedding(tokens_2d)  # [B, pn, pn, embed_dim]
            embeddings = embeddings.permute(0, 3, 1, 2).contiguous()  # [B, embed_dim, pn, pn]
            
            # 后量化卷积
            embeddings = self.post_quant_convs[i](embeddings)
            
            ms_embeddings.append(embeddings)
        
        # 解码
        if self.share_decoder:
            # 共享解码器：融合所有尺度特征
            # 将所有尺度上采样到相同大小并连接
            target_size = self.gene_spatial_dim
            fused_features = []
            
            for emb in ms_embeddings:
                upsampled = F.interpolate(emb, size=(target_size, target_size), mode='bilinear', align_corners=False)
                fused_features.append(upsampled)
            
            # 连接所有尺度特征
            fused = torch.cat(fused_features, dim=1)  # [B, embed_dim*num_scales, H, W]
            
            # 共享解码器重建
            reconstructed = self.decoder(fused)
        else:
            # 独立解码器：每个尺度独立解码后融合
            scale_outputs = []
            
            for i, emb in enumerate(ms_embeddings):
                decoded = self.decoders[i](emb)  # [B, 1, H, W]
                scale_outputs.append(decoded)
            
            # 简单平均融合
            reconstructed = torch.stack(scale_outputs, dim=0).mean(dim=0)
        
        return reconstructed
    
    def forward(self, gene_pseudo_img: torch.Tensor, return_tokens: bool = False, return_loss: bool = True):
        """
        前向传播 - 完整的编码-量化-解码过程
        
        Args:
            gene_pseudo_img: [B, 1, H, W] 输入基因伪图像
            return_tokens: 是否返回离散tokens
            return_loss: 是否计算并返回损失
            
        Returns:
            如果return_tokens=True:
                (reconstructed, ms_tokens, total_loss)
            否则:
                (reconstructed, total_loss)
        """
        # 编码
        ms_tokens, ms_embeddings = self.encode(gene_pseudo_img)
        
        # 解码
        reconstructed = self.decode(ms_tokens)
        
        # 计算损失
        if return_loss:
            # 重建损失
            recon_loss = F.mse_loss(reconstructed, gene_pseudo_img)
            
            # VQ损失(commitment loss)
            vq_loss = 0.0
            for i, emb in enumerate(ms_embeddings):
                # 每个量化器的commitment loss已经在量化过程中计算
                # 这里添加额外的正则化项
                vq_loss += F.mse_loss(emb.detach(), emb) * 0.25
            
            total_loss = recon_loss + vq_loss / len(ms_embeddings)
        else:
            total_loss = None
        
        if return_tokens:
            return reconstructed, ms_tokens, total_loss
        else:
            return reconstructed, total_loss
    
    def get_tokens(self, gene_pseudo_img: torch.Tensor) -> List[torch.Tensor]:
        """仅获取离散tokens，用于VAR训练"""
        ms_tokens, _ = self.encode(gene_pseudo_img)
        return ms_tokens
    
    def reconstruct_from_tokens(self, ms_tokens: List[torch.Tensor]) -> torch.Tensor:
        """从tokens重建，用于VAR推理"""
        return self.decode(ms_tokens)


def test_multiscale_vqvae_st():
    """测试多尺度VQVAE_ST"""
    print("🧪 测试多尺度VQVAE_ST...")
    
    # 模型参数
    gene_count = 225  # 15x15
    patch_nums = (1, 2, 3, 4, 5)
    batch_size = 4
    
    # 创建模型
    model = MultiScaleVQVAE_ST(
        gene_count=gene_count,
        patch_nums=patch_nums,
        vocab_size=4096,
        embed_dim=256,
        hidden_dim=128,
        share_decoder=True
    )
    
    # 创建测试数据
    gene_img = torch.randn(batch_size, 1, 15, 15)
    
    print(f"输入形状: {gene_img.shape}")
    
    # 测试前向传播
    reconstructed, ms_tokens, loss = model(gene_img, return_tokens=True, return_loss=True)
    
    print(f"重建形状: {reconstructed.shape}")
    print(f"重建损失: {loss:.4f}")
    print(f"多尺度tokens数量: {len(ms_tokens)}")
    
    total_tokens = 0
    for i, tokens in enumerate(ms_tokens):
        pn = patch_nums[i]
        expected_tokens = pn * pn
        actual_tokens = tokens.shape[1]
        total_tokens += actual_tokens
        print(f"  尺度{i+1} ({pn}x{pn}): {tokens.shape} - {actual_tokens} tokens (期望: {expected_tokens})")
    
    print(f"总token数: {total_tokens} (期望: {sum(pn*pn for pn in patch_nums)})")
    
    # 测试tokens重建
    reconstructed_from_tokens = model.reconstruct_from_tokens(ms_tokens)
    recon_diff = F.mse_loss(reconstructed, reconstructed_from_tokens)
    print(f"Token重建差异: {recon_diff:.6f}")
    
    print("✅ 多尺度VQVAE_ST测试完成!")


if __name__ == "__main__":
    test_multiscale_vqvae_st() 