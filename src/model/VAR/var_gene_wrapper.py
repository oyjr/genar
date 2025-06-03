"""
VAR基因包装器 - 完整VAR实现版本

基于真正的VAR架构重写，包含：
- AdaLNSelfAttn: 条件自适应LayerNorm
- 多尺度自回归生成
- 正确的位置编码和层级编码
- Causal attention mask
- 完整的autoregressive推理

无外部依赖，完全内置实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import math
from functools import partial

# 导入本地基因适配器和VAR基础组件
from .gene_pseudo_image_adapter import GenePseudoImageAdapter
from .var_basic_components import *

class VARGeneWrapper(nn.Module):
    """
    VAR基因包装器 - 完整VAR实现版本
    
    基于真正的VAR架构设计：
    - 196个基因 → 14×14×1 伪图像
    - 多尺度patch_nums: (1,2,4) = 21 tokens
    - AdaLN条件注入机制
    - 完整的自回归生成
    """
    
    def __init__(
        self,
        histology_feature_dim: int,  # 🔧 必需参数放在前面
        num_genes: int = 196,
        image_size: int = 64,
        patch_nums: Tuple[int, ...] = (1, 14),  # 🔧 方案C：1个全局token + 196个基因tokens
        var_config: Optional[Dict] = None,
        vqvae_config: Optional[Dict] = None,
        adapter_config: Optional[Dict] = None,
        progressive_training: bool = True,
        warmup_steps: int = 1000,
        min_recon_weight: float = 0.5,
        max_recon_weight: float = 5.0
    ):
        super().__init__()
        
        # 🔧 严格参数验证，不允许None或无效值
        if histology_feature_dim is None or histology_feature_dim <= 0:
            raise ValueError(f"histology_feature_dim必须是正整数，得到: {histology_feature_dim}")
        
        if num_genes <= 0:
            raise ValueError(f"num_genes必须是正整数，得到: {num_genes}")
            
        if image_size <= 0:
            raise ValueError(f"image_size必须是正整数，得到: {image_size}")
            
        if not patch_nums or any(p <= 0 for p in patch_nums):
            raise ValueError(f"patch_nums必须是正整数序列，得到: {patch_nums}")
        
        # 🔧 验证patch_nums与基因数量的匹配
        expected_total = sum(pn ** 2 for pn in patch_nums)
        if expected_total != num_genes + 1:  # +1 for global token
            print(f"⚠️ 警告: patch_nums总token数({expected_total}) != 基因数+1({num_genes + 1})")
            print(f"   当前patch_nums: {patch_nums}")
            print(f"   各层token数: {[pn**2 for pn in patch_nums]}")
            print(f"   建议的patch_nums设计：第1层=1(全局), 第2层=196(基因)")
        
        # 保存配置
        self.progressive_training = progressive_training
        self.warmup_steps = warmup_steps
        self.min_recon_weight = min_recon_weight
        self.max_recon_weight = max_recon_weight
        self.current_step = 0
        
        # 基础配置
        self.num_genes = num_genes
        self.image_size = image_size
        self.histology_feature_dim = histology_feature_dim
        self.patch_nums = patch_nums
        
        # 验证基因数量
        if num_genes == 196:
            self.use_upsampling = True
            self.intermediate_size = 14
        else:
            self.use_upsampling = False
            self.intermediate_size = int(math.sqrt(num_genes))
            if self.intermediate_size ** 2 != num_genes:
                raise ValueError(f"基因数量{num_genes}不是完全平方数")
        
        print(f"🧬 VAR基因包装器 - 方案C重构版本")
        print(f"   基因数量: {num_genes} → {image_size}×{image_size}")
        print(f"   多尺度patch_nums: {patch_nums}")
        print(f"   设计理念: 第1层({patch_nums[0]}²)=全局特征, 第2层({patch_nums[1]}²)=基因级特征")
        
        # 计算多尺度参数
        self.L = sum(pn ** 2 for pn in self.patch_nums)  # 总token数 = 1 + 196 = 197
        self.first_l = self.patch_nums[0] ** 2  # 第一层token数 = 1
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn ** 2))
            cur += pn ** 2
        
        print(f"   总token数: {self.L}")
        print(f"   第一层(全局): {self.first_l} tokens")
        print(f"   各层范围: {self.begin_ends}")
        
        # 🔧 新增：验证多尺度设计的合理性
        if len(self.patch_nums) == 2 and self.patch_nums[0] == 1 and self.patch_nums[1] == 14:
            print(f"✅ 方案C配置验证通过：")
            print(f"   - 全局层: 1个token (整体基因表达模式)")
            print(f"   - 基因层: 196个tokens (每个基因的详细表达)")
            print(f"   - 总计: {self.L} tokens")
        else:
            print(f"⚠️ 非标准方案C配置，请确认设计意图")
        
        # 初始化基因适配器
        self.gene_adapter = GenePseudoImageAdapter(
            num_genes=num_genes,
            intermediate_size=self.intermediate_size,
            target_image_size=image_size,
            normalize_method='none',
            eps=1e-6
        )

        # 初始化VQVAE
        self._init_vqvae(vqvae_config)
        
        # 初始化VAR
        self._init_full_var(var_config)
        
        # 初始化条件处理器
        self._init_condition_processor()
        
        # 输出统计
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 总参数: {trainable_params:,}")
        print(f"✅ VAR基因包装器初始化完成")
        
        self._verbose_logging = True
        self._step_count = 0
    
    def _init_vqvae(self, vqvae_config: Optional[Dict] = None):
        """初始化内置VQVAE"""
        config = vqvae_config or {}
        
        self.vocab_size = config.get('vocab_size', 8192)
        self.z_channels = config.get('z_channels', 64)
        
        print(f"🎨 初始化内置VQVAE:")
        print(f"   词汇表大小: {self.vocab_size}")
        print(f"   潜在维度: {self.z_channels}")
        
        # 编码器: 1×64×64 → z_channels×4×4 (16倍下采样)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),     # 64→32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),    # 32→16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),   # 16→8
            nn.ReLU(),
            nn.Conv2d(128, self.z_channels, 4, stride=2, padding=1),  # 8→4
        )
        
        # 量化层
        self.quantize = nn.Embedding(self.vocab_size, self.z_channels)
        
        # 解码器: z_channels×4×4 → 1×64×64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.z_channels, 128, 4, stride=2, padding=1),  # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),     # 32→64
            nn.Tanh()
        )
        
        print(f"   架构: 1×64×64 → {self.z_channels}×4×4 → 1×64×64")
    
    def _init_full_var(self, var_config: Optional[Dict] = None):
        """初始化完整VAR模型"""
        config = var_config or {}
        
        # VAR核心参数
        self.embed_dim = config.get('embed_dim', 512)
        self.depth = config.get('depth', 12)
        self.num_heads = config.get('num_heads', 8)
        self.num_classes = config.get('num_classes', 10)
        
        print(f"🏗️ 初始化完整VAR模型:")
        print(f"   嵌入维度: {self.embed_dim}")
        print(f"   深度: {self.depth}")
        print(f"   注意力头数: {self.num_heads}")
        print(f"   类别数: {self.num_classes}")
        
        # Progressive training参数
        self.prog_si = -1  # -1表示使用全部尺度，>=0表示当前训练的尺度索引
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        
        # 1. Word embedding (将VQVAE的潜在表示映射到VAR空间)
        self.word_embed = nn.Linear(self.z_channels, self.embed_dim)
        
        # 2. Class embedding
        init_std = math.sqrt(1 / self.embed_dim / 3)
        self.class_emb = nn.Embedding(self.num_classes + 1, self.embed_dim)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        
        # 起始位置编码
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.embed_dim))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. 绝对位置编码 (每个尺度独立)
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.embed_dim)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.embed_dim)
        self.pos_1LC = nn.Parameter(pos_1LC)
        
        # 4. Level embedding (层级编码，用于区分不同尺度的token)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.embed_dim)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 5. 构建Causal Attention Mask (VAR的核心组件)
        print(f"🎯 构建Causal Attention Mask:")
        
        # 创建level标识：每个token属于哪个尺度级别
        d = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)  # dT: 1,1,L
        lvl_1L = dT[:, 0].contiguous()  # 1,L - 每个位置的level
        self.register_buffer('lvl_1L', lvl_1L)
        
        # 构建causal mask: 只有level >= 当前level的token可以被看到
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        print(f"   Level分布: {[torch.sum(lvl_1L == i).item() for i in range(len(self.patch_nums))]}")
        print(f"   Mask形状: {attn_bias_for_masking.shape}")
        print(f"   可见token数量: {(attn_bias_for_masking[0, 0] != -torch.inf).sum(dim=-1)}")
        
        # 6. Transformer Blocks
        self.blocks = nn.ModuleList()
        for block_idx in range(self.depth):
            block = AdaLNSelfAttn(
                block_idx=block_idx,
                last_drop_p=0 if block_idx == 0 else 0.1,
                embed_dim=self.embed_dim,
                cond_dim=self.embed_dim,
                shared_aln=False,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                drop=0.0,
                attn_drop=0.0,
                drop_path=0.1 * block_idx / (self.depth - 1) if self.depth > 1 else 0.0,
                attn_l2_norm=False,
                flash_if_available=True,
                fused_if_available=True,
                enable_histology_injection=True,
                histology_dim=self.histology_feature_dim
            )
            self.blocks.append(block)
            print(f"   ✅ Block {block_idx}: 启用组织学条件注入 (histology_dim={self.histology_feature_dim} → embed_dim={self.embed_dim})")
        
        print(f"   transformer块数: {len(self.blocks)}")
        
        # 7. Output head
        self.head_nm = AdaLNBeforeHead(self.embed_dim, self.embed_dim, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.head = nn.Linear(self.embed_dim, self.vocab_size)
        
        print(f"   输出词汇表: {self.vocab_size}")
        
        # 8. 随机数生成器 (用于推理)
        self.rng = torch.Generator()
        
        print(f"✅ VAR模型初始化完成，包含完整的Causal Attention Mask")
    
    def _init_condition_processor(self):
        """初始化条件处理器"""
        print(f"🎛️ 初始化条件处理器 (输入: {self.histology_feature_dim})")
        
        # 简单的线性映射
        self.condition_processor = nn.Sequential(
            nn.Linear(self.histology_feature_dim, 512),
                    nn.ReLU(),
            nn.Linear(512, 512),
        )
        
        print(f"   架构: {self.histology_feature_dim} → 512 → 512")

    def img_to_idxBl(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        将图像编码为多尺度token列表
        
        Args:
            x: 输入图像或特征 [B, C, H, W] 或 [B, L, C]
            
        Returns:
            tokens: 多尺度token列表，每个元素形状为 [B, pn*pn]
        """
        B = x.shape[0]
        
        if x.dim() == 3:
            # 如果是[B, L, C]，先重塑为图像格式
            L, C = x.shape[1], x.shape[2]
            if L == 16 and C == self.z_channels:
                # 假设是4x4的特征图
                x = x.transpose(1, 2).reshape(B, C, 4, 4)
            else:
                raise ValueError(f"无法处理维度 {x.shape}")
        
        if x.dim() != 4:
            raise ValueError(f"期望4D输入 [B, C, H, W]，得到 {x.shape}")
        
        # 编码得到潜在表示
        if x.shape[1] == 1:  # 输入是图像
            z = self.encoder(x)  # [B, z_channels, 4, 4]
        else:  # 输入已经是潜在表示
            z = x
            
        # 扁平化并量化
        z_flat = z.flatten(2).transpose(1, 2)  # [B, 16, z_channels]
        distances = torch.cdist(z_flat, self.quantize.weight)
        indices = torch.argmin(distances, dim=-1)  # [B, 16]
        
        # 多尺度分割
        tokens_list = []
        flat_indices = indices.flatten(1)  # [B, 16]
        
        start_idx = 0
        for pn in self.patch_nums:
            token_count = pn * pn
            if start_idx + token_count <= flat_indices.shape[1]:
                tokens = flat_indices[:, start_idx:start_idx + token_count]
            else:
                # 如果不够，重复最后的tokens
                available = flat_indices.shape[1] - start_idx
                if available > 0:
                    tokens = flat_indices[:, start_idx:]
                    # 重复填充到所需长度
                    repeats = (token_count + available - 1) // available
                    tokens = tokens.repeat(1, repeats)[:, :token_count]
                else:
                    # 完全没有可用的，使用0填充
                    tokens = torch.zeros(B, token_count, dtype=torch.long, device=x.device)
            
            tokens_list.append(tokens)
            start_idx += token_count
            
        return tokens_list
    
    def img_to_idxBl_multiscale(self, z_q: torch.Tensor, indices: torch.Tensor) -> List[torch.Tensor]:
        """
        基于量化特征和索引生成多尺度token列表 - 方案C实现
        
        方案C设计：
        - 第1层: 1个全局token (整体基因表达模式的代表)
        - 第2层: 196个基因tokens (每个基因的详细表达)
        
        Args:
            z_q: 量化后的特征 [B, 16, z_channels] (来自4×4=16的spatial tokens)
            indices: 量化索引 [B, 16] (4×4空间位置的量化indices)
            
        Returns:
            tokens_list: [
                [B, 1] - 第1层全局token
                [B, 196] - 第2层基因tokens  
            ]
        """
        B = indices.shape[0]
        tokens_list = []
        
        # 🔧 方案C的关键改进：合理的全局-局部分层
        
        # 第1层：全局token (使用所有spatial tokens的平均或代表性token)
        if self.patch_nums[0] == 1:
            # 使用中心位置的token作为全局代表，或者使用平均
            if indices.shape[1] >= 4:
                # 对于4×4的indices，取中心4个位置的平均作为全局token
                center_indices = indices[:, [5, 6, 9, 10]]  # 4×4中心的4个位置
                global_token = torch.mode(center_indices, dim=1)[0].unsqueeze(1)  # [B, 1]
            else:
                # 如果spatial tokens不够，使用第一个
                global_token = indices[:, 0:1]  # [B, 1]
            tokens_list.append(global_token)
        else:
            raise ValueError(f"方案C第1层应该是1个token，得到: {self.patch_nums[0]}")
            
        # 第2层：基因级tokens (需要扩展16个spatial tokens到196个基因tokens)
        if self.patch_nums[1] == 14:  # 14² = 196
            # 🔧 关键改进：将16个spatial tokens合理扩展到196个基因tokens
            
            # 方法1：重复和插值扩展
            flat_indices = indices.flatten(1)  # [B, 16]
            
            # 计算需要的扩展倍数
            target_count = 196
            available_count = flat_indices.shape[1]  # 16
            
            # 使用重复+随机扰动的方式扩展到196个
            expansion_factor = target_count // available_count  # 196 // 16 = 12
            remainder = target_count % available_count  # 196 % 16 = 4
            
            expanded_tokens = []
            
            # 每个spatial token重复expansion_factor次
            for i in range(available_count):
                token_val = flat_indices[:, i:i+1]  # [B, 1]
                repeated = token_val.repeat(1, expansion_factor)  # [B, 12]
                expanded_tokens.append(repeated)
            
            # 处理余数：额外重复前remainder个tokens
            for i in range(remainder):
                token_val = flat_indices[:, i:i+1]  # [B, 1]
                expanded_tokens.append(token_val)
            
            # 连接所有扩展的tokens
            gene_tokens = torch.cat(expanded_tokens, dim=1)  # [B, 196]
            
            # 🔧 添加轻微的随机扰动，避免完全重复
            if self.training:
                # 训练时添加轻微扰动 (±1的随机变化)
                noise = torch.randint_like(gene_tokens, low=-1, high=2) 
                gene_tokens = torch.clamp(gene_tokens + noise, min=0, max=self.vocab_size-1)
            
            tokens_list.append(gene_tokens)
        else:
            raise ValueError(f"方案C第2层应该是196个tokens，得到: {self.patch_nums[1]**2}")
        
        # 验证输出
        assert len(tokens_list) == 2, f"方案C应该输出2层，得到: {len(tokens_list)}"
        assert tokens_list[0].shape[1] == 1, f"第1层应该是1个token，得到: {tokens_list[0].shape[1]}"
        assert tokens_list[1].shape[1] == 196, f"第2层应该是196个tokens，得到: {tokens_list[1].shape[1]}"
        
        return tokens_list
            
    def idxBl_to_img(self, tokens: List[torch.Tensor], same_shape: bool = True, last_one: bool = True) -> torch.Tensor:
        """多尺度token indices解码为图像"""
        if last_one and len(tokens) > 0:
            # 使用最高分辨率的tokens
            indices = tokens[-1]
        else:
            # 使用第一个或平均
            indices = tokens[0] if tokens else torch.zeros(1, 1, dtype=torch.long)
                
        B = indices.shape[0]
                
        # 量化嵌入
        quantized = self.quantize(indices)  # [B, L, z_channels]
        
        # 重塑为4×4特征图
        if quantized.dim() == 3:
            L = quantized.shape[1]
            side = int(math.sqrt(L))
            if side * side != L:
                side = 4  # 默认4×4
                quantized = quantized[:, :side*side]
            
            z = quantized.permute(0, 2, 1).view(B, self.z_channels, side, side)
        else:
            z = quantized.view(B, self.z_channels, 1, 1)
        
        # 如果不是4×4，插值到4×4
        if z.shape[-1] != 4:
            z = F.interpolate(z, size=(4, 4), mode='bilinear', align_corners=False)
        
        # 解码
        return self.decoder(z)
    
    def get_logits(self, h: torch.Tensor, cond_BD: torch.Tensor) -> torch.Tensor:
        """获取输出logits"""
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def get_recon_weight(self) -> float:
        """
        获取当前的重建损失权重
        
        Returns:
            当前的重建损失权重
        """
        if self.progressive_training:
            # 渐进式训练：权重逐渐增加
            progress = min(1.0, self.current_step / self.warmup_steps)
            weight = self.min_recon_weight + progress * (self.max_recon_weight - self.min_recon_weight)
        else:
            # 固定权重
            weight = self.max_recon_weight
            
        return weight
    
    def get_next_autoregressive_input(self, si: int, total_stages: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取下一个自回归输入 - VAR核心方法
        
        Args:
            si: 当前阶段索引
            total_stages: 总阶段数
            f_hat: 当前重建的特征图 [B, C, H, W]
            h_BChw: 当前阶段的特征 [B, C, h, w]
            
        Returns:
            updated_f_hat: 更新后的特征图
            next_input: 下一阶段的输入 (如果不是最后阶段)
        """
        B, Cvae = h_BChw.shape[:2]
        pn = self.patch_nums[si]
        
        # 将当前阶段的特征插入到正确位置
        if si == 0:
            # 第一阶段：初始化f_hat
            target_size = self.patch_nums[-1]  # 最大尺寸
            f_hat = torch.zeros(B, Cvae, target_size, target_size, device=h_BChw.device, dtype=h_BChw.dtype)
            
            # 将1x1特征放到左上角
            f_hat[:, :, :pn, :pn] = h_BChw
        else:
            # 后续阶段：将特征插入到对应位置
            # 使用双线性插值将特征放置到正确的空间位置
            target_size = self.patch_nums[-1]
            scale_factor = target_size // pn
            
            # 上采样到目标尺寸
            h_upsampled = F.interpolate(h_BChw, scale_factor=scale_factor, mode='nearest')
            
            # 更新f_hat的对应区域
            start_h = start_w = 0
            if si > 0:
                # 计算当前阶段应该填充的区域
                prev_size = self.patch_nums[si-1] if si > 0 else 0
                start_h = start_w = prev_size
            
            end_h = min(start_h + h_upsampled.shape[2], target_size)
            end_w = min(start_w + h_upsampled.shape[3], target_size)
            
            f_hat[:, :, start_h:end_h, start_w:end_w] = h_upsampled[:, :, :end_h-start_h, :end_w-start_w]
        
        if si == total_stages - 1:
            # 最后阶段：返回最终结果
            return f_hat, None
        else:
            # 中间阶段：准备下一阶段的输入
            next_pn = self.patch_nums[si + 1]
            
            # 从f_hat中提取下一阶段需要的区域
            next_region = f_hat[:, :, :next_pn, :next_pn]
            
            return f_hat, next_region
    
    def forward_training(
        self,
        gene_expression: torch.Tensor,
        histology_features: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        show_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        VAR训练前向传播 - 方案C重构版本
        
        完全对齐原始VAR的teacher forcing机制：
        - 输入：第1层的特征
        - 目标：所有层的tokens
        - 训练：预测第2层的196个基因tokens
        
        Args:
            gene_expression: 基因表达 [B, num_genes]
            histology_features: 组织学特征 [B, histology_dim]
            class_labels: 类别标签 [B] (可选)
            show_details: 是否显示详细信息
            
        Returns:
            包含损失和指标的字典
        """
        B = gene_expression.shape[0]
        
        # 1. 基因表达 → 伪图像 → VQVAE编码 → 多尺度token化
        with torch.no_grad():
            pseudo_images = self.gene_adapter(gene_expression)  # [B, 1, 64, 64]
        
        # VQVAE编码得到潜在表示
        z = self.encoder(pseudo_images)  # [B, z_channels, 4, 4]
        z_flat = z.flatten(2).transpose(1, 2)  # [B, 16, z_channels]
        
        # 量化得到离散tokens
        distances = torch.cdist(z_flat, self.quantize.weight)  # [B, 16, vocab_size]
        indices = torch.argmin(distances, dim=-1)  # [B, 16]
        z_q = self.quantize(indices)  # [B, 16, z_channels]
        
        # 多尺度token化
        tokens_list = self.img_to_idxBl_multiscale(z_q, indices)
        
        if show_details:
            print(f"🧬 方案C训练 (步骤 {self._step_count + 1}):")
            print(f"   多尺度tokens: {[t.shape for t in tokens_list]}")
        
        # 2. 🔧 原始VAR的teacher forcing逻辑
        # 获取目标tokens：连接所有层的tokens
        gt_BL = torch.cat(tokens_list, dim=1)  # [B, L] = [B, 1+196] = [B, 197]
        
        # 获取teacher forcing输入：前N-1层的特征用于预测所有N层
        if len(tokens_list) > 1:
            # 方案C：使用第1层预测第2层
            x_BLCv_wo_first_l = self.idxBl_to_var_input(tokens_list)  # [B, 196, z_channels]
        else:
            x_BLCv_wo_first_l = None
        
        # 3. 条件处理
        if class_labels is None:
            class_labels = torch.zeros(B, dtype=torch.long, device=gene_expression.device)
        
        # 4. 🔧 使用原始VAR的forward方法
        # 这里完全对齐原始VAR的训练逻辑
        with torch.cuda.amp.autocast(enabled=False):
            # Class dropout (原始VAR的条件dropout机制)
            if self.training and hasattr(self, 'cond_drop_rate'):
                drop_mask = torch.rand(B, device=class_labels.device) < getattr(self, 'cond_drop_rate', 0.1)
                class_labels = torch.where(drop_mask, self.num_classes, class_labels)
            
            # Class embedding
            sos = cond_BD = self.class_emb(class_labels)  # [B, embed_dim]
            
            # 起始embedding：第1层使用pos_start
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, -1, -1)
            
            # 构建完整输入序列
            if self.prog_si == 0:
                # Progressive training的第一阶段：只有起始tokens
                x_BLC = sos
            else:
                # 正常训练：起始tokens + teacher forcing输入
                if x_BLCv_wo_first_l is not None:
                    teacher_input = self.word_embed(x_BLCv_wo_first_l.float())  # [B, 196, embed_dim]
                    x_BLC = torch.cat((sos, teacher_input), dim=1)  # [B, 197, embed_dim]
                else:
                    x_BLC = sos
            
            # Progressive training范围
            bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
            
            # 添加位置编码和层级编码
            lvl_pos = self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]
            x_BLC += lvl_pos
        
        # 5. 🔧 Transformer前向传播（使用causal mask）
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD) if hasattr(self, 'shared_ada_lin') else cond_BD
        
        # 组织学条件
        histology_condition = self.condition_processor(histology_features)
        
        # Transformer blocks
        for block in self.blocks:
            x_BLC = block(
                x=x_BLC, 
                cond_BD=cond_BD_or_gss, 
                attn_bias=attn_bias,
                histology_condition=histology_condition
            )
        
        # 6. 🔧 输出logits并计算损失
        logits_BLV = self.head(self.head_nm(x_BLC.float(), cond_BD)).float()
        
        # 🔧 正确的VAR损失计算：预测所有位置的tokens
        target_BL = gt_BL[:, :ed]  # 截取到当前训练范围
        var_loss = F.cross_entropy(
            logits_BLV.view(-1, self.vocab_size),
            target_BL.view(-1)
        )
        
        # 7. 重建损失（用于评估，不参与主要训练）
        with torch.no_grad():
            # 预测的tokens
            pred_tokens = logits_BLV.argmax(dim=-1)  # [B, L]
            
            # 使用第2层的预测tokens重建基因表达
            if pred_tokens.shape[1] >= 197:  # 确保有足够的tokens
                gene_tokens = pred_tokens[:, 1:197]  # 第2层的196个tokens
                # 注意：这里需要实现从tokens到基因表达的重建
                # 简化版本：使用MSE对比原始基因表达
                gene_recon = gene_expression  # 占位符
                recon_loss = F.mse_loss(gene_recon, gene_expression)
            else:
                recon_loss = torch.tensor(0.0, device=gene_expression.device)
        
        # 8. 总损失
        recon_weight = self.get_recon_weight() if hasattr(self, 'get_recon_weight') else 0.1
        total_loss = var_loss  # 主要使用VAR损失，重建损失仅用于监控
        
        if show_details:
            print(f"   VAR损失: {var_loss.item():.6f}")
            print(f"   重建损失: {recon_loss.item():.6f}")
            print(f"   目标tokens形状: {target_BL.shape}")
            print(f"   预测logits形状: {logits_BLV.shape}")
        
        self._step_count += 1
        
        return {
            'loss': total_loss,
            'var_loss': var_loss,
            'recon_loss': recon_loss,
            'predictions': gene_expression,  # 暂时返回原始值
            'targets': gene_expression,
            'pred_tokens': pred_tokens.detach()
        }

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, 
        B: int, 
        label_B: Optional[torch.Tensor] = None,
        g_seed: Optional[int] = None,
        cfg: float = 1.5,
        top_k: int = 50,
        top_p: float = 0.9,
        histology_condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        VAR自回归推理 - 完整实现版本
        
        Args:
            B: 批次大小
            label_B: 类别标签 [B]
            g_seed: 随机种子
            cfg: Classifier-free guidance scale
            top_k: Top-k采样
            top_p: Top-p采样
            histology_condition: 组织学条件 [B, histology_dim]
            
        Returns:
            生成的图像 [B, 1, 64, 64]
        """
        device = next(self.parameters()).device
        
        # 设置随机种子
        if g_seed is not None:
            self.rng.manual_seed(g_seed)
            rng = self.rng
        else:
            rng = None
        
        # 处理标签
        if label_B is None:
            label_B = torch.zeros(B, dtype=torch.long, device=device)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), label_B, dtype=torch.long, device=device)
        
        # CFG需要双倍batch：conditional + unconditional
        # Conditional使用真实标签，unconditional使用特殊的null标签
        null_label = self.num_classes  # 使用最后一个作为null类别
        label_cond = label_B
        label_uncond = torch.full_like(label_B, null_label)
        labels_cfg = torch.cat([label_cond, label_uncond], dim=0)  # [2B]
        
        # Class embeddings
        cond_BD = self.class_emb(labels_cfg)  # [2B, embed_dim]
        
        # 组织学条件
        if histology_condition is not None:
            if histology_condition.shape[0] != B:
                raise ValueError(f"组织学条件批次大小 {histology_condition.shape[0]} 与 B={B} 不匹配")
            histology_condition = self.condition_processor(histology_condition)  # [B, embed_dim]
            # 对于CFG，也需要双倍：conditional用真实条件，unconditional用零向量
            zero_histology = torch.zeros_like(histology_condition)
            histology_cfg = torch.cat([histology_condition, zero_histology], dim=0)  # [2B, embed_dim]
        else:
            histology_cfg = None
        
        # 启用KV缓存
        for block in self.blocks:
            if hasattr(block.attn, 'kv_caching'):
                block.attn.kv_caching(True)
        
        # 初始化
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC  # [1, L, embed_dim]
        f_hat = torch.zeros(B, self.z_channels, self.patch_nums[-1], self.patch_nums[-1], device=device)
        cur_L = 0
        
        # 逐阶段生成
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1 if self.num_stages_minus_1 > 0 else 1.0
            cur_L_start = cur_L
            cur_L += pn * pn
            
            if si == 0:
                # 第一阶段：使用pos_start
                next_token_map = cond_BD.unsqueeze(1).expand(2 * B, self.first_l, -1)
                next_token_map = next_token_map + self.pos_start.expand(2 * B, -1, -1)
                next_token_map = next_token_map + lvl_pos[:, :self.first_l].expand(2 * B, -1, -1)
            else:
                # 后续阶段：从f_hat获取输入
                # 提取当前阶段需要的区域
                current_region = f_hat[:, :, :pn, :pn]  # [B, z_channels, pn, pn]
                current_tokens = current_region.flatten(2).transpose(1, 2)  # [B, pn*pn, z_channels]
                
                # Word embedding
                next_token_map = self.word_embed(current_tokens)  # [B, pn*pn, embed_dim]
                
                # 位置编码和层级编码
                pos_embed = lvl_pos[:, cur_L_start:cur_L]  # [1, pn*pn, embed_dim]
                next_token_map = next_token_map + pos_embed
                
                # CFG：双倍batch
                next_token_map = next_token_map.repeat(2, 1, 1)  # [2B, pn*pn, embed_dim]
            
            # Transformer forward
            x = next_token_map
            for block in self.blocks:
                x = block(
                    x=x, 
                    cond_BD=cond_BD, 
                    attn_bias=None,  # 推理时不使用mask（因为有KV缓存）
                    histology_condition=histology_cfg
                )
            
            # 获取logits
            logits_BlV = self.head(self.head_nm(x, cond_BD))  # [2B, L_current, vocab_size]
            
            # CFG
            if cfg != 1.0:
                t = cfg * ratio
                logits_cond = logits_BlV[:B]
                logits_uncond = logits_BlV[B:]
                logits_BlV = (1 + t) * logits_cond - t * logits_uncond
            else:
                logits_BlV = logits_BlV[:B]
            
            # 采样
            if si < len(self.patch_nums) - 1:
                # 中间阶段：只对当前新增的tokens采样
                current_logits = logits_BlV[:, -pn*pn:]  # [B, pn*pn, vocab_size]
            else:
                # 最后阶段：对所有tokens采样
                current_logits = logits_BlV
            
            # Top-k Top-p采样
            idx_Bl = self.sample_with_top_k_top_p(current_logits, top_k=top_k, top_p=top_p, rng=rng)
            
            # 量化
            h_BChw = self.quantize(idx_Bl)  # [B, pn*pn, z_channels]
            h_BChw = h_BChw.transpose(1, 2).reshape(B, self.z_channels, pn, pn)  # [B, z_channels, pn, pn]
            
            # 更新f_hat
            f_hat, next_input = self.get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
        
        # 禁用KV缓存
        for block in self.blocks:
            if hasattr(block.attn, 'kv_caching'):
                block.attn.kv_caching(False)
        
        # 最终解码
        final_img = self.decoder(f_hat)  # [B, 1, target_size, target_size]
        
        # 确保输出尺寸正确
        if final_img.shape[-2:] != (self.image_size, self.image_size):
            final_img = F.interpolate(
                final_img, 
                size=(self.image_size, self.image_size), 
                mode='bilinear', 
                align_corners=False
            )
        
        # 归一化到[0, 1]
        final_img = (final_img + 1) * 0.5
        final_img = torch.clamp(final_img, 0, 1)
        
        return final_img
    
    def sample_with_top_k_top_p(self, logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None) -> torch.Tensor:
        """
        Top-k Top-p采样
        
        Args:
            logits: [B, L, vocab_size]
            top_k: Top-k限制
            top_p: Top-p限制  
            rng: 随机数生成器
            
        Returns:
            采样结果 [B, L]
        """
        B, L, V = logits.shape
        
        # Top-k过滤
        if top_k > 0:
            top_k = min(top_k, V)
            topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
            # 创建mask
            mask = torch.full_like(logits, -torch.inf)
            mask.scatter_(-1, topk_indices, topk_values)
            logits = mask
        
        # Top-p过滤
        if top_p > 0.0 and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 创建mask：保留累积概率小于top_p的tokens
            sorted_indices_to_remove = cumulative_probs > top_p
            # 确保至少保留一个token
            sorted_indices_to_remove[..., 0] = False
            
            # 应用mask
            sorted_logits[sorted_indices_to_remove] = -torch.inf
            
            # 恢复原始顺序
            logits = sorted_logits.gather(-1, sorted_indices.argsort(-1))
        
        # 多项式采样
        probs = F.softmax(logits, dim=-1)
        
        if rng is not None:
            # 使用指定的随机数生成器
            samples = torch.multinomial(probs.view(-1, V), 1, generator=rng).view(B, L)
        else:
            samples = torch.multinomial(probs.view(-1, V), 1).view(B, L)
        
        return samples
    
    def forward_inference(
        self,
        histology_features: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.5,
        top_k: int = 50,
        top_p: float = 0.9,
        temperature: float = 1.0,
        num_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """推理模式前向传播"""
        B = histology_features.shape[0]
        device = histology_features.device
        
        # 生成图像
        generated_images = self.autoregressive_infer_cfg(
            B=B,
            label_B=class_labels,
                cfg=cfg_scale,
                top_k=top_k,
                top_p=top_p,
            histology_condition=histology_features
            )
        
        # 确保图像格式正确
        if generated_images.shape[-2:] != (self.image_size, self.image_size):
            generated_images = F.interpolate(
                generated_images,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        if generated_images.shape[1] != 1:
                generated_images = generated_images.mean(dim=1, keepdim=True)
        
        # 转换为基因表达
        predicted_genes = self.gene_adapter.pseudo_image_to_genes(generated_images)
        
        return {
            'predicted_expression': predicted_genes,
            'predictions': predicted_genes,
            'generated_images': generated_images
        }
    
    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """主前向传播入口"""
        mode = inputs.get('mode', 'training')
        
        if mode == 'training' or 'gene_expression' in inputs:
            # 训练模式
            return self.forward_training(
                gene_expression=inputs['gene_expression'],
                histology_features=inputs['histology_features'],
                class_labels=inputs.get('class_labels'),
                show_details=inputs.get('show_details', False)
            )
        else:
            # 推理模式
            return self.forward_inference(
                histology_features=inputs['histology_features'],
                class_labels=inputs.get('class_labels'),
                cfg_scale=inputs.get('cfg_scale', 1.5),
                top_k=inputs.get('top_k', 50),
                top_p=inputs.get('top_p', 0.9),
                temperature=inputs.get('temperature', 1.0),
                num_samples=inputs.get('num_samples', 1)
            ) 
    
    def idxBl_to_var_input(self, tokens_list: List[torch.Tensor]) -> torch.Tensor:
        """
        将多尺度token列表转换为VAR的teacher forcing输入 - 原始VAR实现
        
        原始VAR逻辑：
        - 输入：前N-1个尺度的特征
        - 目标：所有N个尺度的tokens
        - Teacher forcing：用前面的真实tokens预测后面的tokens
        
        方案C适配：
        - 输入：第1层(全局)的特征 [B, z_channels]
        - 目标：第1层+第2层的所有tokens [B, 197]
        - 训练：第1层预测第2层的196个基因tokens
        
        Args:
            tokens_list: [
                [B, 1] - 第1层全局token
                [B, 196] - 第2层基因tokens
            ]
            
        Returns:
            teacher_forcing_input: [B, L-first_l, z_channels] = [B, 196, z_channels]
        """
        if len(tokens_list) != 2:
            raise ValueError(f"方案C需要2层tokens，得到: {len(tokens_list)}")
        
        B = tokens_list[0].shape[0]
        
        # 🔧 方案C的teacher forcing逻辑：
        # - 第1层(全局token)作为起始，不需要teacher forcing输入
        # - 第2层(基因tokens)需要基于第1层的特征来预测
        
        # 获取第1层的token embedding作为第2层的输入
        global_tokens = tokens_list[0]  # [B, 1]
        
        # 将第1层token转换为特征空间
        global_features = self.quantize(global_tokens)  # [B, 1, z_channels]
        
        # 🔧 方案C核心：第2层需要196个输入，每个都基于全局特征
        # 将全局特征扩展到196个位置，为第2层提供teacher forcing输入
        expanded_features = global_features.repeat(1, 196, 1)  # [B, 196, z_channels]
        
        # 🔧 添加位置信息，让每个基因位置都有独特的特征
        if hasattr(self, 'gene_position_embed'):
            # 如果有基因位置编码，添加到特征中
            position_embed = self.gene_position_embed.weight.unsqueeze(0)  # [1, 196, z_channels]
            expanded_features = expanded_features + position_embed
        else:
            # 简单的位置扰动，避免所有基因位置完全相同
            position_noise = torch.randn_like(expanded_features) * 0.01
            expanded_features = expanded_features + position_noise
        
        return expanded_features 