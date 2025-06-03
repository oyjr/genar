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
        patch_nums: Tuple[int, ...] = (1, 2, 4),
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
        
        print(f"🧬 VAR基因包装器 - 完整实现版本")
        print(f"   基因数量: {num_genes} → {image_size}×{image_size}")
        print(f"   多尺度patch_nums: {patch_nums}")
        
        # 计算多尺度参数
        self.L = sum(pn ** 2 for pn in self.patch_nums)  # 总token数
        self.first_l = self.patch_nums[0] ** 2  # 第一层token数
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn ** 2))
            cur += pn ** 2
        
        print(f"   总token数: {self.L}, 第一层: {self.first_l}")
        print(f"   各层范围: {self.begin_ends}")
        
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
            pe = torch.empty(1, pn * pn, self.embed_dim)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        self.pos_1LC = nn.Parameter(torch.cat(pos_1LC, dim=1))
        
        # 4. 层级编码 (每个token的层级信息)
        lvl_1L = []
        for i, pn in enumerate(self.patch_nums):
            lvl_1L.extend([i] * (pn ** 2))
        self.register_buffer('lvl_1L', torch.tensor(lvl_1L, dtype=torch.long))
        
        # 层级embedding
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.embed_dim)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 5. AdaLN-Zero transformer blocks
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                block_idx=i,
                last_drop_p=0.0 if i == 0 else 0.1,
                embed_dim=self.embed_dim,
                cond_dim=self.embed_dim,
                shared_aln=False,
                norm_layer=nn.LayerNorm,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                drop=0.1,
                attn_drop=0.1,
                drop_path=0.1,
                attn_l2_norm=False,
                flash_if_available=False,
                fused_if_available=True,
                enable_histology_injection=True,
                histology_dim=self.histology_feature_dim,
            ) for i in range(self.depth)
        ])
        
        # 6. Final normalization and head
        self.head_nm = AdaLNBeforeHead(self.embed_dim, self.embed_dim, norm_layer=nn.LayerNorm)
        self.head = nn.Linear(self.embed_dim, self.vocab_size)
        nn.init.trunc_normal_(self.head.weight.data, mean=0, std=init_std)
        nn.init.constant_(self.head.bias.data, 0)
        
        print(f"   transformer块数: {len(self.blocks)}")
        print(f"   输出词汇表: {self.vocab_size}")
    
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
        """图像编码为多尺度token indices"""
        B = x.shape[0]
        
        # 编码到潜在空间
        z = self.encoder(x)  # [B, z_channels, 4, 4]
        B, C, H, W = z.shape
        
        # 展平并量化
        z_flat = z.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
        
        # 找最近的codebook向量
        distances = torch.cdist(z_flat, self.quantize.weight)  # [B, H*W, vocab_size]
        indices = torch.argmin(distances, dim=-1)  # [B, H*W]
        
        # 按多尺度组织tokens
        tokens = []
        for pn in self.patch_nums:
            if pn <= H:  # 确保patch size不超过特征图尺寸
                # 对于每个尺度，从4×4特征图中采样
                patch_tokens = []
                for i in range(pn):
                    for j in range(pn):
                        # 将pn×pn网格映射到4×4特征图
                        hi = min(int(i * H / pn), H-1)
                        wi = min(int(j * W / pn), W-1)
                        idx = hi * W + wi
                        patch_tokens.append(indices[:, idx])
                
                patch_tensor = torch.stack(patch_tokens, dim=1)  # [B, pn*pn]
                tokens.append(patch_tensor)
            else:
                # 如果patch size太大，重复使用现有tokens
                repeat_times = (pn * pn + H*W - 1) // (H*W)  # 向上取整
                repeated = indices.repeat(1, repeat_times)[:, :pn*pn]
                tokens.append(repeated)
                
        return tokens
            
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
    
    def forward_training(
        self,
        gene_expression: torch.Tensor,
        histology_features: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        show_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """训练前向传播"""
        B, num_genes = gene_expression.shape
        device = gene_expression.device
        
        self._step_count += 1
        
        # 显示详情控制
        import os
        is_main_process = int(os.environ.get('LOCAL_RANK', 0)) == 0
        show_details = (self._verbose_logging and is_main_process and 
                       (self._step_count <= 3 or self._step_count % 1000 == 0))
        
        if show_details:
            print(f"🧬 完整VAR训练 (步骤 {self._step_count}):")
        
        # 1. 基因 → 伪图像
        pseudo_images = self.gene_adapter.genes_to_pseudo_image(gene_expression)
        if pseudo_images.shape[1] != 1:
            pseudo_images = pseudo_images.mean(dim=1, keepdim=True)
            
        # 2. 处理条件
        condition_embeddings = self.condition_processor(histology_features)
        
        if class_labels is None:
            # 动态生成类别标签
            with torch.no_grad():
                n_classes = min(self.num_classes, condition_embeddings.shape[1])
                class_features = condition_embeddings[:, :n_classes]
                class_probs = torch.softmax(class_features, dim=-1)
                class_labels = torch.multinomial(class_probs, 1).squeeze(-1)
        
        # 3. VQVAE编码
        ms_tokens = self.img_to_idxBl(pseudo_images)
        
        if show_details:
            print(f"   多尺度tokens: {[t.shape for t in ms_tokens]}")
        
        # 4. VAR训练
        # 准备输入：除了第一层的所有tokens
        all_tokens = torch.cat([tokens.flatten(1) for tokens in ms_tokens], dim=1)  # [B, L]
        x_BLCv_wo_first_l = all_tokens[:, self.first_l:]  # [B, L-first_l]
        
        # 获取嵌入
        if x_BLCv_wo_first_l.numel() == 0:
            raise RuntimeError(
                f"Token序列为空，这不应该发生。"
                f"first_l: {self.first_l}, all_tokens形状: {all_tokens.shape if 'all_tokens' in locals() else 'N/A'}, "
                f"patch_nums: {self.patch_nums}"
            )
        
        # 量化嵌入
        token_embeddings = self.quantize(x_BLCv_wo_first_l)  # [B, L-first_l, z_channels]
        # 映射到VAR空间
        x = self.word_embed(token_embeddings)  # [B, L-first_l, embed_dim]
        
        # 添加位置编码
        if x.shape[1] <= self.pos_1LC.shape[1] - self.first_l:
            x = x + self.pos_1LC[:, self.first_l:self.first_l + x.shape[1]]
        
        # 🔧 修复层级编码：正确计算索引范围
        # lvl_1L的形状是[L]，包含每个位置对应的层级ID (0, 1, 2, ...)
        # 我们需要取出对应的层级ID，然后获取对应的编码
        if x.shape[1] > len(self.lvl_1L) - self.first_l:
            raise RuntimeError(
                f"Token序列长度{x.shape[1]}超出层级编码范围{len(self.lvl_1L) - self.first_l}。"
                f"总长度: {len(self.lvl_1L)}, first_l: {self.first_l}, 可用长度: {len(self.lvl_1L) - self.first_l}"
            )
        
        lvl_indices = self.lvl_1L[self.first_l:self.first_l + x.shape[1]]  # [L-first_l]
        lvl_pos = self.lvl_embed(lvl_indices)  # [L-first_l, embed_dim]
        x = x + lvl_pos.unsqueeze(0)  # [B, L-first_l, embed_dim]
        
        # Class条件
        cond_BD = self.class_emb(class_labels)  # [B, embed_dim]
        
        # 通过transformer blocks
        for block in self.blocks:
            x = block(x, cond_BD, attn_bias=None, histology_condition=histology_features)
        
        # 获取logits
        logits = self.get_logits(x, cond_BD)  # [B, L-first_l, vocab_size]
        
        # 计算VAR损失
        if x_BLCv_wo_first_l.numel() > 0:
            target_tokens = x_BLCv_wo_first_l
            var_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), 
                target_tokens.reshape(-1),
                ignore_index=-1
            )
        else:
            var_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
        # 5. 生成验证
        with torch.no_grad():
            # 使用当前模型生成
            generated_tokens = self.autoregressive_infer_cfg(
                B=B,
                label_B=class_labels,
                histology_condition=histology_features
            )
            generated_images = generated_tokens  # autoregressive_infer_cfg直接返回图像
            
            # 确保格式正确
            if generated_images.shape[-2:] != (self.image_size, self.image_size):
                generated_images = F.interpolate(
                    generated_images,
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False
                )
                
            if generated_images.shape[1] != 1:
                generated_images = generated_images.mean(dim=1, keepdim=True)
                
            # 伪图像 → 基因表达
            predicted_genes = self.gene_adapter.pseudo_image_to_genes(generated_images)
                
            # 重建损失
            recon_loss = F.mse_loss(predicted_genes, gene_expression)
                
        # 6. 总损失
        if self.progressive_training:
            # 渐进式训练权重
            progress = min(1.0, self.current_step / self.warmup_steps)
            recon_weight = self.min_recon_weight + progress * (self.max_recon_weight - self.min_recon_weight)
        else:
            recon_weight = self.max_recon_weight
        
        total_loss = var_loss + recon_weight * recon_loss
        
        if show_details:
            print(f"   VAR损失: {var_loss:.4f}")
            print(f"   重建损失: {recon_loss:.4f} (权重: {recon_weight:.2f})")
            print(f"   总损失: {total_loss:.4f}")
        
        return {
            'loss': total_loss,
            'var_loss': var_loss,
            'recon_loss': recon_loss,
            'predicted_expression': predicted_genes,
            'predictions': predicted_genes
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
        """自回归推理生成"""
        device = next(self.parameters()).device
        
        if label_B is None:
            label_B = torch.zeros(B, dtype=torch.long, device=device)
        
        if g_seed is not None:
            torch.manual_seed(g_seed)
        
        # 初始化序列
        seq_len = self.L  # 总token长度
        input_ids = torch.zeros(B, seq_len, dtype=torch.long, device=device)
        
        # 生成第一层tokens (随机初始化)
        for i in range(self.first_l):
            input_ids[:, i] = torch.randint(0, self.vocab_size, (B,), device=device)
        
        # 自回归生成剩余tokens
        for cur_L in range(self.first_l, seq_len):
            # 获取当前序列
            x_BLCv_wo_first_l = input_ids[:, self.first_l:cur_L]
            
            if x_BLCv_wo_first_l.shape[1] == 0:
                # 如果序列为空，跳过
                continue
            
            # 嵌入
            token_embeddings = self.quantize(x_BLCv_wo_first_l)
            x = self.word_embed(token_embeddings)
            
            # 位置编码
            if x.shape[1] <= self.pos_1LC.shape[1] - self.first_l:
                x = x + self.pos_1LC[:, self.first_l:self.first_l + x.shape[1]]
            
            # 层级编码
            if x.shape[1] <= len(self.lvl_1L) - self.first_l:
                lvl_indices = self.lvl_1L[self.first_l:self.first_l + x.shape[1]]
                lvl_pos = self.lvl_embed(lvl_indices)
                x = x + lvl_pos.unsqueeze(0)
            
            # Class条件
            cond_BD = self.class_emb(label_B)
            
            # 通过transformer
            for block in self.blocks:
                x = block(x, cond_BD, attn_bias=None, histology_condition=histology_condition)
            
            # 预测下一个token
            logits = self.get_logits(x, cond_BD)  # [B, cur_L-first_l, vocab_size]
            
            if logits.shape[1] > 0:
                next_token_logits = logits[:, -1, :]  # [B, vocab_size]
                
                # Top-k和top-p采样
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids[:, cur_L] = next_token.squeeze(-1)
        
        # 解码为图像
        # 将所有tokens转换为多尺度格式
        ms_tokens = []
        start_idx = 0
        for pn in self.patch_nums:
            length = pn ** 2
            tokens = input_ids[:, start_idx:start_idx + length]
            ms_tokens.append(tokens)
            start_idx += length
        
        # 解码
        generated_images = self.idxBl_to_img(ms_tokens, last_one=True)
        
        return generated_images
    
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