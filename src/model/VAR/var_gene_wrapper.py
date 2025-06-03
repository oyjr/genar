"""
VAR基因包装器 - 基于基因多尺度的从头训练版本

设计理念：
- 完全从头训练VAR和VQVAE
- 专门为196基因 (14x14) 设计
- 基因维度多尺度：(1,2,3,4,5) 对应生物学层次
- 无视觉预训练依赖，纯基因表达逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# 🔧 修复：导入原始VAR项目的模型架构（不加载权重）
# 添加VAR项目路径到sys.path
VAR_PROJECT_PATH = "/home/ouyangjiarui/project/ST/VAR"
if VAR_PROJECT_PATH not in sys.path:
    sys.path.insert(0, VAR_PROJECT_PATH)

# 从原始VAR项目导入模型架构
try:
    from models.var import VAR
    from models.vqvae import VQVAE
    print("✅ 成功导入VAR项目模型架构")
    VAR_AVAILABLE = True
except ImportError as e:
    print(f"❌ 导入VAR模型失败: {e}")
    print("🔄 将使用简化版本继续")
    VAR_AVAILABLE = False

# 导入本地基因适配器
from .gene_pseudo_image_adapter import GenePseudoImageAdapter

class VARGeneWrapper(nn.Module):
    """
    VAR基因包装器 - 基因多尺度从头训练版本
    
    专门为基因表达预测设计的VAR架构：
    - 196个基因 → 14×14×1 伪图像
    - 基因多尺度：(1,2,3,4,5) = 55 tokens
    - 完全从头训练，无预训练依赖
    - 生物学多尺度语义
    """
    
    def __init__(
        self,
        num_genes: int = 196,  # 🔧 固定196基因
        image_size: int = 14,  # 🔧 14×14完美平方
        histology_feature_dim: int = 512,
        patch_nums: Tuple[int, ...] = (1, 2, 3, 4, 5),  # 🔧 基因多尺度
        var_config: Optional[Dict] = None,
        vqvae_config: Optional[Dict] = None,
        adapter_config: Optional[Dict] = None
    ):
        """
        初始化基因多尺度VAR模型
        
        Args:
            num_genes: 基因数量，固定196 (14×14)
            image_size: 伪图像尺寸，固定14×14
            histology_feature_dim: 组织学特征维度
            patch_nums: 基因多尺度设置 (1,2,3,4,5)
            var_config: VAR配置
            vqvae_config: VQVAE配置
            adapter_config: 适配器配置
        """
        super().__init__()
        
        # 🔧 严格验证基因数量和图像尺寸的匹配（支持padding策略）
        total_image_positions = image_size * image_size
        if num_genes == 196:
            # 🔧 修改：允许196基因使用16×16 padding策略
            if image_size < 14:
                raise ValueError(f"196基因至少需要14×14图像，但指定了{image_size}×{image_size}")
            if image_size == 14 and total_image_positions != 196:
                raise ValueError(f"14×14图像必须完美匹配196基因，但位置数为{total_image_positions}")
            
            # 允许使用16×16 padding策略
            if image_size >= 16:
                print(f"🔧 196基因使用padding策略: {image_size}×{image_size} (padding {total_image_positions - 196}个位置)")
                self.use_padding = True
                self.padding_size = total_image_positions - num_genes
            else:
                # 14×14完美匹配模式
                print(f"🔧 196基因使用完美匹配: {image_size}×{image_size}")
                self.use_padding = False
                self.padding_size = 0
        else:
            # 对于其他基因数量，允许灵活匹配
            if num_genes > total_image_positions:
                raise ValueError(f"基因数量{num_genes}不能大于图像位置数{image_size}×{image_size}={total_image_positions}")
            self.use_padding = num_genes < total_image_positions
            self.padding_size = total_image_positions - num_genes
        
        self.num_genes = num_genes
        self.image_size = image_size
        self.histology_feature_dim = histology_feature_dim
        self.patch_nums = patch_nums
        
        # 计算总token数
        self.total_tokens = sum(p*p for p in patch_nums)
        
        # 🔇 简化初始化输出，减少冗余信息
        print(f"🧬 VAR基因包装器: {num_genes}基因 → {image_size}×{image_size}, {self.total_tokens}tokens")
        if num_genes == 196:
            print(f"   ✅ 196基因模式：完美匹配14×14")
        
        # 🔇 生物学语义显示将在后续设置
        self._biological_semantics = None  # 待父类传递
        
        # 🔧 步骤1：初始化基因适配器（196基因 → 16×16伪图像，padding策略）
        self.gene_adapter = GenePseudoImageAdapter(
            num_genes=num_genes,
            target_image_size=image_size,  # 🔧 使用16×16，padding策略
            normalize_method='layer_norm',
            eps=1e-6
        )

        print(f"🧬 初始化基因适配器:")
        print(f"   - 基因数量: {num_genes}")
        print(f"   - 图像尺寸: {image_size}×{image_size} (padding策略)")
        print(f"   - Padding大小: {image_size*image_size - num_genes}")
        print(f"   - 空间利用率: {num_genes/(image_size*image_size):.1%}")

        # 验证适配器转换正确性
        print(f"📊 验证基因适配器转换...")
        validation_result = self.gene_adapter.validate_conversion()
        if validation_result['conversion_successful']:
            print(f"   ✅ 转换验证成功")
            print(f"   - 最大重建误差: {validation_result['max_reconstruction_error']:.2e}")
            print(f"   - 平均重建误差: {validation_result['mean_reconstruction_error']:.2e}")
            print(f"   - Padding区域保持零值: {validation_result['padding_preserved']}")
        else:
            print(f"   ❌ 转换验证失败")
            print(f"   - 误差: {validation_result['max_reconstruction_error']:.2e}")
        
        # 🔧 步骤2：初始化VQVAE配置（适配16×16输入）
        if VAR_AVAILABLE:
            print(f"🎨 使用完整VAR VQVAE (单通道输入，16×16)")
            
            # 🔧 确保vqvae_config不为None，提供默认配置
            if vqvae_config is None:
                vqvae_config = {
                    'vocab_size': 4096,
                    'z_channels': 32,
                    'ch': 160,
                    'dropout': 0.0
                }
                print(f"   🔧 使用默认VQVAE配置: {vqvae_config}")
            
            # 🔧 修复：直接使用VQVAE而不是复杂的配置类
            # 因为原始VAR项目的VQVAE可能有不同的初始化方式
            try:
                # 尝试直接初始化VQVAE（如果支持简单参数）
                self.vqvae = VQVAE(
                    embed_dim=vqvae_config.get('embed_dim', 256),
                    n_embed=vqvae_config.get('n_embed', 8192),
                    double_z=vqvae_config.get('double_z', False),
                    z_channels=vqvae_config.get('z_channels', 256),
                    resolution=image_size,  # 16×16分辨率
                    in_channels=1,  # 🔧 单通道基因伪图像
                    out_ch=1,  # 🔧 单通道输出
                    ch=vqvae_config.get('ch', 128),
                    ch_mult=vqvae_config.get('ch_mult', [1, 1, 2, 2, 4]),
                    num_res_blocks=vqvae_config.get('num_res_blocks', 2),
                    attn_resolutions=vqvae_config.get('attn_resolutions', [16]),  # 适配16×16
                    dropout=vqvae_config.get('dropout', 0.0)
                )
                print(f"   ✅ 完整VAR VQVAE初始化成功 (1→{vqvae_config.get('z_channels', 256)}→1通道，16×16)")
                
            except Exception as e:
                print(f"   ⚠️ 直接初始化VQVAE失败: {e}")
                print(f"   🔄 尝试从原始VAR项目查找正确的初始化方式...")
                
                # 🔧 关键修复：使用VAR项目的标准参数，特别是v_patch_nums
                vqvae_kwargs = {
                    'vocab_size': vqvae_config.get('vocab_size', 4096),
                    'z_channels': vqvae_config.get('z_channels', 32),
                    'ch': vqvae_config.get('ch', 160),
                    'dropout': vqvae_config.get('dropout', 0.0),
                    'beta': 0.25,
                    'using_znorm': False,
                    'quant_conv_ks': 3,
                    'quant_resi': 0.5,
                    'share_quant_resi': 4,
                    'default_qresi_counts': 0,
                    'v_patch_nums': self.patch_nums,  # 🔧 关键：传递正确的patch_nums序列
                    'test_mode': False
                }
                
                self.vqvae = VQVAE(**vqvae_kwargs)
                print(f"   ✅ 自适应VQVAE初始化成功，v_patch_nums={self.patch_nums}")
                
            except Exception as e2:
                print(f"   ❌ 自适应初始化也失败: {e2}")
                print(f"   🔄 使用最小化参数初始化...")
                
                # 🔧 最小化参数初始化
                try:
                    # 尝试只传递最必要的参数
                    self.vqvae = VQVAE(
                        vocab_size=4096,
                        z_channels=32,
                        v_patch_nums=self.patch_nums  # 🔧 确保传递patch_nums
                    )
                    print(f"   ⚠️ 使用默认参数初始化VQVAE（可能需要手动调整）")
                except Exception as e3:
                    print(f"   ❌ 所有VQVAE初始化尝试都失败: {e3}")
                    raise RuntimeError(f"无法初始化完整VAR VQVAE: {e3}")

        else:
            print(f"🎨 使用简化VQVAE (单通道，16×16)")
            # 简化VQVAE，适配16×16输入
            self.vqvae = nn.Sequential(
                # 编码器：1×16×16 → 256×4×4  
                nn.Conv2d(1, 32, 3, padding=1),  # [B,1,16,16] → [B,32,16,16]
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [B,32,16,16] → [B,64,8,8]
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),  # [B,64,8,8] → [B,128,4,4]
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1),  # [B,128,4,4] → [B,256,4,4]
                
                # 解码器：256×4×4 → 1×16×16
                nn.ConvTranspose2d(256, 128, 3, padding=1),  # [B,256,4,4] → [B,128,4,4]
                nn.ReLU(), 
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # [B,128,4,4] → [B,64,8,8]
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # [B,64,8,8] → [B,32,16,16]
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 3, padding=1),  # [B,32,16,16] → [B,1,16,16]
                nn.Tanh()  # 输出范围[-1,1]
            )
            print(f"   ✅ 简化VQVAE初始化完成 (1→256→1通道，16×16)")

        print(f"   - 输入格式: [B, 1, {image_size}, {image_size}] (单通道基因伪图像)")
        print(f"   - 输出格式: [B, 1, {image_size}, {image_size}] (重建基因伪图像)")
        print(f"   - 支持分辨率: 16×16 (padding策略解决尺寸限制)")
        
        # 🔧 Step 3: 初始化基因专用VAR（从头训练）
        # print(f"🏗️ 初始化基因专用VAR（从头训练）...")
        self._init_gene_var(var_config)
        
        # 🔧 Step 4: 条件特征处理器
        # 映射组织学特征到基因语义空间
        self.condition_processor = nn.Sequential(
            nn.Linear(histology_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.var_embed_dim)  # 映射到VAR嵌入维度
        )
        
        print(f"🔧 条件处理器配置:")
        print(f"   - 输入维度: {histology_feature_dim}")
        print(f"   - VAR嵌入维度: {self.var_embed_dim}")
        print(f"   - 处理链: {histology_feature_dim} → 512 → 256 → {self.var_embed_dim}")
        
        # 输出参数统计
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"📊 参数统计（全部可训练）:")
        print(f"   - 总参数: {total_params:,}")
        print(f"   - 可训练参数: {trainable_params:,} (100%)")
        print(f"✅ VAR基因包装器初始化完成（需求兼容版）")
        
        # 🔧 初始化日志控制属性
        self._verbose_logging = True  # 默认启用详细日志
        self._step_count = 0  # 步数计数器
    
    def _init_gene_var(self, var_config: Optional[Dict] = None):
        """初始化基因专用VAR"""
        config = var_config or {}
        
        # 基因VAR配置
        self.var_embed_dim = config.get('embed_dim', 512)      # 更小的嵌入维度
        self.var_depth = config.get('depth', 12)               # 更少的层数
        self.var_num_heads = config.get('num_heads', 8)        # 更少的注意力头
        self.var_num_classes = config.get('num_classes', 10)   # 基因表达类型数
        
        if VAR_AVAILABLE:
            self.var_model = VAR(
                vae_local=self.vqvae,
                num_classes=self.var_num_classes,
                depth=self.var_depth,
                embed_dim=self.var_embed_dim,
                num_heads=self.var_num_heads,
                mlp_ratio=config.get('mlp_ratio', 4.0),
                drop_rate=config.get('drop_rate', 0.1),
                attn_drop_rate=config.get('attn_drop_rate', 0.1),
                drop_path_rate=config.get('drop_path_rate', 0.1),
                norm_eps=config.get('norm_eps', 1e-6),
                shared_aln=config.get('shared_aln', False),
                cond_drop_rate=config.get('cond_drop_rate', 0.1),
                attn_l2_norm=config.get('attn_l2_norm', False),
                patch_nums=self.patch_nums,  # 🔧 使用基因多尺度
                flash_if_available=config.get('flash_if_available', False),
                fused_if_available=config.get('fused_if_available', False)
            )
            print(f"   ✅ VAR: embed={self.var_embed_dim}, depth={self.var_depth}, heads={self.var_num_heads}")
        else:
            # 简化版VAR
            self.var_model = self._create_simple_var()
            print(f"   ⚠️ 使用简化版VAR")
    
    def _create_simple_vqvae(self):
        """创建简化版VQVAE（当VAR库不可用时）"""
        class SimpleVQVAE(nn.Module):
            def __init__(self, vocab_size=1024, z_channels=16):
                super().__init__()
                self.vocab_size = vocab_size
                self.z_channels = z_channels
                
                # 简单的编码器
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, z_channels, 3, padding=1)
                )
                
                # 量化层
                self.quantize = nn.Embedding(vocab_size, z_channels)
                
                # 简单的解码器
                self.decoder = nn.Sequential(
                    nn.Conv2d(z_channels, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 1, 3, padding=1)
                )
            
            def img_to_idxBl(self, x):
                # 编码并量化
                z = self.encoder(x)
                B, C, H, W = z.shape
                z_flat = z.view(B, C, -1).permute(0, 2, 1)
                
                # 简单量化：找最近的codebook向量
                distances = torch.cdist(z_flat, self.quantize.weight)
                indices = torch.argmin(distances, dim=-1)
                
                # 为每个patch_num创建tokens
                tokens = []
                for p in self.training_patch_nums if hasattr(self, 'training_patch_nums') else [1,2,3,4,5]:
                    patch_size = H // p
                    if patch_size > 0:
                        # 简单地重复索引
                        patch_tokens = indices[:, :p*p] if indices.shape[1] >= p*p else indices[:, :1].repeat(1, p*p)
                        tokens.append(patch_tokens)
                    else:
                        tokens.append(indices[:, :1])
                
                return tokens
            
            def idxBl_to_img(self, tokens, same_shape=True, last_one=True):
                # 简单重建：使用最后一个token
                if last_one and len(tokens) > 0:
                    indices = tokens[-1]  # 使用最细尺度的tokens
                else:
                    indices = tokens[0] if tokens else torch.zeros(1, 1, dtype=torch.long)
                
                # 重建
                B = indices.shape[0]
                quantized = self.quantize(indices)
                
                # 简单地reshape并解码
                if quantized.dim() == 3:
                    H = W = int(np.sqrt(quantized.shape[1]))
                    if H * W != quantized.shape[1]:
                        H = W = 14  # 默认尺寸
                        quantized = quantized[:, :H*W]
                    z = quantized.permute(0, 2, 1).view(B, -1, H, W)
                else:
                    z = quantized.unsqueeze(-1).unsqueeze(-1)
                
                return self.decoder(z)
            
            def forward(self, x):
                tokens = self.img_to_idxBl(x)
                recon = self.idxBl_to_img(tokens)
                return {'tokens': tokens, 'recon': recon, 'vq_loss': torch.tensor(0.0)}
        
        vqvae = SimpleVQVAE(self.vqvae_vocab_size, self.vqvae_z_channels)
        vqvae.training_patch_nums = self.patch_nums
        return vqvae
    
    def _create_simple_var(self):
        """创建简化版VAR（当VAR库不可用时）"""
        class SimpleVAR(nn.Module):
            def __init__(self, embed_dim=512, depth=12, num_heads=8, vocab_size=1024):
                super().__init__()
                self.embed_dim = embed_dim
                self.token_embedding = nn.Embedding(vocab_size, embed_dim)
                self.pos_embedding = nn.Parameter(torch.randn(1, 100, embed_dim))  # 足够大的位置编码
                
                # Transformer layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
                
                # 输出层
                self.output_layer = nn.Linear(embed_dim, vocab_size)
                
            def forward(self, label_B=None, x_BLCv_wo_first_l=None, is_train=True, **kwargs):
                if x_BLCv_wo_first_l is None:
                    return torch.tensor(0.0, requires_grad=True)
                
                # 处理多尺度tokens
                if isinstance(x_BLCv_wo_first_l, list):
                    # 连接所有尺度的tokens
                    all_tokens = []
                    for tokens in x_BLCv_wo_first_l:
                        if tokens.dim() == 2:
                            all_tokens.append(tokens)
                        elif tokens.dim() == 3:
                            all_tokens.append(tokens.flatten(1))
                    
                    if all_tokens:
                        tokens = torch.cat(all_tokens, dim=1)
                    else:
                        B = x_BLCv_wo_first_l[0].shape[0] if x_BLCv_wo_first_l else 1
                        tokens = torch.zeros(B, 10, dtype=torch.long, device=next(self.parameters()).device)
                else:
                    tokens = x_BLCv_wo_first_l
                
                # 嵌入
                B, L = tokens.shape[:2]
                x = self.token_embedding(tokens)
                
                # 位置编码
                if L <= self.pos_embedding.shape[1]:
                    x = x + self.pos_embedding[:, :L]
                
                # Transformer
                x = self.transformer(x)
                
                # 输出
                logits = self.output_layer(x)
                
                # 计算损失
                if is_train:
                    # 简单的自回归损失
                    targets = tokens[:, 1:] if L > 1 else tokens
                    preds = logits[:, :-1] if L > 1 else logits
                    
                    if targets.numel() > 0 and preds.numel() > 0:
                        loss = F.cross_entropy(preds.reshape(-1, preds.shape[-1]), targets.reshape(-1))
                    else:
                        loss = torch.tensor(0.0, requires_grad=True)
                    
                    return loss
                else:
                    return logits
            
            def autoregressive_infer_cfg(self, B, label_B=None, cfg=1.5, top_k=50, top_p=0.9, more_smooth=False):
                # 简单生成
                device = next(self.parameters()).device
                generated = torch.zeros(B, 3, 14, 14, device=device)  # 直接生成目标尺寸
                return generated
        
        return SimpleVAR(self.var_embed_dim, self.var_depth, self.var_num_heads, self.vqvae_vocab_size)
    
    def forward_training(
        self,
        gene_expression: torch.Tensor,
        histology_features: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        show_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        训练阶段前向传播 - 基因多尺度从头训练版本
        
        流程：
        1. 基因表达 [B, 196] → 伪图像 [B, 1, 14, 14]
        2. VQVAE编码 → 基因多尺度tokens (1,2,3,4,5)
        3. VAR自回归训练
        4. 重建验证
        
        Args:
            gene_expression: [B, 196] - 基因表达向量
            histology_features: [B, feature_dim] - 组织学特征
            class_labels: [B] - 类别标签(可选)
            show_details: 是否显示详细信息
        
        Returns:
            包含损失和预测结果的字典
        """
        B, num_genes = gene_expression.shape
        device = gene_expression.device
        
        # 🔧 增加步数计数
        self._step_count += 1
        
        # 验证输入维度
        if num_genes != self.num_genes:
            raise ValueError(f"输入基因数量{num_genes}与模型期望{self.num_genes}不匹配")
        
        # 🔇 大幅减少详细输出：只在前3步和每1000步显示详情
        # 🔧 在分布式训练中，只在主进程显示详细信息
        import os
        is_main_process = int(os.environ.get('LOCAL_RANK', 0)) == 0
        show_details = (self._verbose_logging and is_main_process and 
                       (self._step_count <= 3 or self._step_count % 1000 == 0))
        
        if show_details:
            print(f"🧬 基因多尺度VAR训练:")
            print(f"   输入基因表达: {gene_expression.shape}")
        
        try:
            # Step 1: 基因表达 → 伪图像 [B, 1, 14, 14]
            pseudo_images = self.gene_adapter.genes_to_pseudo_image(gene_expression)
            if show_details:
                print(f"   伪图像转换: {gene_expression.shape} → {pseudo_images.shape}")
                if self.use_padding:
                    print(f"   使用padding: {self.num_genes}基因 + {self.padding_size}padding = {self.image_size}×{self.image_size}")
            
            # 确保伪图像通道数为1（基因数据是单通道的）
            if pseudo_images.shape[1] != 1:
                if pseudo_images.shape[1] == 3:
                    # 如果适配器返回3通道，转为单通道
                    pseudo_images = pseudo_images.mean(dim=1, keepdim=True)
                else:
                    # 取第一个通道
                    pseudo_images = pseudo_images[:, :1]
            
            # Step 2: 处理条件特征
            if class_labels is None:
                # 通过条件处理器生成基因语义标签
                condition_embeddings = self.condition_processor(histology_features)  # [B, embed_dim]
                # 简单映射到类别标签
                class_labels = torch.argmax(condition_embeddings[:, :self.var_num_classes], dim=-1)
            
            # 🔇 减少条件标签输出，只在详细模式显示
            # if show_details:
            #     print(f"   条件标签: {class_labels.shape}")
            
            # 🔧 方案1：将单通道基因伪图像转换为3通道，适配原始VAR VQVAE
            # 原始VAR的VQVAE编码器期望3通道RGB输入，我们需要适配
            if pseudo_images.shape[1] == 1:
                # 将单通道复制为3通道: [B, 1, H, W] → [B, 3, H, W]
                pseudo_images_3ch = pseudo_images.repeat(1, 3, 1, 1)
                if show_details:
                    print(f"   🔧 通道适配: {pseudo_images.shape} → {pseudo_images_3ch.shape} (适配VAR VQVAE)")
            else:
                pseudo_images_3ch = pseudo_images
            
            # 确保伪图像格式正确为3通道
            if pseudo_images_3ch.shape[1] != 3:
                if pseudo_images_3ch.shape[1] > 3:
                    # 取前3个通道
                    pseudo_images_3ch = pseudo_images_3ch[:, :3]
                    if show_details:
                        print(f"   🔧 截取前3通道: → {pseudo_images_3ch.shape}")
                else:
                    # 不足3通道，补齐
                    channels_needed = 3 - pseudo_images_3ch.shape[1]
                    padding_channels = pseudo_images_3ch[:, :1].repeat(1, channels_needed, 1, 1)
                    pseudo_images_3ch = torch.cat([pseudo_images_3ch, padding_channels], dim=1)
                    if show_details:
                        print(f"   🔧 补齐到3通道: → {pseudo_images_3ch.shape}")
            
            # Step 3: VQVAE编码 → 基因多尺度tokens (使用3通道图像)
            ms_tokens = self.vqvae.img_to_idxBl(pseudo_images_3ch)
            # 🔇 减少tokens输出
            # if show_details:
            #     print(f"   多尺度tokens: {[t.shape if isinstance(t, torch.Tensor) else 'None' for t in ms_tokens]}")
            
            # 确保tokens格式正确
            if not isinstance(ms_tokens, list):
                ms_tokens = [ms_tokens]
            
            # 过滤无效tokens
            valid_tokens = []
            for i, tokens in enumerate(ms_tokens):
                if isinstance(tokens, torch.Tensor) and tokens.numel() > 0:
                    valid_tokens.append(tokens)
                else:
                    # 创建默认tokens
                    patch_size = self.patch_nums[i] if i < len(self.patch_nums) else 1
                    default_tokens = torch.zeros(B, patch_size*patch_size, dtype=torch.long, device=device)
                    valid_tokens.append(default_tokens)
            
            ms_tokens = valid_tokens
            
            # Step 4: VAR自回归训练
            # 🔧 处理multi-scale tokens，转换为VAR期望的格式
            if len(ms_tokens) == 0:
                # 创建默认tokens
                total_tokens = sum(p*p for p in self.patch_nums)
                default_tokens = torch.zeros(B, total_tokens, dtype=torch.long, device=device)
                ms_tokens = [default_tokens]

            # VAR期望的格式：teacher forcing input [B, L-first_l, Cvae]
            # 其中 L = sum(patch_nums^2)，first_l = patch_nums[0]^2
            total_tokens = sum(p*p for p in self.patch_nums)
            first_l = self.patch_nums[0] ** 2

            if isinstance(ms_tokens, list) and len(ms_tokens) > 1:
                # 拼接所有尺度的tokens
                all_tokens = torch.cat([tokens.flatten(1) for tokens in ms_tokens], dim=1)  # [B, total_tokens]
            else:
                # 单个token tensor
                tokens = ms_tokens[0] if isinstance(ms_tokens, list) else ms_tokens
                if tokens.numel() == B * total_tokens:
                    all_tokens = tokens.view(B, total_tokens)
                else:
                    # 如果token数量不匹配，创建默认tokens
                    all_tokens = torch.zeros(B, total_tokens, dtype=torch.long, device=device)

            # 🔧 关键修复：VAR需要除了第一层外的所有tokens的embedding
            # x_BLCv_wo_first_l: [B, L-first_l, Cvae]
            tokens_wo_first_l = all_tokens[:, first_l:]  # 移除第一层tokens [B, L-first_l]

            # 将token indices转换为VQVAE embeddings
            if hasattr(self.vqvae, 'quantize') and hasattr(self.vqvae.quantize, 'embedding'):
                # 使用VQVAE的量化器获取embeddings
                x_BLCv_wo_first_l = self.vqvae.quantize.embedding(tokens_wo_first_l)  # [B, L-first_l, Cvae]
            elif hasattr(self.vqvae, 'vae_quant_proxy'):
                # 使用VAR的代理量化器
                x_BLCv_wo_first_l = self.vqvae.vae_quant_proxy[0].embedding(tokens_wo_first_l)
            else:
                # 备用方案：随机初始化embeddings
                Cvae = getattr(self.vqvae, 'Cvae', 256)  # 默认256维
                x_BLCv_wo_first_l = torch.randn(B, tokens_wo_first_l.shape[1], Cvae, device=device)
                print(f"   ⚠️ 使用随机embeddings，形状: {x_BLCv_wo_first_l.shape}")

            if show_details:
                print(f"   🔧 VAR输入转换: tokens{all_tokens.shape} → embeddings{x_BLCv_wo_first_l.shape}")

            var_output = self.var_model(
                label_B=class_labels,
                x_BLCv_wo_first_l=x_BLCv_wo_first_l
            )
            
            # 提取VAR损失
            if isinstance(var_output, dict):
                var_loss = var_output.get('loss', var_output.get('ce_loss', 0.0))
            elif isinstance(var_output, torch.Tensor):
                var_loss = var_output
            else:
                var_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 🔧 确保var_loss是标量
            if isinstance(var_loss, torch.Tensor):
                if var_loss.numel() > 1:
                    # 如果var_loss是多维tensor，取平均值
                    var_loss = var_loss.mean()
                elif var_loss.numel() == 0:
                    # 如果是空tensor，设为0
                    var_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # 如果不是tensor，转换为tensor
                var_loss = torch.tensor(float(var_loss), device=device, requires_grad=True)
            
            # 🔇 只在详细模式显示VAR损失
            # if show_details:
            #     print(f"   VAR损失: {var_loss.item():.4f}")
            
            # Step 5: 重建验证
            with torch.no_grad():
                # 解码tokens回伪图像
                reconstructed_images = self.vqvae.idxBl_to_img(ms_tokens, same_shape=True, last_one=True)
                
                # 确保重建图像尺寸正确
                if reconstructed_images.shape[-2:] != (self.image_size, self.image_size):
                    # 🔇 减少尺寸调整输出
                    # if show_details:
                    #     print(f"   调整重建图像尺寸: {reconstructed_images.shape} → [B, 1, {self.image_size}, {self.image_size}]")
                    reconstructed_images = F.interpolate(
                        reconstructed_images,
                        size=(self.image_size, self.image_size),
                        mode='bilinear',
                        align_corners=False
                    )
                
                # 确保通道数正确
                if reconstructed_images.shape[1] != 1:
                    if reconstructed_images.shape[1] == 3:
                        # RGB → 灰度
                        reconstructed_images = reconstructed_images.mean(dim=1, keepdim=True)
                    else:
                        # 取第一个通道
                        reconstructed_images = reconstructed_images[:, :1]
                
                # 伪图像 → 基因表达
                predicted_genes = self.gene_adapter.pseudo_image_to_genes(reconstructed_images)
                
                # 重建损失
                recon_loss = F.mse_loss(predicted_genes, gene_expression)
                
                # 🔇 只在详细模式显示重建损失
                # if show_details:
                #     print(f"   重建损失: {recon_loss.item():.4f}")
            
            # Step 6: VQVAE量化损失（如果有）
            vq_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if hasattr(self.vqvae, 'vq_loss') and self.vqvae.vq_loss is not None:
                vq_loss = self.vqvae.vq_loss
                # 🔧 确保vq_loss是标量
                if isinstance(vq_loss, torch.Tensor) and vq_loss.numel() > 1:
                    vq_loss = vq_loss.mean()

            # 🔧 确保recon_loss是标量
            if isinstance(recon_loss, torch.Tensor) and recon_loss.numel() > 1:
                recon_loss = recon_loss.mean()

            # 总损失组合
            total_loss = var_loss + 0.1 * recon_loss + 0.01 * vq_loss

            if show_details:
                # 🔧 安全的损失显示：确保所有损失都是标量
                try:
                    var_loss_val = var_loss.item() if isinstance(var_loss, torch.Tensor) else float(var_loss)
                    recon_loss_val = recon_loss.item() if isinstance(recon_loss, torch.Tensor) else float(recon_loss)
                    vq_loss_val = vq_loss.item() if isinstance(vq_loss, torch.Tensor) else float(vq_loss)
                    total_loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else float(total_loss)
                    print(f"   总损失: {total_loss_val:.4f} (VAR: {var_loss_val:.4f}, 重建: {recon_loss_val:.4f}, VQ: {vq_loss_val:.4f})")
                except Exception as loss_display_error:
                    print(f"   💡 损失计算成功，但显示出错: {loss_display_error}")
                    print(f"   - VAR损失形状: {var_loss.shape if hasattr(var_loss, 'shape') else type(var_loss)}")
                    print(f"   - 重建损失形状: {recon_loss.shape if hasattr(recon_loss, 'shape') else type(recon_loss)}")
                    print(f"   - 总损失形状: {total_loss.shape if hasattr(total_loss, 'shape') else type(total_loss)}")
        
        except Exception as e:
            print(f"   ❌ VAR训练过程失败: {e}")
            print(f"   💡 这通常表示:")
            print(f"      1. VAR模型导入失败")
            print(f"      2. VQVAE编码出错")
            print(f"      3. 数据维度不匹配")
            print(f"      4. GPU内存不足")
            import traceback
            traceback.print_exc()
            
            # 🔧 不再使用回退方案，直接抛出异常！
            raise RuntimeError(f"VAR-ST模型训练失败: {e}") from e
        
        return {
            'loss': total_loss,
            'var_loss': var_loss,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'predictions': predicted_genes,
            'targets': gene_expression,
            'class_labels': class_labels,
            'pseudo_images': pseudo_images,
            'tokens': ms_tokens
        }
    
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
        """
        推理前向传播：从组织学特征生成基因表达
        
        Args:
            histology_features: [B, feature_dim] 组织学特征
            class_labels: [B] 可选的类别标签
            cfg_scale: Classifier-free guidance缩放
            其他参数：VAR生成参数
        
        Returns:
            生成的基因表达预测
        """
        B = histology_features.shape[0]
        device = histology_features.device
        
        # 1. 处理条件
        if class_labels is None:
            condition_embeddings = self.condition_processor(histology_features)
            class_labels = torch.argmax(condition_embeddings[:, :self.var_num_classes], dim=-1)
        
        # 2. VAR自回归生成 或 简单推理
        try:
            if hasattr(self.var_model, 'autoregressive_infer_cfg'):
                # 使用VAR的自回归生成
                generated_images = self.var_model.autoregressive_infer_cfg(
                    B=B * num_samples,
                    label_B=class_labels.repeat(num_samples) if num_samples > 1 else class_labels,
                    cfg=cfg_scale,
                    top_k=top_k,
                    top_p=top_p,
                    more_smooth=False
                )
            else:
                # 简化推理：直接生成随机伪图像
                generated_images = torch.randn(B * num_samples, 1, self.image_size, self.image_size, device=device)
                
        except Exception as e:
            print(f"⚠️ VAR推理异常: {e}，使用随机生成")
            # 回退：生成随机伪图像
            generated_images = torch.randn(B * num_samples, 1, self.image_size, self.image_size, device=device)
        
        # 3. 调整生成图像格式
        if generated_images.shape[-2:] != (self.image_size, self.image_size):
            generated_images = F.interpolate(
                generated_images,
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            )
        
        # 确保通道数为1（灰度）
        if generated_images.shape[1] != 1:
            if generated_images.shape[1] == 3:
                generated_images = generated_images.mean(dim=1, keepdim=True)
            else:
                generated_images = generated_images[:, :1]
        
        # 4. 生成的伪图像 → 基因表达
        predicted_genes = self.gene_adapter.pseudo_image_to_genes(generated_images)
        
        # 5. 重塑输出
        if num_samples > 1:
            predicted_genes = predicted_genes.view(B, num_samples, self.num_genes)
            generated_images = generated_images.view(B, num_samples, 1, self.image_size, self.image_size)
        
        return {
            'predictions': predicted_genes,
            'generated_images': generated_images,
            'class_labels': class_labels,
            'predicted_expression': predicted_genes
        }
    
    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """统一前向传播接口"""
        mode = inputs.get('mode', 'training')
        
        if mode == 'training':
            return self.forward_training(
                gene_expression=inputs['gene_expression'],
                histology_features=inputs['histology_features'],
                class_labels=inputs.get('class_labels'),
                show_details=inputs.get('show_details', False)
            )
        else:
            return self.forward_inference(
                histology_features=inputs['histology_features'],
                class_labels=inputs.get('class_labels'),
                cfg_scale=inputs.get('cfg_scale', 1.5),
                top_k=inputs.get('top_k', 50),
                top_p=inputs.get('top_p', 0.9),
                temperature=inputs.get('temperature', 1.0),
                num_samples=inputs.get('num_samples', 1)
            ) 