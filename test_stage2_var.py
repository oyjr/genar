"""
Stage 2基因VAR Transformer测试脚本

测试以下功能：
1. ConditionProcessor条件处理器
2. GeneVARTransformer模型架构
3. Stage2Trainer训练流程
4. 条件生成功能
5. 与Stage 1 VQVAE的集成

验证Stage 2的准备情况
"""

import sys
import os
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model.VAR.gene_var_transformer import (
    ConditionProcessor, 
    GeneVARTransformer, 
    Stage2Trainer
)
from model.VAR.multi_scale_gene_vqvae import MultiScaleGeneVQVAE


def test_condition_processor():
    """测试条件处理器"""
    print("🎯 测试 ConditionProcessor...")
    
    # 初始化条件处理器
    processor = ConditionProcessor(
        histology_dim=1024,
        spatial_dim=2,
        histology_hidden_dim=512,
        spatial_hidden_dim=128,
        condition_embed_dim=640
    )
    
    print(f"   参数数量: {sum(p.numel() for p in processor.parameters())}")
    
    # 测试数据
    B = 8
    histology_features = torch.randn(B, 1024)
    spatial_coords = torch.rand(B, 2)
    
    print(f"   输入 - 组织学特征: {histology_features.shape}")
    print(f"   输入 - 空间坐标: {spatial_coords.shape}")
    
    # 前向传播
    condition_embed = processor(histology_features, spatial_coords)
    
    print(f"   输出 - 条件嵌入: {condition_embed.shape}")
    assert condition_embed.shape == (B, 640), f"条件嵌入形状错误: {condition_embed.shape}"
    
    # 测试不同批次大小
    for batch_size in [1, 4, 16, 32]:
        hist_feat = torch.randn(batch_size, 1024)
        spatial = torch.rand(batch_size, 2)
        embed = processor(hist_feat, spatial)
        assert embed.shape == (batch_size, 640), f"批次大小{batch_size}测试失败"
    
    print("   ✅ ConditionProcessor测试通过！")


def test_gene_var_transformer():
    """测试基因VAR Transformer"""
    print("🤖 测试 GeneVARTransformer...")
    
    # 初始化模型
    transformer = GeneVARTransformer(
        vocab_size=4096,
        embed_dim=640,
        num_heads=8,
        num_layers=6,  # 减少层数用于测试
        feedforward_dim=2560,
        dropout=0.1,
        max_sequence_length=1500,
        condition_embed_dim=640
    )
    
    print(f"   参数数量: {sum(p.numel() for p in transformer.parameters())}")
    
    # 测试数据
    B = 4
    seq_len = 1446  # 多尺度token总数: 1+8+32+32*32*200 = 1446
    vocab_size = 4096
    
    input_tokens = torch.randint(0, vocab_size, (B, seq_len))
    condition_embed = torch.randn(B, 640)
    target_tokens = torch.randint(0, vocab_size, (B, seq_len))
    
    print(f"   输入 - Token序列: {input_tokens.shape}")
    print(f"   输入 - 条件嵌入: {condition_embed.shape}")
    print(f"   输入 - 目标Token: {target_tokens.shape}")
    
    # 前向传播 (训练模式)
    outputs = transformer(input_tokens, condition_embed, target_tokens)
    
    print(f"   输出 - Logits: {outputs['logits'].shape}")
    print(f"   输出 - Loss: {outputs['loss'].item():.4f}")
    print(f"   输出 - Accuracy: {outputs['accuracy'].item():.4f}")
    
    assert outputs['logits'].shape == (B, seq_len, vocab_size), f"Logits形状错误: {outputs['logits'].shape}"
    assert 'loss' in outputs, "缺少loss输出"
    assert 'accuracy' in outputs, "缺少accuracy输出"
    
    # 前向传播 (推理模式)
    outputs_inference = transformer(input_tokens, condition_embed)
    assert 'loss' not in outputs_inference, "推理模式不应该有loss"
    assert outputs_inference['logits'].shape == (B, seq_len, vocab_size)
    
    print("   ✅ GeneVARTransformer前向传播测试通过！")
    
    # 测试生成功能
    print("   🎲 测试生成功能...")
    generated_tokens = transformer.generate(
        condition_embed=condition_embed,
        max_length=100,  # 较短长度用于测试
        temperature=1.0
    )
    
    print(f"   生成的Token序列: {generated_tokens.shape}")
    assert generated_tokens.shape == (B, 100), f"生成序列形状错误: {generated_tokens.shape}"
    print("   ✅ 生成功能测试通过！")


def test_stage2_trainer():
    """测试Stage 2训练器"""
    print("🏋️ 测试 Stage2Trainer...")
    
    # 创建测试用的Stage 1 VQVAE模型
    vqvae_model = MultiScaleGeneVQVAE(
        vocab_size=256,  # 较小的词汇表用于测试
        embed_dim=64,
        beta=0.25
    )
    
    # 创建条件处理器
    condition_processor = ConditionProcessor(
        histology_dim=1024,
        spatial_dim=2,
        condition_embed_dim=512  # 较小维度用于测试
    )
    
    # 创建VAR Transformer
    var_transformer = GeneVARTransformer(
        vocab_size=256,
        embed_dim=512,
        num_heads=4,
        num_layers=2,  # 极少层数用于快速测试
        feedforward_dim=1024,
        condition_embed_dim=512
    )
    
    # 创建训练器
    trainer = Stage2Trainer(
        vqvae_model=vqvae_model,
        var_transformer=var_transformer,
        condition_processor=condition_processor,
        device=torch.device('cpu'),  # 使用CPU测试
        learning_rate=1e-4,
        print_freq=1
    )
    
    print("   ✅ Stage2Trainer初始化成功！")
    
    # 验证VQVAE被冻结
    vqvae_trainable = sum(p.requires_grad for p in trainer.vqvae_model.parameters())
    transformer_trainable = sum(p.requires_grad for p in trainer.var_transformer.parameters())
    condition_trainable = sum(p.requires_grad for p in trainer.condition_processor.parameters())
    
    print(f"   VQVAE可训练参数: {vqvae_trainable}")
    print(f"   Transformer可训练参数: {transformer_trainable}")
    print(f"   条件处理器可训练参数: {condition_trainable}")
    
    assert vqvae_trainable == 0, "VQVAE应该被冻结"
    assert transformer_trainable > 0, "Transformer应该可训练"
    assert condition_trainable > 0, "条件处理器应该可训练"
    
    # 创建测试数据
    B, gene_dim = 16, 200
    gene_expressions = torch.randn(B, gene_dim)
    histology_features = torch.randn(B, 1024)
    spatial_coords = torch.rand(B, 2)
    
    train_dataset = TensorDataset(gene_expressions, histology_features, spatial_coords)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    print(f"   训练数据: {len(train_dataset)}个样本, 批次大小: 8")
    
    # 测试训练一个epoch
    print("   开始训练epoch...")
    train_stats = trainer.train_epoch(train_dataloader, epoch=1)
    
    print(f"   ✅ 训练Loss: {train_stats['loss']:.4f}")
    print(f"   ✅ 训练Accuracy: {train_stats['accuracy']:.4f}")
    
    # 测试验证
    print("   开始验证epoch...")
    val_stats = trainer.validate_epoch(train_dataloader, epoch=1)
    
    print(f"   ✅ 验证Loss: {val_stats['loss']:.4f}")
    print(f"   ✅ 验证Accuracy: {val_stats['accuracy']:.4f}")
    
    print("   ✅ Stage2Trainer测试通过！")


def test_checkpoint_functionality():
    """测试checkpoint功能"""
    print("💾 测试 Checkpoint功能...")
    
    # 创建简化模型
    vqvae_model = MultiScaleGeneVQVAE(vocab_size=256, embed_dim=64)
    condition_processor = ConditionProcessor(condition_embed_dim=256)
    var_transformer = GeneVARTransformer(
        vocab_size=256, 
        embed_dim=256, 
        num_heads=2, 
        num_layers=1,
        condition_embed_dim=256
    )
    
    trainer = Stage2Trainer(
        vqvae_model=vqvae_model,
        var_transformer=var_transformer,
        condition_processor=condition_processor,
        device=torch.device('cpu')
    )
    
    # 保存checkpoint
    checkpoint_path = "test_stage2_checkpoint.pth"
    trainer.save_checkpoint(checkpoint_path, epoch=5, metadata={"test": "stage2"})
    
    # 修改一些参数
    original_weight = trainer.var_transformer.token_embedding.weight.clone()
    trainer.var_transformer.token_embedding.weight.data.fill_(0.5)
    
    # 加载checkpoint
    checkpoint = trainer.load_checkpoint(checkpoint_path)
    loaded_weight = trainer.var_transformer.token_embedding.weight.clone()
    
    # 验证参数恢复
    assert torch.allclose(original_weight, loaded_weight), "Checkpoint加载失败"
    assert checkpoint['epoch'] == 5, f"Epoch不匹配: {checkpoint['epoch']}"
    assert checkpoint['metadata']['test'] == "stage2", "Metadata不匹配"
    
    # 清理
    os.remove(checkpoint_path)
    
    print("   ✅ Checkpoint测试通过！")


def test_integration_with_stage1():
    """测试与Stage 1的集成"""
    print("🔗 测试与Stage 1集成...")
    
    # 创建Stage 1模型并进行一次前向传播
    vqvae_model = MultiScaleGeneVQVAE(vocab_size=4096, embed_dim=128)
    
    # 创建测试数据
    B = 4
    gene_expressions = torch.randn(B, 200)
    
    # Stage 1编码
    with torch.no_grad():
        vqvae_result = vqvae_model(gene_expressions)
        tokens = vqvae_result['tokens']
    
    print(f"   Stage 1输出tokens:")
    total_tokens = 0
    for scale, scale_tokens in tokens.items():
        print(f"     {scale}: {scale_tokens.shape}")
        total_tokens += scale_tokens.numel() // B
    
    print(f"   每个样本的总token数: {total_tokens}")
    
    # 展平tokens为序列
    token_sequence = []
    for scale in ['global', 'pathway', 'module', 'individual']:
        scale_tokens = tokens[scale].view(B, -1)
        token_sequence.append(scale_tokens)
    
    full_token_sequence = torch.cat(token_sequence, dim=1)
    print(f"   展平后的token序列: {full_token_sequence.shape}")
    
    # 创建Stage 2模型
    condition_processor = ConditionProcessor()
    var_transformer = GeneVARTransformer(
        vocab_size=4096,
        max_sequence_length=full_token_sequence.shape[1] + 100
    )
    
    # 创建条件信息
    histology_features = torch.randn(B, 1024)
    spatial_coords = torch.rand(B, 2)
    condition_embed = condition_processor(histology_features, spatial_coords)
    
    # Stage 2前向传播
    outputs = var_transformer(full_token_sequence, condition_embed, full_token_sequence)
    
    print(f"   Stage 2输出:")
    print(f"     Logits: {outputs['logits'].shape}")
    print(f"     Loss: {outputs['loss'].item():.4f}")
    
    # 测试生成
    generated = var_transformer.generate(
        condition_embed=condition_embed,
        max_length=full_token_sequence.shape[1],
        temperature=1.0
    )
    
    print(f"   生成的token序列: {generated.shape}")
    
    print("   ✅ 与Stage 1集成测试通过！")


def test_different_input_formats():
    """测试不同输入格式"""
    print("📝 测试不同输入格式...")
    
    condition_processor = ConditionProcessor()
    
    # 测试不同的输入格式
    test_cases = [
        ("标准输入", torch.randn(4, 1024), torch.rand(4, 2)),
        ("单样本", torch.randn(1, 1024), torch.rand(1, 2)),
        ("大批次", torch.randn(32, 1024), torch.rand(32, 2))
    ]
    
    for name, hist_feat, spatial in test_cases:
        embed = condition_processor(hist_feat, spatial)
        print(f"   {name}: {hist_feat.shape} + {spatial.shape} → {embed.shape}")
        assert embed.shape[0] == hist_feat.shape[0], f"{name}批次大小不匹配"
        assert embed.shape[1] == 640, f"{name}嵌入维度错误"
    
    print("   ✅ 不同输入格式测试通过！")


if __name__ == "__main__":
    print("🧪 开始Stage 2基因VAR Transformer测试\n")
    
    try:
        test_condition_processor()
        print()
        
        test_gene_var_transformer()
        print()
        
        test_stage2_trainer()
        print()
        
        test_checkpoint_functionality()
        print()
        
        test_integration_with_stage1()
        print()
        
        test_different_input_formats()
        print()
        
        print("🎉 所有测试通过！")
        print("📊 Stage 2 VAR Transformer测试摘要:")
        print("   - 条件处理器: ✅ 正常")
        print("   - VAR Transformer: ✅ 正常")
        print("   - 训练器功能: ✅ 正常")
        print("   - Checkpoint功能: ✅ 正常")
        print("   - Stage 1集成: ✅ 正常")
        print("   - 生成功能: ✅ 正常")
        print()
        print("✅ Step 3 基因VAR Transformer创建完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 