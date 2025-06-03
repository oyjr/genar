"""
Stage 1多尺度基因VQVAE测试脚本

测试以下功能：
1. MultiScaleGeneVQVAE模型完整性
2. 编码解码流程正确性
3. 损失计算准确性
4. checkpoint保存和加载
5. Stage1Trainer训练流程
6. 与共享组件的集成

验证Stage 1的准备情况
"""

import sys
import os
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model.VAR.multi_scale_gene_vqvae import MultiScaleGeneVQVAE, Stage1Trainer


def test_multi_scale_gene_vqvae():
    """测试多尺度基因VQVAE模型"""
    print("🧬 测试 MultiScaleGeneVQVAE...")
    
    # 初始化模型
    model = MultiScaleGeneVQVAE(
        vocab_size=4096,
        embed_dim=128,
        beta=0.25,
        hierarchical_loss_weight=0.1,
        vq_loss_weight=0.25
    )
    
    print(f"   模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建测试数据
    B = 8
    gene_expression = torch.randn(B, 200)
    
    print(f"   输入基因表达: {gene_expression.shape}")
    
    # 1. 测试编码
    encode_result = model.encode(gene_expression)
    print(f"   编码结果keys: {list(encode_result.keys())}")
    
    # 验证tokens形状
    expected_token_shapes = {
        'global': (B, 1),
        'pathway': (B, 8),
        'module': (B, 32),
        'individual': (B, 200)
    }
    
    for scale, expected_shape in expected_token_shapes.items():
        actual_shape = encode_result['tokens'][scale].shape
        assert actual_shape == expected_shape, f"{scale} tokens形状错误: {actual_shape} != {expected_shape}"
        print(f"   ✅ {scale} tokens: {actual_shape}")
    
    # 2. 测试解码
    decode_result = model.decode(encode_result['quantized'])
    print(f"   解码结果keys: {list(decode_result.keys())}")
    
    final_recon = decode_result['final_reconstruction']
    assert final_recon.shape == gene_expression.shape, f"重建形状错误: {final_recon.shape} != {gene_expression.shape}"
    print(f"   ✅ 最终重建: {final_recon.shape}")
    
    # 3. 测试从tokens解码
    tokens_decode_result = model.decode_from_tokens(encode_result['tokens'])
    assert tokens_decode_result['final_reconstruction'].shape == gene_expression.shape
    print(f"   ✅ 从tokens解码: {tokens_decode_result['final_reconstruction'].shape}")
    
    # 4. 测试完整前向传播
    forward_result = model(gene_expression)
    print(f"   前向传播结果keys: {list(forward_result.keys())}")
    
    # 验证损失存在
    required_losses = ['total_loss', 'total_reconstruction_loss', 'total_hierarchical_loss', 'total_vq_loss']
    for loss_name in required_losses:
        assert loss_name in forward_result, f"缺少损失: {loss_name}"
        loss_value = forward_result[loss_name]
        assert isinstance(loss_value, torch.Tensor) and loss_value.dim() == 0, f"{loss_name}应该是标量"
        print(f"   ✅ {loss_name}: {loss_value.item():.4f}")
    
    # 5. 测试codebook利用率
    utilization = model.update_codebook_usage(encode_result['tokens'])
    print(f"   ✅ Codebook利用率: {utilization:.4f}")
    
    # 6. 测试随机token生成
    random_tokens = model.generate_random_tokens(batch_size=4, device=torch.device('cpu'))
    for scale, expected_shape in [('global', (4, 1)), ('pathway', (4, 8)), ('module', (4, 32)), ('individual', (4, 200))]:
        assert random_tokens[scale].shape == expected_shape, f"随机{scale} tokens形状错误"
        assert random_tokens[scale].min() >= 0 and random_tokens[scale].max() < 4096, f"随机{scale} tokens范围错误"
    print(f"   ✅ 随机tokens生成: {[f'{k}:{v.shape}' for k, v in random_tokens.items()]}")
    
    print("   ✅ MultiScaleGeneVQVAE测试通过！")
    return model


def test_loss_computation():
    """测试损失计算的正确性"""
    print("\n🧬 测试损失计算...")
    
    model = MultiScaleGeneVQVAE()
    
    # 创建测试数据
    B = 4
    gene_expression = torch.randn(B, 200)
    
    # 前向传播
    result = model(gene_expression)
    
    # 验证损失关系
    total_loss = result['total_loss']
    recon_loss = result['total_reconstruction_loss']
    hier_loss = result['total_hierarchical_loss']
    vq_loss = result['total_vq_loss']
    
    # 手动计算期望的总损失
    expected_total = recon_loss + 0.1 * hier_loss + 0.25 * vq_loss
    
    # 验证损失计算正确性
    torch.testing.assert_close(total_loss, expected_total, rtol=1e-5, atol=1e-6)
    print(f"   ✅ 损失计算正确: {total_loss.item():.4f} == {expected_total.item():.4f}")
    
    # 验证重建损失合理性
    manual_recon_loss = torch.nn.functional.mse_loss(result['final_reconstruction'], gene_expression)
    torch.testing.assert_close(recon_loss, manual_recon_loss, rtol=1e-5, atol=1e-6)
    print(f"   ✅ 重建损失正确: {recon_loss.item():.4f}")
    
    # 验证分层损失
    decomposed = result['decomposed']
    decoded = result['decoded']
    manual_hier_losses = []
    
    for scale in ['global', 'pathway', 'module', 'individual']:
        scale_loss = torch.nn.functional.mse_loss(decoded[scale], decomposed[scale])
        manual_hier_losses.append(scale_loss)
        
        # 检查individual scale loss是否存在
        assert f'{scale}_recon_loss' in result, f"缺少{scale}_recon_loss"
        torch.testing.assert_close(result[f'{scale}_recon_loss'], scale_loss, rtol=1e-5, atol=1e-6)
    
    manual_total_hier = sum(manual_hier_losses)
    torch.testing.assert_close(hier_loss, manual_total_hier, rtol=1e-5, atol=1e-6)
    print(f"   ✅ 分层损失正确: {hier_loss.item():.4f}")
    
    print("   ✅ 损失计算测试通过！")


def test_checkpoint_save_load():
    """测试checkpoint保存和加载"""
    print("\n🧬 测试Checkpoint保存和加载...")
    
    # 创建原始模型
    original_model = MultiScaleGeneVQVAE(vocab_size=512, embed_dim=64)  # 使用较小参数便于测试
    
    # 创建测试数据并训练几步
    gene_expression = torch.randn(4, 200)
    optimizer = optim.Adam(original_model.parameters(), lr=1e-3)
    
    # 训练几步
    for step in range(3):
        optimizer.zero_grad()
        result = original_model(gene_expression)
        loss = result['total_loss']
        loss.backward()
        optimizer.step()
    
    # 保存checkpoint
    checkpoint_path = "test_stage1_checkpoint.pth"
    original_model.save_stage1_checkpoint(
        path=checkpoint_path,
        epoch=10,
        optimizer_state=optimizer.state_dict()
    )
    
    # 加载checkpoint
    loaded_model, checkpoint_info = MultiScaleGeneVQVAE.load_stage1_checkpoint(
        path=checkpoint_path,
        device=torch.device('cpu')
    )
    
    # 验证加载的模型
    assert checkpoint_info['epoch'] == 10, f"epoch加载错误: {checkpoint_info['epoch']} != 10"
    assert checkpoint_info['stage'] == 'stage1_vqvae', f"stage信息错误: {checkpoint_info['stage']}"
    assert checkpoint_info['optimizer_state_dict'] is not None, "优化器状态未保存"
    
    # 验证模型参数一致性
    original_result = original_model(gene_expression)
    loaded_result = loaded_model(gene_expression)
    
    torch.testing.assert_close(
        original_result['final_reconstruction'], 
        loaded_result['final_reconstruction'], 
        rtol=1e-6, atol=1e-7
    )
    
    print(f"   ✅ Checkpoint保存和加载正确")
    print(f"   ✅ 模型配置: {checkpoint_info}")
    
    # 清理测试文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    print("   ✅ Checkpoint测试通过！")
    return loaded_model


def test_stage1_trainer():
    """测试Stage1训练器"""
    print("\n🧬 测试Stage1Trainer...")
    
    # 创建模型和优化器
    model = MultiScaleGeneVQVAE(vocab_size=256, embed_dim=64)  # 小模型便于测试
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device('cpu')
    
    # 创建训练器
    trainer = Stage1Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        print_freq=2
    )
    
    # 创建模拟数据
    num_samples = 32
    gene_expressions = torch.randn(num_samples, 200)
    
    # 创建数据加载器
    dataset = TensorDataset(gene_expressions)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    print(f"   训练数据: {num_samples}个样本, 批次大小: 8")
    
    # 测试训练一个epoch
    print(f"   开始训练epoch...")
    train_losses = trainer.train_epoch(train_loader, epoch=1)
    
    # 验证训练损失
    expected_keys = ['total_loss', 'reconstruction_loss', 'hierarchical_loss', 'vq_loss', 'codebook_utilization']
    for key in expected_keys:
        assert key in train_losses, f"缺少训练损失: {key}"
        print(f"   ✅ 训练{key}: {train_losses[key]:.4f}")
    
    # 测试验证一个epoch
    print(f"   开始验证epoch...")
    val_losses = trainer.validate_epoch(val_loader, epoch=1)
    
    # 验证验证损失
    for key in expected_keys:
        assert key in val_losses, f"缺少验证损失: {key}"
        print(f"   ✅ 验证{key}: {val_losses[key]:.4f}")
    
    # 测试统计信息
    stats = trainer.get_training_stats()
    assert 'epoch_losses' in stats, "缺少epoch损失统计"
    assert 'codebook_utilizations' in stats, "缺少codebook利用率统计"
    assert stats['num_epochs_trained'] == 1, f"训练epoch数错误: {stats['num_epochs_trained']} != 1"
    
    print(f"   ✅ 训练统计: {stats['num_epochs_trained']}个epoch已完成")
    print(f"   ✅ Codebook利用率历史: {stats['codebook_utilizations']}")
    
    print("   ✅ Stage1Trainer测试通过！")
    return trainer


def test_integration_with_shared_components():
    """测试与共享组件的集成"""
    print("\n🧬 测试与共享组件集成...")
    
    model = MultiScaleGeneVQVAE()
    
    # 创建测试数据
    B = 6
    gene_expression = torch.randn(B, 200)
    
    # 测试完整pipeline
    result = model(gene_expression)
    
    # 验证所有组件都正常工作
    assert 'decomposed' in result, "多尺度分解失败"
    assert 'encoded' in result, "多尺度编码失败"  
    assert 'tokens' in result, "共享量化失败"
    assert 'quantized' in result, "量化特征缺失"
    assert 'decoded' in result, "多尺度解码失败"
    assert 'final_reconstruction' in result, "残差重建失败"
    
    # 验证数据流一致性
    decomposed = result['decomposed']
    tokens = result['tokens']
    final_recon = result['final_reconstruction']
    
    # 验证每个尺度的处理
    for scale in ['global', 'pathway', 'module', 'individual']:
        assert scale in decomposed, f"缺少{scale}分解"
        assert scale in tokens, f"缺少{scale}tokens"
        
        # 验证tokens范围
        scale_tokens = tokens[scale]
        assert scale_tokens.min() >= 0, f"{scale} tokens最小值错误"
        assert scale_tokens.max() < 4096, f"{scale} tokens最大值错误"
    
    # 验证重建质量
    reconstruction_error = torch.nn.functional.mse_loss(final_recon, gene_expression)
    print(f"   ✅ 重建误差: {reconstruction_error.item():.4f}")
    
    # 验证token总数
    total_tokens = sum(tokens[scale].numel() for scale in tokens.keys())
    expected_total = B * (1 + 8 + 32 + 200)  # B * (1+8+32+200) = B * 241
    assert total_tokens == expected_total, f"Token总数错误: {total_tokens} != {expected_total}"
    print(f"   ✅ Token总数: {total_tokens} (期望: {expected_total})")
    
    # 验证残差重建的累积性质
    recon_result = result['reconstruction_result']
    cumulative = recon_result['cumulative_without_individual']
    individual = recon_result['individual_contribution']
    final = recon_result['final_reconstruction']
    
    expected_final = cumulative + individual
    torch.testing.assert_close(final, expected_final, rtol=1e-5, atol=1e-6)
    print(f"   ✅ 残差重建逻辑正确")
    
    print("   ✅ 与共享组件集成测试通过！")


def test_different_batch_sizes():
    """测试不同批次大小的兼容性"""
    print("\n🧬 测试不同批次大小...")
    
    model = MultiScaleGeneVQVAE()
    
    test_batch_sizes = [1, 4, 16, 32]
    
    for batch_size in test_batch_sizes:
        gene_expression = torch.randn(batch_size, 200)
        
        # 测试前向传播
        result = model(gene_expression)
        
        # 验证输出形状
        assert result['final_reconstruction'].shape == (batch_size, 200), f"批次{batch_size}重建形状错误"
        
        # 验证tokens形状
        expected_token_shapes = {
            'global': (batch_size, 1),
            'pathway': (batch_size, 8),
            'module': (batch_size, 32),
            'individual': (batch_size, 200)
        }
        
        for scale, expected_shape in expected_token_shapes.items():
            actual_shape = result['tokens'][scale].shape
            assert actual_shape == expected_shape, f"批次{batch_size} {scale} tokens形状错误"
        
        print(f"   ✅ 批次大小 {batch_size}: 所有测试通过")
    
    print("   ✅ 不同批次大小测试通过！")


if __name__ == "__main__":
    print("🚀 开始测试Stage 1多尺度基因VQVAE...")
    
    try:
        # 执行所有测试
        model = test_multi_scale_gene_vqvae()
        test_loss_computation()
        loaded_model = test_checkpoint_save_load()
        trainer = test_stage1_trainer()
        test_integration_with_shared_components()
        test_different_batch_sizes()
        
        print(f"\n🎉 所有测试通过！")
        print(f"📊 Stage 1 VQVAE测试摘要:")
        print(f"   - 模型参数数量: {sum(p.numel() for p in model.parameters())}")
        print(f"   - 词汇表大小: {model.vocab_size}")
        print(f"   - 嵌入维度: {model.embed_dim}")
        print(f"   - 支持的批次大小: 1, 4, 16, 32+")
        print(f"   - 训练器功能: ✅ 正常")
        print(f"   - Checkpoint功能: ✅ 正常")
        
        print(f"\n✅ Step 2 多尺度基因VQVAE创建完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 