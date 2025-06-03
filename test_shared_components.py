"""
共享组件测试脚本

测试以下组件：
1. SharedVectorQuantizer: 共享量化器
2. MultiScaleEncoder: 多尺度编码器
3. MultiScaleDecoder: 多尺度解码器
4. ResidualReconstructor: 残差重建器
5. MultiScaleDecomposer: 多尺度分解器

验证数据流和形状变化的正确性
"""

import sys
import os
sys.path.append('src')

import torch
import torch.nn.functional as F
import numpy as np
from model.VAR.shared_components import (
    SharedVectorQuantizer,
    GlobalEncoder, PathwayEncoder, ModuleEncoder, IndividualEncoder,
    GlobalDecoder, PathwayDecoder, ModuleDecoder, IndividualDecoder,
    ResidualReconstructor,
    MultiScaleDecomposer
)


def test_shared_vector_quantizer():
    """测试共享向量量化器"""
    print("🔧 测试 SharedVectorQuantizer...")
    
    vq = SharedVectorQuantizer(vocab_size=4096, embed_dim=128, beta=0.25)
    
    # 测试不同输入形状
    test_cases = [
        torch.randn(4, 128),          # [B, embed_dim]
        torch.randn(4, 1, 128),       # [B, 1, embed_dim] 
        torch.randn(4, 8, 128),       # [B, 8, embed_dim]
        torch.randn(4, 32, 128),      # [B, 32, embed_dim]
        torch.randn(4, 200, 128),     # [B, 200, embed_dim]
    ]
    
    expected_token_shapes = [
        (4,),           # [B]
        (4, 1),         # [B, 1]
        (4, 8),         # [B, 8]
        (4, 32),        # [B, 32]
        (4, 200),       # [B, 200]
    ]
    
    for i, (x, expected_shape) in enumerate(zip(test_cases, expected_token_shapes)):
        tokens, quantized, vq_loss = vq(x)
        
        # 验证形状
        assert tokens.shape == expected_shape, f"案例{i+1}: tokens形状错误 {tokens.shape} != {expected_shape}"
        assert quantized.shape == x.shape, f"案例{i+1}: quantized形状错误 {quantized.shape} != {x.shape}"
        assert isinstance(vq_loss, torch.Tensor) and vq_loss.dim() == 0, f"案例{i+1}: vq_loss应该是标量"
        
        # 验证token范围
        assert tokens.min() >= 0, f"案例{i+1}: tokens最小值应该>=0"
        assert tokens.max() < 4096, f"案例{i+1}: tokens最大值应该<4096"
        
        # 测试解码
        decoded = vq.decode(tokens)
        assert decoded.shape == quantized.shape, f"案例{i+1}: 解码形状错误"
        
        print(f"   ✅ 案例{i+1}: {x.shape} → tokens{tokens.shape}, quantized{quantized.shape}, loss={vq_loss.item():.4f}")
    
    print("   ✅ SharedVectorQuantizer测试通过！")


def test_multi_scale_decomposer():
    """测试多尺度分解器"""
    print("\n🔧 测试 MultiScaleDecomposer...")
    
    decomposer = MultiScaleDecomposer()
    
    # 创建测试数据
    B = 4
    gene_expression = torch.randn(B, 200)
    
    # 多尺度分解
    decomposed = decomposer(gene_expression)
    
    # 验证输出
    expected_shapes = {
        'global': (B, 1),
        'pathway': (B, 8),
        'module': (B, 32),
        'individual': (B, 200)
    }
    
    for key, expected_shape in expected_shapes.items():
        assert key in decomposed, f"缺少{key}分解结果"
        actual_shape = decomposed[key].shape
        assert actual_shape == expected_shape, f"{key}形状错误: {actual_shape} != {expected_shape}"
        print(f"   ✅ {key}: {actual_shape}")
    
    # 验证global层是整体平均
    expected_global = gene_expression.mean(dim=1, keepdim=True)
    torch.testing.assert_close(decomposed['global'], expected_global, rtol=1e-5, atol=1e-6)
    
    print("   ✅ MultiScaleDecomposer测试通过！")


def test_multi_scale_encoders():
    """测试多尺度编码器"""
    print("\n🔧 测试 MultiScale Encoders...")
    
    encoders = {
        'global': GlobalEncoder(embed_dim=128),
        'pathway': PathwayEncoder(embed_dim=128),
        'module': ModuleEncoder(embed_dim=128),
        'individual': IndividualEncoder(embed_dim=128)
    }
    
    # 创建测试输入
    B = 4
    inputs = {
        'global': torch.randn(B, 1),
        'pathway': torch.randn(B, 8), 
        'module': torch.randn(B, 32),
        'individual': torch.randn(B, 200)
    }
    
    expected_output_shapes = {
        'global': (B, 1, 128),
        'pathway': (B, 8, 128),
        'module': (B, 32, 128),
        'individual': (B, 200, 128)
    }
    
    for name, encoder in encoders.items():
        input_tensor = inputs[name]
        encoded = encoder(input_tensor)
        expected_shape = expected_output_shapes[name]
        
        assert encoded.shape == expected_shape, f"{name}编码器输出形状错误: {encoded.shape} != {expected_shape}"
        print(f"   ✅ {name}编码器: {input_tensor.shape} → {encoded.shape}")
    
    print("   ✅ MultiScale Encoders测试通过！")


def test_multi_scale_decoders():
    """测试多尺度解码器"""
    print("\n🔧 测试 MultiScale Decoders...")
    
    decoders = {
        'global': GlobalDecoder(embed_dim=128),
        'pathway': PathwayDecoder(embed_dim=128),
        'module': ModuleDecoder(embed_dim=128),
        'individual': IndividualDecoder(embed_dim=128)
    }
    
    # 创建测试输入 (量化后的特征)
    B = 4
    inputs = {
        'global': torch.randn(B, 1, 128),
        'pathway': torch.randn(B, 8, 128),
        'module': torch.randn(B, 32, 128),
        'individual': torch.randn(B, 200, 128)
    }
    
    expected_output_shapes = {
        'global': (B, 1),
        'pathway': (B, 8),
        'module': (B, 32),
        'individual': (B, 200)
    }
    
    for name, decoder in decoders.items():
        input_tensor = inputs[name]
        decoded = decoder(input_tensor)
        expected_shape = expected_output_shapes[name]
        
        assert decoded.shape == expected_shape, f"{name}解码器输出形状错误: {decoded.shape} != {expected_shape}"
        print(f"   ✅ {name}解码器: {input_tensor.shape} → {decoded.shape}")
    
    print("   ✅ MultiScale Decoders测试通过！")


def test_residual_reconstructor():
    """测试残差重建器"""
    print("\n🔧 测试 ResidualReconstructor...")
    
    reconstructor = ResidualReconstructor()
    
    # 创建测试输入
    B = 4
    global_recon = torch.randn(B, 1)
    pathway_recon = torch.randn(B, 8)
    module_recon = torch.randn(B, 32)
    individual_recon = torch.randn(B, 200)
    
    # 残差重建
    result = reconstructor(global_recon, pathway_recon, module_recon, individual_recon)
    
    # 验证输出
    expected_keys = [
        'global_broadcast', 'pathway_broadcast', 'module_broadcast',
        'individual_contribution', 'cumulative_without_individual', 'final_reconstruction'
    ]
    
    for key in expected_keys:
        assert key in result, f"缺少输出键: {key}"
        assert result[key].shape == (B, 200), f"{key}形状错误: {result[key].shape} != {(B, 200)}"
    
    # 验证重建逻辑
    expected_final = (result['global_broadcast'] + 
                     result['pathway_broadcast'] + 
                     result['module_broadcast'] + 
                     result['individual_contribution'])
    
    torch.testing.assert_close(result['final_reconstruction'], expected_final, rtol=1e-5, atol=1e-6)
    
    # 验证广播逻辑
    expected_global_broadcast = global_recon.expand(B, 200)
    torch.testing.assert_close(result['global_broadcast'], expected_global_broadcast, rtol=1e-5, atol=1e-6)
    
    expected_pathway_broadcast = pathway_recon.repeat_interleave(25, dim=1)
    torch.testing.assert_close(result['pathway_broadcast'], expected_pathway_broadcast, rtol=1e-5, atol=1e-6)
    
    print(f"   ✅ 残差重建: 各层形状 {[v.shape for v in result.values()]}")
    print("   ✅ ResidualReconstructor测试通过！")


def test_complete_pipeline():
    """测试完整的数据流pipeline"""
    print("\n🔧 测试完整数据流Pipeline...")
    
    # 初始化所有组件
    decomposer = MultiScaleDecomposer()
    vq = SharedVectorQuantizer(vocab_size=4096, embed_dim=128, beta=0.25)
    
    encoders = {
        'global': GlobalEncoder(embed_dim=128),
        'pathway': PathwayEncoder(embed_dim=128),
        'module': ModuleEncoder(embed_dim=128),
        'individual': IndividualEncoder(embed_dim=128)
    }
    
    decoders = {
        'global': GlobalDecoder(embed_dim=128),
        'pathway': PathwayDecoder(embed_dim=128),
        'module': ModuleDecoder(embed_dim=128),
        'individual': IndividualDecoder(embed_dim=128)
    }
    
    reconstructor = ResidualReconstructor()
    
    # 创建测试数据
    B = 4
    gene_expression = torch.randn(B, 200)
    
    print(f"   输入基因表达: {gene_expression.shape}")
    
    # Step 1: 多尺度分解
    decomposed = decomposer(gene_expression)
    print(f"   多尺度分解: {[f'{k}:{v.shape}' for k, v in decomposed.items()]}")
    
    # Step 2: 多尺度编码
    encoded = {}
    for scale in ['global', 'pathway', 'module', 'individual']:
        encoded[scale] = encoders[scale](decomposed[scale])
    print(f"   多尺度编码: {[f'{k}:{v.shape}' for k, v in encoded.items()]}")
    
    # Step 3: 共享量化
    quantized = {}
    tokens = {}
    vq_losses = []
    
    for scale in ['global', 'pathway', 'module', 'individual']:
        scale_tokens, scale_quantized, scale_vq_loss = vq(encoded[scale])
        tokens[scale] = scale_tokens
        quantized[scale] = scale_quantized
        vq_losses.append(scale_vq_loss)
    
    total_vq_loss = sum(vq_losses)
    print(f"   共享量化: tokens{[f'{k}:{v.shape}' for k, v in tokens.items()]}")
    print(f"   VQ损失: {[f'{loss.item():.4f}' for loss in vq_losses]}, 总计: {total_vq_loss.item():.4f}")
    
    # Step 4: 多尺度解码
    decoded = {}
    for scale in ['global', 'pathway', 'module', 'individual']:
        decoded[scale] = decoders[scale](quantized[scale])
    print(f"   多尺度解码: {[f'{k}:{v.shape}' for k, v in decoded.items()]}")
    
    # Step 5: 残差重建
    reconstruction_result = reconstructor(
        decoded['global'], decoded['pathway'], 
        decoded['module'], decoded['individual']
    )
    
    final_reconstruction = reconstruction_result['final_reconstruction']
    print(f"   最终重建: {final_reconstruction.shape}")
    
    # 计算重建误差
    reconstruction_loss = F.mse_loss(final_reconstruction, gene_expression)
    print(f"   重建MSE损失: {reconstruction_loss.item():.4f}")
    
    # 验证所有token都在正确范围内
    all_tokens = torch.cat([tokens[scale].flatten() for scale in tokens.keys()])
    assert all_tokens.min() >= 0, f"Token最小值 {all_tokens.min()} < 0"
    assert all_tokens.max() < 4096, f"Token最大值 {all_tokens.max()} >= 4096"
    
    print(f"   所有tokens范围: [{all_tokens.min()}, {all_tokens.max()}]")
    print("   ✅ 完整Pipeline测试通过！")
    
    return {
        'input': gene_expression,
        'decomposed': decomposed,
        'encoded': encoded,
        'tokens': tokens,
        'quantized': quantized,
        'decoded': decoded,
        'reconstructed': final_reconstruction,
        'reconstruction_loss': reconstruction_loss,
        'vq_loss': total_vq_loss
    }


def test_var_compatibility():
    """测试与VAR原始设计的兼容性"""
    print("\n🔧 测试VAR兼容性...")
    
    # 验证关键参数与VAR一致
    vq = SharedVectorQuantizer(vocab_size=4096, embed_dim=128)
    
    # 测试不同batch size
    for batch_size in [1, 4, 16, 32]:
        # 测试所有可能的sequence长度
        test_sequences = [
            torch.randn(batch_size, 1, 128),    # Global
            torch.randn(batch_size, 8, 128),    # Pathway  
            torch.randn(batch_size, 32, 128),   # Module
            torch.randn(batch_size, 200, 128),  # Individual
            torch.randn(batch_size, 241, 128),  # 完整序列 1+8+32+200
        ]
        
        for i, seq in enumerate(test_sequences):
            tokens, quantized, vq_loss = vq(seq)
            
            # 验证tokens范围
            assert tokens.min() >= 0 and tokens.max() < 4096, f"Batch{batch_size}序列{i}: tokens范围错误"
            
            # 验证形状一致性
            assert quantized.shape == seq.shape, f"Batch{batch_size}序列{i}: 量化形状不一致"
            
        print(f"   ✅ Batch size {batch_size}: 所有序列长度测试通过")
    
    print("   ✅ VAR兼容性测试通过！")


if __name__ == "__main__":
    print("🚀 开始测试共享组件...")
    
    try:
        # 执行所有测试
        test_shared_vector_quantizer()
        test_multi_scale_decomposer()
        test_multi_scale_encoders()
        test_multi_scale_decoders()
        test_residual_reconstructor()
        
        # 完整流程测试
        pipeline_result = test_complete_pipeline()
        
        # VAR兼容性测试
        test_var_compatibility()
        
        print(f"\n🎉 所有测试通过！")
        print(f"📊 Pipeline结果摘要:")
        print(f"   - 输入维度: {pipeline_result['input'].shape}")
        print(f"   - 重建损失: {pipeline_result['reconstruction_loss'].item():.4f}")
        print(f"   - VQ损失: {pipeline_result['vq_loss'].item():.4f}")
        print(f"   - Token数量: {sum(t.numel() for t in pipeline_result['tokens'].values())}")
        
        print(f"\n✅ Step 1 共享组件模块创建完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 