"""
Two-Stage VAR-ST 内存占用分析
详细分析Stage 1和Stage 2的内存占用差异

运行此脚本来测量：
1. 模型参数数量
2. 训练时的GPU内存占用
3. 前向传播的内存峰值
4. 梯度和优化器状态的内存占用
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import psutil
import gc
from typing import Dict, Tuple
import tempfile

from model.VAR.two_stage_var_st import TwoStageVARST
from model.VAR.multi_scale_gene_vqvae import MultiScaleGeneVQVAE
from model.VAR.gene_var_transformer import GeneVARTransformer, ConditionProcessor


def get_gpu_memory_usage():
    """获取当前GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)    # GB
        return allocated, reserved
    return 0, 0


def get_model_parameters(model: nn.Module) -> Dict[str, int]:
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }


def analyze_stage1_memory():
    """分析Stage 1 (VQVAE) 的内存占用"""
    print("🔍 Stage 1 (VQVAE) 内存分析")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 创建Stage 1模型
    stage1_model = TwoStageVARST(
        num_genes=200,
        current_stage=1,
        device=device
    )
    stage1_model = stage1_model.to(device)
    
    # 分析模型参数
    stage1_params = get_model_parameters(stage1_model)
    vqvae_params = get_model_parameters(stage1_model.stage1_vqvae)
    
    print(f"📊 Stage 1 模型参数:")
    print(f"   完整模型参数: {stage1_params['total']:,}")
    print(f"   VQVAE参数: {vqvae_params['total']:,}")
    print(f"   可训练参数: {stage1_params['trainable']:,}")
    
    # 分析VQVAE各组件参数
    components_params = {}
    components_params['decomposer'] = get_model_parameters(stage1_model.stage1_vqvae.decomposer)['total']
    components_params['encoders'] = get_model_parameters(stage1_model.stage1_vqvae.encoders)['total']
    components_params['quantizer'] = get_model_parameters(stage1_model.stage1_vqvae.shared_quantizer)['total']
    components_params['decoders'] = get_model_parameters(stage1_model.stage1_vqvae.decoders)['total']
    components_params['reconstructor'] = get_model_parameters(stage1_model.stage1_vqvae.reconstructor)['total']
    
    print(f"\n🔧 VQVAE组件参数分布:")
    for component, params in components_params.items():
        percentage = (params / vqvae_params['total']) * 100
        print(f"   {component}: {params:,} ({percentage:.1f}%)")
    
    if torch.cuda.is_available():
        allocated_after_model, reserved_after_model = get_gpu_memory_usage()
        print(f"\n💾 模型加载后GPU内存:")
        print(f"   已分配: {allocated_after_model:.2f} GB")
        print(f"   已保留: {reserved_after_model:.2f} GB")
    
    # 模拟训练
    batch_sizes = [8, 16, 32, 64]
    print(f"\n🏋️ Stage 1 训练内存占用 (不同batch size):")
    
    for batch_size in batch_sizes:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            # 创建训练数据
            gene_expression = torch.randn(batch_size, 200, device=device)
            
            # 前向传播
            stage1_model.train()
            output = stage1_model(gene_expression)
            loss = output['loss']
            
            if torch.cuda.is_available():
                allocated_forward, _ = get_gpu_memory_usage()
                peak_forward = torch.cuda.max_memory_allocated() / (1024**3)
            
            # 反向传播
            loss.backward()
            
            if torch.cuda.is_available():
                allocated_backward, _ = get_gpu_memory_usage()
                peak_backward = torch.cuda.max_memory_allocated() / (1024**3)
                
                print(f"   Batch {batch_size:2d}: 前向 {peak_forward:.2f}GB, 反向 {peak_backward:.2f}GB")
            else:
                print(f"   Batch {batch_size:2d}: CPU模式，无GPU内存统计")
            
            # 清理
            del gene_expression, output, loss
            stage1_model.zero_grad()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   Batch {batch_size:2d}: ❌ GPU内存不足")
                break
            else:
                raise
    
    return stage1_params, components_params


def analyze_stage2_memory():
    """分析Stage 2 (VAR Transformer) 的内存占用"""
    print("\n🔍 Stage 2 (VAR Transformer) 内存分析")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # 首先创建并保存Stage 1模型
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "stage1_for_memory_test.ckpt")
        
        stage1_model = TwoStageVARST(num_genes=200, current_stage=1, device=device)
        stage1_model = stage1_model.to(device)
        stage1_model.save_stage_checkpoint(ckpt_path, stage=1)
        
        # 创建Stage 2模型
        stage2_model = TwoStageVARST(
            num_genes=200,
            histology_feature_dim=1024,
            spatial_coord_dim=2,
            current_stage=2,
            stage1_ckpt_path=ckpt_path,
            device=device
        )
        stage2_model = stage2_model.to(device)
        
        # 分析模型参数
        stage2_params = get_model_parameters(stage2_model)
        vqvae_params = get_model_parameters(stage2_model.stage1_vqvae)
        var_params = get_model_parameters(stage2_model.stage2_var)
        condition_params = get_model_parameters(stage2_model.condition_processor)
        
        print(f"📊 Stage 2 模型参数:")
        print(f"   完整模型参数: {stage2_params['total']:,}")
        print(f"   VQVAE参数 (冻结): {vqvae_params['total']:,}")
        print(f"   VAR Transformer参数: {var_params['total']:,}")
        print(f"   条件处理器参数: {condition_params['total']:,}")
        print(f"   可训练参数: {stage2_params['trainable']:,}")
        print(f"   冻结参数: {stage2_params['frozen']:,}")
        
        # 分析VAR Transformer组件参数
        var_components = {}
        var_components['token_embedding'] = get_model_parameters(stage2_model.stage2_var.token_embedding)['total']
        var_components['transformer_decoder'] = get_model_parameters(stage2_model.stage2_var.transformer_decoder)['total']
        var_components['output_projection'] = get_model_parameters(stage2_model.stage2_var.output_projection)['total']
        var_components['condition_projection'] = get_model_parameters(stage2_model.stage2_var.condition_projection)['total']
        
        print(f"\n🔧 VAR Transformer组件参数分布:")
        for component, params in var_components.items():
            percentage = (params / var_params['total']) * 100
            print(f"   {component}: {params:,} ({percentage:.1f}%)")
        
        if torch.cuda.is_available():
            allocated_after_model, reserved_after_model = get_gpu_memory_usage()
            print(f"\n💾 模型加载后GPU内存:")
            print(f"   已分配: {allocated_after_model:.2f} GB")
            print(f"   已保留: {reserved_after_model:.2f} GB")
        
        # 模拟训练
        batch_sizes = [4, 8, 16, 32]  # Stage 2通常需要更小的batch size
        print(f"\n🏋️ Stage 2 训练内存占用 (不同batch size):")
        
        for batch_size in batch_sizes:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                # 创建训练数据
                gene_expression = torch.randn(batch_size, 200, device=device)
                histology_features = torch.randn(batch_size, 1024, device=device)
                spatial_coords = torch.randn(batch_size, 2, device=device)
                
                # 前向传播
                stage2_model.train()
                output = stage2_model(
                    gene_expression=gene_expression,
                    histology_features=histology_features,
                    spatial_coords=spatial_coords
                )
                loss = output['loss']
                
                if torch.cuda.is_available():
                    allocated_forward, _ = get_gpu_memory_usage()
                    peak_forward = torch.cuda.max_memory_allocated() / (1024**3)
                
                # 反向传播
                loss.backward()
                
                if torch.cuda.is_available():
                    allocated_backward, _ = get_gpu_memory_usage()
                    peak_backward = torch.cuda.max_memory_allocated() / (1024**3)
                    
                    print(f"   Batch {batch_size:2d}: 前向 {peak_forward:.2f}GB, 反向 {peak_backward:.2f}GB")
                else:
                    print(f"   Batch {batch_size:2d}: CPU模式，无GPU内存统计")
                
                # 清理
                del gene_expression, histology_features, spatial_coords, output, loss
                stage2_model.zero_grad()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"   Batch {batch_size:2d}: ❌ GPU内存不足")
                    break
                else:
                    raise
        
        return stage2_params, var_components


def compare_memory_usage():
    """对比Stage 1和Stage 2的内存使用情况"""
    print("\n" + "=" * 70)
    print("📊 内存占用对比分析")
    print("=" * 70)
    
    # 分析Stage 1
    stage1_params, stage1_components = analyze_stage1_memory()
    
    # 分析Stage 2  
    stage2_params, stage2_components = analyze_stage2_memory()
    
    # 对比分析
    print("\n📈 内存占用对比:")
    print("=" * 50)
    
    print(f"🔸 模型参数对比:")
    print(f"   Stage 1总参数: {stage1_params['total']:,}")
    print(f"   Stage 2总参数: {stage2_params['total']:,}")
    param_ratio = stage2_params['total'] / stage1_params['total']
    print(f"   Stage 2 / Stage 1: {param_ratio:.2f}x")
    
    print(f"\n🔸 可训练参数对比:")
    print(f"   Stage 1可训练: {stage1_params['trainable']:,}")
    print(f"   Stage 2可训练: {stage2_params['trainable']:,}")
    trainable_ratio = stage2_params['trainable'] / stage1_params['trainable']
    print(f"   Stage 2 / Stage 1: {trainable_ratio:.2f}x")
    
    print(f"\n🔸 关键观察:")
    print(f"   • Stage 2包含完整的Stage 1模型(冻结)")
    print(f"   • Stage 2额外增加VAR Transformer: {stage2_params['total'] - stage1_params['total']:,} 参数")
    print(f"   • VAR Transformer参数占Stage 2总参数的 {((stage2_params['total'] - stage1_params['total']) / stage2_params['total'] * 100):.1f}%")
    
    # 内存使用建议
    print(f"\n💡 内存使用建议:")
    print(f"   🔹 Stage 1训练:")
    print(f"      - 推荐batch size: 16-64")
    print(f"      - 相对内存友好，主要是VQVAE计算")
    print(f"      - 适合较大的batch size进行稳定训练")
    
    print(f"   🔹 Stage 2训练:")
    print(f"      - 推荐batch size: 4-16")
    print(f"      - 内存需求更高 ({param_ratio:.1f}x参数量)")
    print(f"      - Transformer自注意力机制内存复杂度高")
    print(f"      - 需要载入冻结的Stage 1模型")
    
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n🖥️  当前GPU内存: {total_memory:.1f} GB")
        
        if total_memory < 16:
            print("   ⚠️  GPU内存较小，建议：")
            print("      - Stage 1: batch_size ≤ 32")
            print("      - Stage 2: batch_size ≤ 8")
        elif total_memory < 24:
            print("   ✅ GPU内存中等，建议：")
            print("      - Stage 1: batch_size ≤ 64") 
            print("      - Stage 2: batch_size ≤ 16")
        else:
            print("   🚀 GPU内存充足，建议：")
            print("      - Stage 1: batch_size ≤ 128")
            print("      - Stage 2: batch_size ≤ 32")


def main():
    """主函数"""
    print("🔬 Two-Stage VAR-ST 内存占用分析")
    print("=" * 70)
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🖥️  GPU: {gpu_name}")
        print(f"💾 GPU内存: {gpu_memory:.1f} GB")
    else:
        print("⚠️  未检测到GPU，将使用CPU进行分析")
    
    try:
        compare_memory_usage()
        
        print("\n" + "=" * 70)
        print("✅ 内存分析完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 