"""
Two-Stage VAR-ST Integration Test
完整测试两阶段VAR-ST模型的训练和推理流程

修复了之前测试中的关键问题：
1. 确保Stage 2正确加载Stage 1 checkpoint
2. 验证training_stage正确映射到current_stage 
3. 测试错误处理和边界条件
"""

import sys
import os
sys.path.insert(0, 'src')

import torch
import tempfile
import pytest
from pathlib import Path

from model.VAR.two_stage_var_st import TwoStageVARST
from model.model_interface import ModelInterface
from addict import Dict


def test_stage_parameter_mapping():
    """测试training_stage到current_stage的参数映射"""
    print("🧪 测试参数映射...")
    
    # 模拟配置 - Stage 1
    config_stage1 = Dict({
        'MODEL': {
            'model_name': 'TWO_STAGE_VAR_ST',
            'training_stage': 1,  # 注意：这里是training_stage
            'stage1_ckpt_path': None,
            'num_genes': 200,
            'histology_feature_dim': 1024,
            'spatial_coord_dim': 2,
        }
    })
    
    # 测试Stage 1参数映射
    try:
        model_interface = ModelInterface(config_stage1)
        model = model_interface.model
        
        # 验证参数映射正确
        assert model.current_stage == 1, f"期望current_stage=1，实际得到{model.current_stage}"
        print("✅ Stage 1参数映射正确")
    except Exception as e:
        print(f"❌ Stage 1参数映射失败: {e}")
        raise
    
    # 模拟配置 - Stage 2 (但没有checkpoint，应该报错)
    config_stage2_no_ckpt = Dict({
        'MODEL': {
            'model_name': 'TWO_STAGE_VAR_ST',
            'training_stage': 2,
            'stage1_ckpt_path': None,  # 没有checkpoint
            'num_genes': 200,
            'histology_feature_dim': 1024,
            'spatial_coord_dim': 2,
        }
    })
    
    # 测试Stage 2没有checkpoint时的错误处理
    print("🧪 测试Stage 2缺少checkpoint的错误处理...")
    try:
        model_interface = ModelInterface(config_stage2_no_ckpt)
        print("❌ 应该报错但没有报错!")
        assert False, "Stage 2没有checkpoint应该报错"
    except ValueError as e:
        if "stage1_ckpt_path is required" in str(e) or "Two-stage VAR-ST配置错误" in str(e):
            print("✅ Stage 2缺少checkpoint正确报错")
        else:
            print(f"❌ 错误信息不正确: {e}")
            raise
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        raise


def test_stage1_training():
    """测试Stage 1 VQVAE训练"""
    print("\n🧪 测试Stage 1训练...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4
    num_genes = 200
    
    # 创建Stage 1模型
    model = TwoStageVARST(
        num_genes=num_genes,
        current_stage=1,  # 直接使用current_stage参数
        device=device
    )
    
    model = model.to(device)
    
    # 模拟基因表达数据
    gene_expression = torch.randn(batch_size, num_genes, device=device)
    
    # Stage 1前向传播
    model.train()
    output = model(gene_expression)
    
    # 验证输出
    assert 'loss' in output, "输出应包含loss"
    assert 'reconstructed' in output, "输出应包含reconstructed"
    assert 'tokens' in output, "输出应包含tokens"
    assert 'stage1_losses' in output, "输出应包含stage1_losses"
    
    # 验证loss可以反向传播
    loss = output['loss']
    loss.backward()
    
    print(f"✅ Stage 1训练测试通过")
    print(f"   - Loss: {loss.item():.4f}")
    print(f"   - Reconstructed shape: {output['reconstructed'].shape}")
    
    return model


def test_stage1_checkpoint_saving_loading():
    """测试Stage 1 checkpoint保存和加载"""
    print("\n🧪 测试Stage 1 checkpoint保存和加载...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建并训练Stage 1模型
    model1 = TwoStageVARST(
        num_genes=200,
        current_stage=1,
        device=device
    )
    model1 = model1.to(device)
    
    # 保存checkpoint
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "stage1_test.ckpt")
        model1.save_stage_checkpoint(ckpt_path, stage=1)
        
        assert os.path.exists(ckpt_path), "Checkpoint文件应该被创建"
        print(f"✅ Stage 1 checkpoint保存成功: {ckpt_path}")
        
        # 测试加载checkpoint
        model2 = TwoStageVARST(
            num_genes=200,
            current_stage=2,  # Stage 2模式
            stage1_ckpt_path=ckpt_path,  # 加载Stage 1
            device=device
        )
        model2 = model2.to(device)
        
        print("✅ Stage 1 checkpoint加载成功")
        
        # 验证VQVAE权重一致
        model1_state = model1.stage1_vqvae.state_dict()
        model2_state = model2.stage1_vqvae.state_dict()
        
        for key in model1_state:
            if key in model2_state:
                diff = torch.norm(model1_state[key] - model2_state[key]).item()
                assert diff < 1e-6, f"权重不一致: {key}, diff={diff}"
        
        print("✅ Stage 1权重加载验证通过")
        
        return ckpt_path, model2


def test_stage2_training():
    """测试Stage 2 VAR Transformer训练"""
    print("\n🧪 测试Stage 2训练...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2  # Stage 2内存需求更大
    num_genes = 200
    histology_dim = 1024
    spatial_dim = 2
    
    # 首先创建并保存Stage 1模型
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "stage1_for_stage2.ckpt")
        
        # 创建Stage 1模型并保存
        stage1_model = TwoStageVARST(
            num_genes=num_genes,
            current_stage=1,
            device=device
        )
        stage1_model = stage1_model.to(device)
        stage1_model.save_stage_checkpoint(ckpt_path, stage=1)
        
        print(f"✅ Stage 1模型保存完成: {ckpt_path}")
        
        # 创建Stage 2模型
        stage2_model = TwoStageVARST(
            num_genes=num_genes,
            histology_feature_dim=histology_dim,
            spatial_coord_dim=spatial_dim,
            current_stage=2,
            stage1_ckpt_path=ckpt_path,
            device=device
        )
        stage2_model = stage2_model.to(device)
        
        print("✅ Stage 2模型创建成功")
        
        # 验证Stage 1被冻结，Stage 2可训练
        vqvae_trainable = any(p.requires_grad for p in stage2_model.stage1_vqvae.parameters())
        var_trainable = any(p.requires_grad for p in stage2_model.stage2_var.parameters())
        condition_trainable = any(p.requires_grad for p in stage2_model.condition_processor.parameters())
        
        assert not vqvae_trainable, "Stage 1 VQVAE应该被冻结"
        assert var_trainable, "Stage 2 VAR应该可训练"
        assert condition_trainable, "Condition processor应该可训练"
        
        print("✅ Stage 2参数冻结/解冻状态正确")
        
        # 模拟训练数据
        gene_expression = torch.randn(batch_size, num_genes, device=device)
        histology_features = torch.randn(batch_size, histology_dim, device=device)
        spatial_coords = torch.randn(batch_size, spatial_dim, device=device)
        
        # Stage 2前向传播
        stage2_model.train()
        output = stage2_model(
            gene_expression=gene_expression,
            histology_features=histology_features,
            spatial_coords=spatial_coords
        )
        
        # 验证输出
        assert 'loss' in output, "输出应包含loss"
        assert 'logits' in output, "输出应包含logits"
        assert 'stage2_losses' in output, "输出应包含stage2_losses"
        
        # 验证loss可以反向传播
        loss = output['loss']
        loss.backward()
        
        print(f"✅ Stage 2训练测试通过")
        print(f"   - Loss: {loss.item():.4f}")
        print(f"   - Logits shape: {output['logits'].shape}")


def test_end_to_end_inference():
    """测试端到端推理流程"""
    print("\n🧪 测试端到端推理...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    num_genes = 200
    histology_dim = 1024
    spatial_dim = 2
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "stage1_for_inference.ckpt")
        
        # 创建并保存Stage 1模型
        stage1_model = TwoStageVARST(
            num_genes=num_genes,
            current_stage=1,
            device=device
        )
        stage1_model = stage1_model.to(device)
        stage1_model.save_stage_checkpoint(ckpt_path, stage=1)
        
        # 创建完整模型用于推理
        model = TwoStageVARST(
            num_genes=num_genes,
            histology_feature_dim=histology_dim,
            spatial_coord_dim=spatial_dim,
            current_stage=2,
            stage1_ckpt_path=ckpt_path,
            device=device
        )
        model = model.to(device)
        model.eval()
        
        # 模拟推理输入
        histology_features = torch.randn(batch_size, histology_dim, device=device)
        spatial_coords = torch.randn(batch_size, spatial_dim, device=device)
        
        # 推理
        with torch.no_grad():
            results = model.inference(
                histology_features=histology_features,
                spatial_coords=spatial_coords,
                temperature=1.0,
                top_k=50,
                top_p=0.9
            )
        
        # 验证推理输出
        assert 'predicted_gene_expression' in results, "应包含预测基因表达"
        assert 'generated_tokens' in results, "应包含生成的tokens"
        assert 'multi_scale_tokens' in results, "应包含多尺度tokens"
        
        predicted_genes = results['predicted_gene_expression']
        assert predicted_genes.shape == (batch_size, num_genes), f"预测基因形状错误: {predicted_genes.shape}"
        
        print(f"✅ 端到端推理测试通过")
        print(f"   - 预测基因表达形状: {predicted_genes.shape}")
        print(f"   - 生成tokens形状: {results['generated_tokens'].shape}")


def test_model_interface_integration():
    """测试ModelInterface集成"""
    print("\n🧪 测试ModelInterface集成...")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "stage1_for_interface.ckpt")
        
        # 首先创建Stage 1 checkpoint
        stage1_model = TwoStageVARST(num_genes=200, current_stage=1)
        stage1_model.save_stage_checkpoint(ckpt_path, stage=1)
        
        # 测试Stage 1 ModelInterface
        config_stage1 = Dict({
            'MODEL': {
                'model_name': 'TWO_STAGE_VAR_ST',
                'training_stage': 1,
                'stage1_ckpt_path': None,
                'num_genes': 200,
                'histology_feature_dim': 1024,
                'spatial_coord_dim': 2,
            }
        })
        
        try:
            interface1 = ModelInterface(config_stage1)
            assert interface1.model.current_stage == 1
            print("✅ Stage 1 ModelInterface测试通过")
        except Exception as e:
            print(f"❌ Stage 1 ModelInterface测试失败: {e}")
            raise
        
        # 测试Stage 2 ModelInterface
        config_stage2 = Dict({
            'MODEL': {
                'model_name': 'TWO_STAGE_VAR_ST',
                'training_stage': 2,
                'stage1_ckpt_path': ckpt_path,
                'num_genes': 200,
                'histology_feature_dim': 1024,
                'spatial_coord_dim': 2,
            }
        })
        
        try:
            interface2 = ModelInterface(config_stage2)
            assert interface2.model.current_stage == 2
            print("✅ Stage 2 ModelInterface测试通过")
        except Exception as e:
            print(f"❌ Stage 2 ModelInterface测试失败: {e}")
            raise


def test_error_conditions():
    """测试错误条件和边界情况"""
    print("\n🧪 测试错误条件...")
    
    # 测试不存在的checkpoint路径
    try:
        model = TwoStageVARST(
            current_stage=2,
            stage1_ckpt_path="/nonexistent/path.ckpt"
        )
        print("❌ 应该因为checkpoint不存在而报错")
        assert False, "应该报错但没有"
    except FileNotFoundError:
        print("✅ 不存在的checkpoint路径正确报错")
    except Exception as e:
        print(f"❌ 意外错误类型: {e}")
        raise
    
    # 测试Stage 2缺少checkpoint
    try:
        model = TwoStageVARST(current_stage=2, stage1_ckpt_path=None)
        print("❌ 应该因为Stage 2缺少checkpoint而报错")
        assert False, "应该报错但没有"
    except ValueError as e:
        if "stage1_ckpt_path is required" in str(e):
            print("✅ Stage 2缺少checkpoint正确报错")
        else:
            print(f"❌ 错误信息不正确: {e}")
            raise
    except Exception as e:
        print(f"❌ 意外错误类型: {e}")
        raise
    
    # 测试无效的stage值
    try:
        model = TwoStageVARST(current_stage=3)
        print("❌ 应该因为无效stage值而报错")
        assert False, "应该报错但没有"
    except ValueError as e:
        if "Invalid stage" in str(e):
            print("✅ 无效stage值正确报错")
        else:
            print(f"❌ 错误信息不正确: {e}")
            raise
    except Exception as e:
        print(f"❌ 意外错误类型: {e}")
        raise


def test_stage2_metrics_skipping():
    """测试Stage 2训练时正确跳过基因表达指标计算"""
    print("\n🧪 测试Stage 2指标跳过...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "stage1_for_metrics_test.ckpt")
        
        # 创建Stage 1模型并保存
        stage1_model = TwoStageVARST(num_genes=200, current_stage=1, device=device)
        stage1_model = stage1_model.to(device)
        stage1_model.save_stage_checkpoint(ckpt_path, stage=1)
        
        # 创建Stage 2 ModelInterface
        config_stage2 = Dict({
            'MODEL': {
                'model_name': 'TWO_STAGE_VAR_ST',
                'training_stage': 2,
                'stage1_ckpt_path': ckpt_path,
                'num_genes': 200,
                'histology_feature_dim': 1024,
                'spatial_coord_dim': 2,
            }
        })
        
        interface = ModelInterface(config_stage2)
        interface = interface.to(device)
        
        # 验证Stage 2配置
        assert interface.model.current_stage == 2, "应该是Stage 2"
        
        # 🔧 修复：使用正确的输入格式
        batch = {
            'target_genes': torch.randn(4, 200, device=device),
            'img': torch.randn(4, 1024, device=device),
            'positions': torch.randn(4, 2, device=device),
        }
        
        # 预处理输入
        processed_batch = interface._preprocess_inputs(batch)
        
        # 验证预处理结果
        assert 'gene_expression' in processed_batch, "应该包含gene_expression"
        assert 'histology_features' in processed_batch, "应该包含histology_features"
        assert 'spatial_coords' in processed_batch, "应该包含spatial_coords"
        
        # 模型前向传播
        results_dict = interface.model(**processed_batch)
        
        # 验证输出格式
        assert 'loss' in results_dict, "应该包含VAR损失"
        assert 'logits' in results_dict, "应该包含VAR logits"
        assert results_dict['logits'].shape == (4, 241, 4096), f"VAR logits形状错误: {results_dict['logits'].shape}"
        
        # 测试指标提取 - 应该返回dummy数据
        logits, target_genes = interface._extract_predictions_and_targets(results_dict, batch)
        
        # 验证返回的是dummy数据（零张量）
        assert logits.shape == (4, 200), f"预期形状 [4, 200]，实际 {logits.shape}"
        assert target_genes.shape == (4, 200), f"预期形状 [4, 200]，实际 {target_genes.shape}"
        assert torch.allclose(logits, torch.zeros_like(logits)), "应该返回零张量作为dummy数据"
        
        print("✅ Stage 2指标跳过测试通过")
        print(f"   - VAR logits形状: {results_dict['logits'].shape}")
        print(f"   - Dummy预测形状: {logits.shape}")
        print(f"   - 是否为零张量: {torch.allclose(logits, torch.zeros_like(logits))}")


def test_stage2_end_to_end_inference():
    """测试Stage 2的端到端推理流程"""
    print("\n🧪 测试Stage 2端到端推理...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "stage1_for_e2e.ckpt")
        
        # 创建并保存Stage 1模型
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
        stage2_model.eval()
        
        # 模拟推理输入
        histology_features = torch.randn(2, 1024, device=device)
        spatial_coords = torch.randn(2, 2, device=device)
        
        # 端到端推理
        with torch.no_grad():
            results = stage2_model.inference(
                histology_features=histology_features,
                spatial_coords=spatial_coords,
                temperature=1.0,
                top_k=50,
                top_p=0.9
            )
        
        # 验证推理输出
        assert 'predicted_gene_expression' in results, "应包含预测基因表达"
        assert 'generated_tokens' in results, "应包含生成的tokens"
        assert 'multi_scale_tokens' in results, "应包含多尺度tokens"
        
        predicted_genes = results['predicted_gene_expression']
        generated_tokens = results['generated_tokens']
        
        # 验证形状
        assert predicted_genes.shape == (2, 200), f"预测基因形状错误: {predicted_genes.shape}"
        assert generated_tokens.shape == (2, 241), f"生成tokens形状错误: {generated_tokens.shape}"
        
        # 验证多尺度tokens结构
        multi_scale_tokens = results['multi_scale_tokens']
        assert multi_scale_tokens['global'].shape == (2, 1), "全局tokens形状错误"
        assert multi_scale_tokens['pathway'].shape == (2, 8), "通路tokens形状错误"
        assert multi_scale_tokens['module'].shape == (2, 32), "模块tokens形状错误"
        assert multi_scale_tokens['individual'].shape == (2, 200), "个体tokens形状错误"
        
        print("✅ Stage 2端到端推理测试通过")
        print(f"   - 预测基因表达形状: {predicted_genes.shape}")
        print(f"   - 生成tokens形状: {generated_tokens.shape}")
        print(f"   - 多尺度tokens: global{multi_scale_tokens['global'].shape}, pathway{multi_scale_tokens['pathway'].shape}, module{multi_scale_tokens['module'].shape}, individual{multi_scale_tokens['individual'].shape}")


def run_all_tests():
    """运行所有测试"""
    print("🚀 开始两阶段VAR-ST集成测试")
    print("=" * 60)
    
    try:
        # 1. 参数映射测试
        test_stage_parameter_mapping()
        
        # 2. Stage 1训练测试
        test_stage1_training()
        
        # 3. Checkpoint保存加载测试
        test_stage1_checkpoint_saving_loading()
        
        # 4. Stage 2训练测试
        test_stage2_training()
        
        # 5. 端到端推理测试
        test_end_to_end_inference()
        
        # 6. ModelInterface集成测试
        test_model_interface_integration()
        
        # 7. 错误条件测试
        test_error_conditions()
        
        # 🔧 新增：Stage 2指标跳过测试
        test_stage2_metrics_skipping()
        
        # 🔧 新增：Stage 2端到端推理测试
        test_stage2_end_to_end_inference()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！两阶段VAR-ST实现正确")
        print("✅ Stage 1 VQVAE训练正常")
        print("✅ Stage 2 VAR Transformer训练正常")
        print("✅ Stage 2正确跳过基因表达指标计算")
        print("✅ 参数映射和错误处理正确")
        print("✅ 端到端推理流程正常")
        print("✅ ModelInterface集成正常")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests() 