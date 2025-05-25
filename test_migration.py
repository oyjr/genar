#!/usr/bin/env python3
"""
MFBP项目数据结构迁移测试脚本

测试从原始HEST格式到新hest1k_datasets格式的迁移是否成功
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 确保导入项目目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def test_imports():
    """测试所有必要的模块导入"""
    print("🔍 测试模块导入...")
    
    try:
        from dataset import STDataset, DataInterface
        print("✅ 数据集模块导入成功")
    except Exception as e:
        print(f"❌ 数据集模块导入失败: {e}")
        return False
    
    try:
        from model import ModelInterface
        print("✅ 模型接口导入成功")
    except Exception as e:
        print(f"❌ 模型接口导入失败: {e}")
        return False
    
    try:
        from model.MFBP.MFBP import MFBP
        print("✅ MFBP模型导入成功")
    except Exception as e:
        print(f"❌ MFBP模型导入失败: {e}")
        return False
    
    return True

def test_stdataset_init():
    """测试STDataset类的初始化"""
    print("\n🔍 测试STDataset初始化...")
    
    # 模拟新的数据格式
    test_data_path = "/tmp/test_hest1k_datasets/PRAD/"
    test_processed_dir = f"{test_data_path}processed_data"
    test_st_dir = f"{test_data_path}st"
    
    # 创建测试目录和文件
    os.makedirs(test_processed_dir, exist_ok=True)
    os.makedirs(test_st_dir, exist_ok=True)
    
    # 创建基因列表文件
    gene_list_file = f"{test_processed_dir}/selected_gene_list.txt"
    with open(gene_list_file, 'w') as f:
        for i in range(200):
            f.write(f"GENE_{i:03d}\n")
    
    # 创建slide列表文件
    slide_list_file = f"{test_processed_dir}/all_slide_lst.txt"
    with open(slide_list_file, 'w') as f:
        slides = ["SPA154", "SPA153", "SPA152", "SPA151", "SPA150"]
        for slide in slides:
            f.write(f"{slide}\n")
    
    try:
        from dataset import STDataset
        
        # 测试只验证模式的初始化（不预加载训练数据）
        dataset = STDataset(
            mode='val',  # 使用验证模式，不会预加载数据
            data_path=test_data_path,
            expr_name='PRAD',
            slide_val='SPA154,SPA153',
            slide_test='SPA152,SPA151',
            encoder_name='uni',
            use_augmented=False
        )
        
        print("✅ STDataset初始化成功")
        print(f"  - 基因数量: {len(dataset.genes)}")
        print(f"  - 训练集slides: {dataset.slide_splits['train']}")
        print(f"  - 验证集slides: {dataset.slide_splits['val']}")
        print(f"  - 测试集slides: {dataset.slide_splits['test']}")
        print(f"  - 当前模式数据集大小: {len(dataset)}")
        
        # 测试参数验证
        print("  - 测试参数验证...")
        assert dataset.mode == 'val'
        assert dataset.encoder_name == 'uni'
        assert not dataset.use_augmented
        assert len(dataset.genes) == 200
        
        return True
        
    except Exception as e:
        print(f"❌ STDataset初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mfbp_model():
    """测试MFBP模型"""
    print("\n🔍 测试MFBP模型...")
    
    try:
        from model.MFBP.MFBP import MFBP
        
        # 创建模型实例
        model = MFBP(num_genes=200, feature_dim=1024)
        
        # 测试前向传播
        # 训练模式：单个spot
        train_input = torch.randn(32, 1024)  # [batch_size, feature_dim]
        train_output = model(train_input)
        
        print("✅ MFBP模型创建成功")
        print(f"  - 训练模式输入形状: {train_input.shape}")
        print(f"  - 训练模式输出形状: {train_output['logits'].shape}")
        
        # 验证模式：多个spots
        val_input = torch.randn(1, 100, 1024)  # [1, num_spots, feature_dim]
        val_output = model(val_input)
        
        print(f"  - 验证模式输入形状: {val_input.shape}")
        print(f"  - 验证模式输出形状: {val_output['logits'].shape}")
        
        # 检查输出格式
        assert 'logits' in train_output, "输出应包含'logits'键"
        assert train_output['logits'].shape == (32, 1, 200), f"训练输出形状错误: {train_output['logits'].shape}"
        assert val_output['logits'].shape == (1, 100, 200), f"验证输出形状错误: {val_output['logits'].shape}"
        
        return True
        
    except Exception as e:
        print(f"❌ MFBP模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """测试配置文件加载"""
    print("\n🔍 测试配置文件加载...")
    
    try:
        from utils import load_config
        
        config_path = "config/hest/base_config.yaml"
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return False
        
        config = load_config(config_path)
        
        # 检查关键配置项
        assert hasattr(config, 'MODEL'), "配置应包含MODEL部分"
        assert hasattr(config, 'DATA'), "配置应包含DATA部分"
        assert hasattr(config, 'TRAINING'), "配置应包含TRAINING部分"
        
        print("✅ 配置文件加载成功")
        print(f"  - 模型名称: {config.MODEL.model_name}")
        print(f"  - 基因数量: {config.MODEL.num_genes}")
        print(f"  - 特征维度: {config.MODEL.feature_dim}")
        return True
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_command_line_args():
    """测试新的命令行参数格式"""
    print("\n🔍 测试命令行参数解析...")
    
    try:
        sys.path.append('src')
        from main import get_parse, validate_args
        
        parser = get_parse()
        
        # 模拟命令行参数
        test_args = [
            '--config', 'config/hest/base_config.yaml',
            '--expr_name', 'PRAD',
            '--data_path', '/tmp/test_data/',
            '--slide_val', 'SPA154,SPA153',
            '--slide_test', 'SPA152',
            '--encoder_name', 'uni',
            '--use_augmented',
            '--mode', 'train'
        ]
        
        # 创建测试目录
        os.makedirs('/tmp/test_data/st', exist_ok=True)
        os.makedirs('/tmp/test_data/processed_data', exist_ok=True)
        
        args = parser.parse_args(test_args)
        args = validate_args(args)
        
        print("✅ 命令行参数解析成功")
        print(f"  - 表达谱名称: {args.expr_name}")
        print(f"  - 数据路径: {args.data_path}")
        print(f"  - 验证集slides: {args.slide_val}")
        print(f"  - 测试集slides: {args.slide_test}")
        print(f"  - 编码器: {args.encoder_name}")
        print(f"  - 使用增强: {args.use_augmented}")
        return True
        
    except Exception as e:
        print(f"❌ 命令行参数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_filename_construction():
    """测试文件名构建逻辑"""
    print("\n🔍 测试文件名构建逻辑...")
    
    try:
        from dataset import STDataset
        
        # 测试标准嵌入文件名
        test_data_path = "/tmp/test_hest1k_datasets/PRAD/"
        test_processed_dir = f"{test_data_path}processed_data"
        
        # 创建测试目录和必要文件
        os.makedirs(test_processed_dir, exist_ok=True)
        os.makedirs(f"{test_data_path}st", exist_ok=True)
        
        # 创建基因列表文件
        with open(f"{test_processed_dir}/selected_gene_list.txt", 'w') as f:
            for i in range(10):
                f.write(f"GENE_{i:03d}\n")
        
        # 创建slide列表文件
        with open(f"{test_processed_dir}/all_slide_lst.txt", 'w') as f:
            f.write("TEST001\n")
        
        # 测试标准嵌入（uni）
        dataset_standard = STDataset(
            mode='val',
            data_path=test_data_path,
            expr_name='PRAD',
            slide_val='TEST001',
            slide_test='',
            encoder_name='uni',
            use_augmented=False
        )
        
        # 测试增强嵌入（uni_aug）
        dataset_augmented = STDataset(
            mode='val',
            data_path=test_data_path,
            expr_name='PRAD',
            slide_val='TEST001',
            slide_test='',
            encoder_name='uni',
            use_augmented=True
        )
        
        # 验证目录路径
        expected_standard_dir = f"{test_processed_dir}/1spot_uni_ebd"
        expected_augmented_dir = f"{test_processed_dir}/1spot_uni_ebd_aug"
        
        assert dataset_standard.emb_dir == expected_standard_dir, f"标准嵌入目录错误: {dataset_standard.emb_dir}"
        assert dataset_augmented.emb_dir == expected_augmented_dir, f"增强嵌入目录错误: {dataset_augmented.emb_dir}"
        
        print("✅ 文件名构建逻辑测试通过")
        print(f"  - 标准嵌入目录: {dataset_standard.emb_dir}")
        print(f"  - 增强嵌入目录: {dataset_augmented.emb_dir}")
        print(f"  - 标准文件名格式: TEST001_uni.pt")
        print(f"  - 增强文件名格式: TEST001_uni_aug.pt")
        
        # 测试CONCH编码器
        dataset_conch = STDataset(
            mode='val',
            data_path=test_data_path,
            expr_name='PRAD', 
            slide_val='TEST001',
            slide_test='',
            encoder_name='conch',
            use_augmented=True
        )
        
        expected_conch_dir = f"{test_processed_dir}/1spot_conch_ebd_aug"
        assert dataset_conch.emb_dir == expected_conch_dir, f"CONCH增强嵌入目录错误: {dataset_conch.emb_dir}"
        print(f"  - CONCH增强目录: {dataset_conch.emb_dir}")
        print(f"  - CONCH增强文件名: TEST001_conch_aug.pt")
        
        return True
        
    except Exception as e:
        print(f"❌ 文件名构建逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始MFBP项目数据结构迁移测试\n")
    
    test_results = []
    
    # 运行各个测试
    test_results.append(("模块导入", test_imports()))
    test_results.append(("STDataset初始化", test_stdataset_init()))
    test_results.append(("MFBP模型", test_mfbp_model()))
    test_results.append(("配置文件加载", test_config_loading()))
    test_results.append(("命令行参数", test_command_line_args()))
    test_results.append(("文件名构建逻辑", test_filename_construction()))
    
    # 汇总结果
    print("\n" + "="*50)
    print("📊 测试结果汇总:")
    print("="*50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print("="*50)
    print(f"总体结果: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！数据结构迁移成功！")
        print("\n📋 新命令行格式示例:")
        print("python src/main.py \\")
        print("    --config config/hest/base_config.yaml \\")
        print("    --expr_name PRAD \\")
        print("    --data_path /data/ouyangjiarui/stem/hest1k_datasets/PRAD/ \\")
        print("    --slide_val \"SPA154,SPA153\" \\")
        print("    --slide_test \"SPA152,SPA151\" \\")
        print("    --encoder_name uni \\")
        print("    --use_augmented \\")
        print("    --mode train")
        return True
    else:
        print("⚠️  部分测试失败，请检查上述错误信息。")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 