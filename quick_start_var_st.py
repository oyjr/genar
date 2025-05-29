#!/usr/bin/env python3
"""
VAR-ST 快速开始脚本

这个脚本演示如何使用现有的 MFBP 框架训练 VAR-ST 模型。
不需要额外的配置文件，所有参数都通过命令行传递。

用法:
    # 单GPU训练
    python quick_start_var_st.py
    
    # 多GPU训练
    python quick_start_var_st.py --gpus 4
    
    # 自定义参数
    python quick_start_var_st.py --gpus 2 --epochs 50 --batch-size 4

作者: VAR-ST 团队
"""

import subprocess
import sys
import argparse


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='VAR-ST 快速开始训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基本用法 - PRAD数据集，单GPU
    python quick_start_var_st.py
    
    # 多GPU训练
    python quick_start_var_st.py --gpus 4
    
    # 自定义参数
    python quick_start_var_st.py --gpus 2 --epochs 50 --batch-size 4 --lr 2e-4
    
    # 使用her2st数据集
    python quick_start_var_st.py --dataset her2st --gpus 1
        """
    )
    
    # 基本参数
    parser.add_argument('--dataset', type=str, default='PRAD', choices=['PRAD', 'her2st'],
                        help='数据集名称 (默认: PRAD)')
    parser.add_argument('--gpus', type=int, default=1,
                        help='GPU数量 (默认: 1)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (默认: 100)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批次大小 (默认: 8，VAR-ST推荐小批量)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率 (默认: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='权重衰减 (默认: 0.01)')
    
    # 解析参数
    args = parser.parse_args()
    
    print("🚀 VAR-ST 空间转录组学模型训练")
    print("=" * 50)
    print(f"📊 数据集: {args.dataset}")
    print(f"💻 GPU数量: {args.gpus}")
    print(f"🔄 训练轮数: {args.epochs}")
    print(f"📦 批次大小: {args.batch_size}")
    print(f"📈 学习率: {args.lr}")
    print(f"🏋️  权重衰减: {args.weight_decay}")
    print("=" * 50)
    
    # 构建训练命令
    cmd = [
        'python', 'src/main.py',
        '--dataset', args.dataset,
        '--model', 'VAR_ST',  # 使用VAR_ST模型
        '--gpus', str(args.gpus),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--weight-decay', str(args.weight_decay)
    ]
    
    # 多GPU时自动启用同步BatchNorm
    if args.gpus > 1:
        cmd.append('--sync-batchnorm')
        print(f"✅ 多GPU训练，已启用同步BatchNorm")
    
    print(f"🔧 执行命令:")
    print(f"   {' '.join(cmd)}")
    print("=" * 50)
    print()
    
    try:
        # 执行训练命令
        result = subprocess.run(cmd, check=True)
        print("\n✅ 训练完成!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败，错误代码: {e.returncode}")
        print("请检查上面的错误信息")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main() 