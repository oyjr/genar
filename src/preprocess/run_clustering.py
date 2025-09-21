#!/usr/bin/env python3
"""
基因聚类预处理主运行脚本

基于训练集数据对基因进行聚类重排序，生成适合VAR模型的基因顺序

使用方法:
    # 处理所有数据集
    python src/preprocess/run_clustering.py --all-datasets
    
    # 处理单个数据集
    python src/preprocess/run_clustering.py --dataset PRAD

Author: Assistant
Date: 2024
"""

import argparse
import sys
import os
import logging

# 添加src路径到Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from preprocess.gene_clustering import GeneClusteringProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='基因聚类预处理 - 基于表达相似性重排序基因',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理所有数据集
  python src/preprocess/run_clustering.py --all-datasets
  
  # 处理单个数据集
  python src/preprocess/run_clustering.py --dataset PRAD
  python src/preprocess/run_clustering.py --dataset her2st
  
注意:
  - 会自动备份原始基因列表为 unclustered_selected_gene_list.txt
  - 生成的新基因列表会覆盖原来的 selected_gene_list.txt
  - 聚类信息保存在 clustering_info.json 中
        """
    )
    
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['PRAD', 'her2st', 'kidney', 'mouse_brain', 'ccRCC'],
        help='指定要处理的数据集'
    )
    
    parser.add_argument(
        '--all-datasets', 
        action='store_true',
        help='处理所有数据集'
    )
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all_datasets:
        parser.print_help()
        print("\n❌ 错误: 请指定 --dataset 或 --all-datasets")
        return 1
    
    # 创建处理器
    processor = GeneClusteringProcessor()
    
    try:
        if args.all_datasets:
            logger.info("🚀 开始处理所有数据集...")
            processor.process_all_datasets()
            logger.info("✅ 所有数据集处理完成!")
            
        elif args.dataset:
            logger.info(f"🚀 开始处理数据集: {args.dataset}")
            processor.process_dataset(args.dataset)
            logger.info(f"✅ {args.dataset} 处理完成!")
            
    except Exception as e:
        logger.error(f"❌ 处理失败: {e}")
        return 1
    
    print("\n🎉 基因聚类预处理完成!")
    print("📋 接下来可以:")
    print("   1. 检查生成的 clustering_info.json 文件")
    print("   2. 直接使用现有训练脚本，基因顺序已更新")
    print("   3. 如需回退，可将 unclustered_selected_gene_list.txt 重命名回去")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)