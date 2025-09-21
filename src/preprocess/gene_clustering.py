"""
Gene clustering and reordering based on expression similarity

This module implements gene clustering to reorder genes based on their 
spatial expression patterns for better biological coherence in VAR models.

Author: Assistant
Date: 2024
"""

import os
import json
import numpy as np
import shutil
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple
import logging

from .utils import (
    load_gene_list, save_gene_list, get_train_slides, 
    load_slide_gene_expression
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneClusteringProcessor:
    """基因聚类处理器"""
    
    def __init__(self, scale_dims: Tuple[int, ...] = (1, 4, 8, 40, 100, 200)):
        """
        Args:
            scale_dims: VAR模型的多尺度配置
        """
        self.scale_dims = scale_dims
        
        # 数据集配置
        self.datasets = {
            'PRAD': {
                'path': '/data/ouyangjiarui/stem/hest1k_datasets/PRAD/',
                'val_slides': 'MEND145'
            },
            'her2st': {
                'path': '/data/ouyangjiarui/stem/hest1k_datasets/her2st/',
                'val_slides': 'SPA148'
            },
            'kidney': {
                'path': '/data/ouyangjiarui/stem/hest1k_datasets/kidney/',
                'val_slides': 'NCBI697'
            },
            'mouse_brain': {
                'path': '/data/ouyangjiarui/stem/hest1k_datasets/mouse_brain/',
                'val_slides': 'NCBI667'
            },
            'ccRCC': {
                'path': '/data/ouyangjiarui/stem/hest1k_datasets/ccRCC/',
                'val_slides': 'INT2'
            }
        }
    
    def process_dataset(self, dataset_name: str) -> None:
        """处理单个数据集的基因聚类"""
        
        if dataset_name not in self.datasets:
            raise ValueError(f"不支持的数据集: {dataset_name}")
        
        dataset_config = self.datasets[dataset_name]
        data_path = os.path.join(dataset_config['path'], 'processed_data')
        val_slide = dataset_config['val_slides']
        
        logger.info(f"🚀 开始处理数据集: {dataset_name}")
        logger.info(f"   数据路径: {data_path}")
        logger.info(f"   验证集slide: {val_slide}")
        
        # 1. 备份原始基因列表
        self._backup_original_gene_list(data_path)
        
        # 2. 获取训练集slides
        train_slides = get_train_slides(data_path, val_slide)
        
        # 3. 加载训练集基因表达数据
        combined_expr = self._load_training_data(data_path, train_slides)
        logger.info(f"📊 训练集总spots: {combined_expr.shape[0]}")
        
        # 4. 执行基因聚类
        clustered_order = self._perform_clustering(combined_expr)
        
        # 5. 重新排列并保存基因列表
        self._save_clustered_gene_list(data_path, clustered_order)
        
        # 6. 保存聚类信息
        self._save_clustering_info(data_path, dataset_name, train_slides, 
                                  combined_expr.shape[0], clustered_order)
        
        logger.info(f"✅ {dataset_name} 基因聚类完成")
    
    def _backup_original_gene_list(self, data_path: str) -> None:
        """备份原始基因列表"""
        original_file = os.path.join(data_path, 'selected_gene_list.txt')
        backup_file = os.path.join(data_path, 'unclustered_selected_gene_list.txt')
        
        if not os.path.exists(backup_file):
            shutil.copy(original_file, backup_file)
            logger.info(f"💾 备份原始基因列表: {backup_file}")
        else:
            logger.info(f"📁 备份文件已存在: {backup_file}")
    
    def _load_training_data(self, data_path: str, train_slides: List[str]) -> np.ndarray:
        """加载训练集的基因表达数据"""
        logger.info(f"📥 加载{len(train_slides)}个训练slides的数据...")
        
        all_expr_data = []
        
        for slide_id in train_slides:
            try:
                slide_expr = load_slide_gene_expression(data_path, slide_id)
                logger.info(f"   {slide_id}: {slide_expr.shape[0]} spots, {slide_expr.shape[1]} genes")
                
                # 只取前200个基因
                if slide_expr.shape[1] >= 200:
                    slide_expr = slide_expr[:, :200]
                else:
                    logger.warning(f"⚠️  {slide_id}只有{slide_expr.shape[1]}个基因，少于200个")
                    continue
                
                all_expr_data.append(slide_expr)
                
            except Exception as e:
                logger.error(f"❌ 加载{slide_id}失败: {e}")
                continue
        
        if not all_expr_data:
            raise ValueError("没有成功加载任何训练数据")
        
        # 合并所有训练数据
        combined_expr = np.concatenate(all_expr_data, axis=0)
        logger.info(f"📊 合并训练数据: {combined_expr.shape}")
        
        return combined_expr
    
    def _perform_clustering(self, gene_expr_matrix: np.ndarray) -> np.ndarray:
        """执行基因聚类"""
        logger.info("🧬 开始基因聚类...")
        
        # 构建基因特征矩阵: [n_genes, n_spots]
        gene_features = gene_expr_matrix.T
        logger.info(f"   基因特征矩阵: {gene_features.shape}")
        
        # 标准化处理（每个基因标准化）
        scaler = StandardScaler()
        gene_features_norm = scaler.fit_transform(gene_features)
        logger.info(f"   标准化完成")
        
        # 第一阶段：聚成4个大群组
        logger.info("   阶段1: 聚类为4个大群组...")
        kmeans_4 = KMeans(n_clusters=4, random_state=42, n_init=10)
        major_clusters = kmeans_4.fit_predict(gene_features_norm)
        
        cluster_sizes = np.bincount(major_clusters)
        logger.info(f"   大群组大小: {cluster_sizes}")
        
        # 第二阶段：每个大群组内部细分
        logger.info("   阶段2: 大群组内部细分...")
        clustered_order = []
        
        for major_group in range(4):
            genes_in_major = np.where(major_clusters == major_group)[0]
            group_features = gene_features_norm[genes_in_major]
            
            if len(genes_in_major) <= 10:
                # 群组太小，直接添加
                clustered_order.extend(genes_in_major.tolist())
                logger.info(f"     群组{major_group}: {len(genes_in_major)}个基因（无细分）")
            else:
                # 群组内部再细分
                n_sub_clusters = max(2, len(genes_in_major) // 12)
                kmeans_sub = KMeans(n_clusters=n_sub_clusters, random_state=42)
                sub_clusters = kmeans_sub.fit_predict(group_features)
                
                # 按子群组排序添加
                for sub_group in range(n_sub_clusters):
                    genes_in_sub = genes_in_major[sub_clusters == sub_group]
                    clustered_order.extend(genes_in_sub.tolist())
                
                logger.info(f"     群组{major_group}: {len(genes_in_major)}个基因 → {n_sub_clusters}个子群")
        
        clustered_order = np.array(clustered_order)
        logger.info(f"✅ 聚类完成，重排序: {len(clustered_order)}个基因")
        
        return clustered_order
    
    def _save_clustered_gene_list(self, data_path: str, clustered_order: np.ndarray) -> None:
        """保存聚类后的基因列表"""
        # 加载原始基因列表
        backup_file = os.path.join(data_path, 'unclustered_selected_gene_list.txt')
        original_genes = load_gene_list(backup_file)
        
        # 重新排列
        clustered_genes = [original_genes[i] for i in clustered_order]
        
        # 保存到原文件位置
        output_file = os.path.join(data_path, 'selected_gene_list.txt')
        save_gene_list(output_file, clustered_genes)
        
        logger.info(f"💾 保存聚类后基因列表: {output_file}")
    
    def _save_clustering_info(self, data_path: str, dataset_name: str, 
                            train_slides: List[str], total_spots: int, 
                            clustered_order: np.ndarray) -> None:
        """保存聚类详细信息"""
        clustering_info = {
            'dataset': dataset_name,
            'train_slides': train_slides,
            'total_spots': total_spots,
            'clustered_order': clustered_order.tolist(),
            'scale_dims': self.scale_dims,
            'timestamp': datetime.now().isoformat(),
            'algorithm': 'kmeans_hierarchical',
            'parameters': {
                'stage1_clusters': 4,
                'normalization': 'zscore',
                'random_state': 42
            }
        }
        
        info_file = os.path.join(data_path, 'clustering_info.json')
        with open(info_file, 'w') as f:
            json.dump(clustering_info, f, indent=2)
        
        logger.info(f"📝 保存聚类信息: {info_file}")
    
    def process_all_datasets(self) -> None:
        """处理所有数据集"""
        for dataset_name in self.datasets.keys():
            try:
                self.process_dataset(dataset_name)
            except Exception as e:
                logger.error(f"❌ 处理{dataset_name}失败: {e}")
                continue