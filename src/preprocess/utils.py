"""
Simple utility functions for gene clustering preprocessing

Author: Assistant
Date: 2024
"""

import os
import numpy as np
import scanpy as sc
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_slide_list(slide_list_file: str) -> List[str]:
    """加载slide列表"""
    with open(slide_list_file, 'r') as f:
        slides = [line.strip() for line in f.readlines() if line.strip()]
    return slides


def load_gene_list(gene_list_file: str) -> List[str]:
    """加载基因列表"""
    with open(gene_list_file, 'r', encoding='utf-8') as f:
        genes = [line.strip() for line in f.readlines() if line.strip()]
    return genes


def save_gene_list(gene_list_file: str, genes: List[str]) -> None:
    """保存基因列表"""
    with open(gene_list_file, 'w', encoding='utf-8') as f:
        for gene in genes:
            f.write(f"{gene}\n")


def load_slide_gene_expression(data_path: str, slide_id: str) -> np.ndarray:
    """
    从h5ad文件加载基因表达数据
    
    Args:
        data_path: processed_data路径 (实际不使用，h5ad文件在统一路径下)
        slide_id: slide ID
        
    Returns:
        np.ndarray: [n_spots, n_genes] 基因表达矩阵
    """
    # h5ad文件统一在这个路径下
    h5ad_file = f"/data/ouyangjiarui/hest/processhest/adata/{slide_id}.h5ad"
    
    if not os.path.exists(h5ad_file):
        raise FileNotFoundError(f"找不到文件: {h5ad_file}")
    
    adata = sc.read_h5ad(h5ad_file)
    
    # 获取表达矩阵
    gene_expr = adata.X
    if hasattr(gene_expr, 'toarray'):
        gene_expr = gene_expr.toarray()
    
    return gene_expr.astype(np.float32)


def get_train_slides(data_path: str, val_slide: str) -> List[str]:
    """获取训练集slide列表"""
    slide_list_file = os.path.join(data_path, 'all_slide_lst.txt')
    all_slides = load_slide_list(slide_list_file)
    
    train_slides = [slide for slide in all_slides if slide != val_slide]
    
    logger.info(f"训练集slides: {len(train_slides)}个，排除验证集: {val_slide}")
    return train_slides