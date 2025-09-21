import os
import pandas as pd
import numpy as np
from scipy import sparse
import torch
import scanpy as sc
import anndata as ad
from torch.utils.data import Dataset
from typing import List, Dict, Optional


class STDataset(Dataset):
    """空间转录组学数据集"""
    
    def __init__(self,
                 mode: str,                    # 'train', 'val', 'test'
                 data_path: str,               # 数据集根路径
                 expr_name: str,               # 数据集名称
                 slide_val: str = '',          # 验证集slides
                 slide_test: str = '',         # 测试集slides
                 encoder_name: str = 'uni',    # 编码器类型
                 use_augmented: bool = False,  # 是否使用增强
                 expand_augmented: bool = False,  # 是否展开增强为多个样本
                 max_gene_count: int = 500):   # 基因表达计数值上限
        """
        空间转录组学数据集
        
        Args:
            mode: 数据集模式 ('train', 'val', 'test')
            data_path: 数据集根路径
            expr_name: 数据集名称
            slide_val: 验证集slides (逗号分隔)
            slide_test: 测试集slides (逗号分隔)
                 encoder_name: 编码器类型 ('uni', 'conch', 'resnet18')
            use_augmented: 是否使用增强数据
            expand_augmented: 是否展开增强数据为多个样本
            max_gene_count: 基因表达计数值上限
        """
        super().__init__()
        
        # 验证输入参数
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"mode must be one of ['train', 'val', 'test'], got {mode}")
        if encoder_name not in ['uni', 'conch', 'resnet18']:
            raise ValueError(
                "encoder_name must be one of ['uni', 'conch', 'resnet18'], "
                f"got {encoder_name}"
            )
        
        # expand_augmented只在训练模式且使用增强时有效
        if expand_augmented and (not use_augmented or mode != 'train'):
            expand_augmented = False
        
        self.mode = mode
        self.data_path = data_path
        self.expr_name = expr_name
        self.encoder_name = encoder_name
        self.use_augmented = use_augmented
        self.expand_augmented = expand_augmented
        self.max_gene_count = max_gene_count
        
        # 构建路径
        self.st_dir = f"{data_path}st"
        self.processed_dir = f"{data_path}processed_data"
        
        # 构建嵌入路径
        emb_suffix = "_aug" if use_augmented else ""
        self.emb_dir = f"{self.processed_dir}/1spot_{encoder_name}_ebd{emb_suffix}"
        
        print(f"初始化STDataset: {mode}模式, {expr_name}数据集, {encoder_name}编码器")
        
        # 加载基因列表（固定使用前200个基因）
        self.genes = self._load_gene_list()
        
        # 加载和划分slides
        self.slide_splits = self._load_slide_splits(slide_val, slide_test)
        self.ids = self.slide_splits[mode]
        self.int2id = dict(enumerate(self.ids))
        
        print(f"加载{len(self.genes)}个基因, {len(self.ids)}个slides")
        
        # 根据模式初始化
        if mode == 'train':
            self._init_train_mode()

    def _load_gene_list(self) -> List[str]:
        """加载基因列表，固定使用前200个基因"""
        gene_file = f"{self.processed_dir}/selected_gene_list.txt"
        
        if not os.path.exists(gene_file):
            raise FileNotFoundError(f"基因列表文件不存在: {gene_file}")
        
        with open(gene_file, 'r', encoding='utf-8') as f:
            all_genes = [line.strip() for line in f.readlines() if line.strip()]
        
        if len(all_genes) < 200:
            raise ValueError(f"数据集只有{len(all_genes)}个基因，少于需要的200个基因")
        
        return all_genes[:200]  # 固定使用前200个基因

    def _load_slide_splits(self, slide_val: str, slide_test: str) -> Dict[str, List[str]]:
        """加载和划分slides"""
        slide_file = f"{self.processed_dir}/all_slide_lst.txt"
        
        if not os.path.exists(slide_file):
            raise FileNotFoundError(f"Slide列表文件不存在: {slide_file}")
        
        with open(slide_file, 'r', encoding='utf-8') as f:
            all_slides = [line.strip() for line in f.readlines() if line.strip()]
        
        # 解析验证集和测试集slides
        val_slides = [s.strip() for s in slide_val.split(',') if s.strip()] if slide_val else []
        test_slides = [s.strip() for s in slide_test.split(',') if s.strip()] if slide_test else []
        
        # 验证slide ID有效性
        all_slides_set = set(all_slides)
        for slide in val_slides + test_slides:
            if slide not in all_slides_set:
                raise ValueError(f"指定的slide ID不存在: {slide}")
        
        # 计算训练集slides
        train_slides = [s for s in all_slides if s not in val_slides and s not in test_slides]
        
        return {
            'train': train_slides,
            'val': val_slides,
            'test': test_slides
        }

    def _init_train_mode(self):
        """初始化训练模式"""
        # 预加载所有训练数据的adata
        self.adata_dict = {}
        lengths = []
        
        for slide_id in self.ids:
            adata = self._load_st(slide_id)
            self.adata_dict[slide_id] = adata
            
            if self.expand_augmented:
                # 展开增强：每个spot变成7个样本
                lengths.append(len(adata) * 7)
                # 预处理展开的数据
                self._prepare_expanded_data(slide_id, adata)
            else:
                lengths.append(len(adata))
        
        self.cumlen = np.cumsum(lengths)
        
        if self.expand_augmented:
            print(f"训练模式：展开增强，总样本数 {self.cumlen[-1]}")
        else:
            print(f"训练模式：标准模式，总样本数 {self.cumlen[-1]}")

    def _prepare_expanded_data(self, slide_id: str, adata: ad.AnnData):
        """准备展开的增强数据"""
        if not hasattr(self, 'expanded_emb_dict'):
            self.expanded_emb_dict = {}
            self.expanded_adata_dict = {}
        
        # 加载增强嵌入（包含所有7个版本的3D tensor）
        emb = self._load_emb(slide_id, None, 'aug')  # [num_spots, 7, feature_dim] 或 [num_spots*7, feature_dim]
        
        # 处理不同的tensor格式
        if len(emb.shape) == 3:
            # 3D格式：[num_spots, 7, feature_dim] -> [num_spots*7, feature_dim]
            num_spots, num_augs, feature_dim = emb.shape
            expanded_emb = emb.reshape(-1, feature_dim)
        else:
            # 已经是展开格式：[num_spots*7, feature_dim]
            expanded_emb = emb
        
        self.expanded_emb_dict[slide_id] = expanded_emb
        
        # 展开AnnData
        expanded_obs_data = []
        expanded_X_data = []
        expanded_positions = []
        
        for aug_idx in range(7):
            for spot_idx in range(len(adata)):
                expanded_obs_data.append({
                    'original_spot_id': spot_idx,
                    'aug_id': aug_idx
                })
                expanded_X_data.append(adata.X[spot_idx])
                expanded_positions.append(adata.obsm['positions'][spot_idx])
        
        # 创建展开的AnnData
        obs_df = pd.DataFrame(expanded_obs_data)
        obs_df.index = obs_df.index.astype(str)  # 确保索引是字符串类型，避免警告

        expanded_adata = ad.AnnData(
            X=sparse.vstack(expanded_X_data) if sparse.issparse(adata.X) else np.vstack(expanded_X_data),
            obs=obs_df,
            var=adata.var.copy()
        )
        expanded_adata.obsm['positions'] = np.array(expanded_positions)
        
        self.expanded_adata_dict[slide_id] = expanded_adata

    def _load_emb(self, slide_id: str, idx: Optional[int] = None, mode: str = 'standard') -> torch.Tensor:
        """加载嵌入特征
        
        Args:
            slide_id: slide标识符
            idx: spot索引，如果None则返回所有spots
            mode: 'standard' 使用标准嵌入, 'aug' 使用增强嵌入
        """
        if mode == 'aug' and self.use_augmented:
            emb_file = f"{self.emb_dir}/{slide_id}_{self.encoder_name}_aug.pt"
        else:
            # 标准模式或者不使用增强时
            base_dir = self.emb_dir.replace('_aug', '')  # 确保使用标准目录
            emb_file = f"{base_dir}/{slide_id}_{self.encoder_name}.pt"
        
        if not os.path.exists(emb_file):
            raise FileNotFoundError(f"嵌入文件不存在: {emb_file}")
        
        features = torch.load(emb_file, map_location='cpu', weights_only=True)
        
        # 处理3D增强嵌入格式
        if mode == 'aug' and len(features.shape) == 3:
            # 3D格式：[num_spots, num_augs, feature_dim]
            if idx is not None:
                return features[idx, 0, :]  # 返回第一个增强版本 [feature_dim]
            else:
                return features  # 返回完整3D tensor
        
        # 标准2D格式处理
        if idx is not None:
            return features[idx]  # [feature_dim]
        else:
            return features  # [num_spots, feature_dim]

    def _load_st(self, slide_id: str) -> ad.AnnData:
        """加载ST数据"""
        st_file = f"{self.st_dir}/{slide_id}.h5ad"
        
        if not os.path.exists(st_file):
            raise FileNotFoundError(f"ST文件不存在: {st_file}")
        
        adata = sc.read_h5ad(st_file)
        
        # 选择指定的基因
        adata = adata[:, self.genes].copy()
        
        # 处理位置信息
        if 'spatial' in adata.obsm:
            # 标准化坐标到0-1范围
            coords = adata.obsm['spatial'].copy()
            coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))
            adata.obsm['positions'] = coords
        elif 'positions' not in adata.obsm:
            # 如果没有位置信息，创建默认位置
            import numpy as np
            adata.obsm['positions'] = np.random.rand(adata.n_obs, 2)
        
        return adata

    def __len__(self) -> int:
        if self.mode == 'train':
            return self.cumlen[-1] if len(self.cumlen) > 0 else 0
        else:
            # 验证/测试模式：计算总spot数
            if not hasattr(self, 'total_spots'):
                self.total_spots = sum(len(self._load_st(slide_id)) for slide_id in self.ids)
            return self.total_spots

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """统一的数据获取方法"""
        if self.mode == 'train':
            return self._get_train_item(index)
        else:
            return self._get_eval_item(index)

    def _get_train_item(self, index: int) -> Dict[str, torch.Tensor]:
        """训练模式获取数据"""
        # 找到对应的slide和样本索引
        i = 0
        while index >= self.cumlen[i]:
            i += 1
        
        sample_idx = index - (self.cumlen[i-1] if i > 0 else 0)
        slide_id = self.int2id[i]
        
        if self.expand_augmented:
            # 使用预展开的数据
            features = self.expanded_emb_dict[slide_id][sample_idx]
            expanded_adata = self.expanded_adata_dict[slide_id]
            expression = expanded_adata[sample_idx].X
            positions = expanded_adata.obsm['positions'][sample_idx]
            
            # 获取增强信息
            original_spot_id = int(expanded_adata.obs['original_spot_id'].iloc[sample_idx])
            aug_id = int(expanded_adata.obs['aug_id'].iloc[sample_idx])
            
            return {
                'img': torch.FloatTensor(features),
                'target_genes': self._process_gene_expression(expression),
                'positions': torch.FloatTensor(positions),
                'slide_id': slide_id,
                'spot_idx': sample_idx,
                'original_spot_id': original_spot_id,
                'aug_id': aug_id
            }
        else:
            # 标准模式
            features = self._load_emb(slide_id, sample_idx, 'standard')
            adata = self.adata_dict[slide_id]
            expression = adata[sample_idx].X
            positions = adata.obsm['positions'][sample_idx]
            
            return {
                'img': features,
                'target_genes': self._process_gene_expression(expression),
                'positions': torch.FloatTensor(positions),
                'slide_id': slide_id,
                'spot_idx': sample_idx
            }

    def _get_eval_item(self, index: int) -> Dict[str, torch.Tensor]:
        """验证/测试模式获取数据"""
        # 计算累积长度（如果还没有）
        if not hasattr(self, 'eval_cumlen'):
            lengths = [len(self._load_st(slide_id)) for slide_id in self.ids]
            self.eval_cumlen = np.cumsum(lengths)
        
        # 找到对应的slide和样本索引
        i = 0
        while index >= self.eval_cumlen[i]:
            i += 1
        
        sample_idx = index - (self.eval_cumlen[i-1] if i > 0 else 0)
        slide_id = self.int2id[i]
        
        # 加载数据
        features = self._load_emb(slide_id, sample_idx, 'standard')
        adata = self._load_st(slide_id)
        expression = adata[sample_idx].X
        positions = adata.obsm['positions'][sample_idx]
        
        return {
            'img': torch.FloatTensor(features),
            'target_genes': self._process_gene_expression(expression),
            'positions': torch.FloatTensor(positions),
            'slide_id': slide_id,
            'spot_idx': sample_idx
        }

    def get_full_slide_for_testing(self, slide_id: str) -> Dict[str, torch.Tensor]:
        """获取完整slide的所有spot数据用于测试"""
        features = self._load_emb(slide_id, None, 'standard')  # [num_spots, feature_dim]
        adata = self._load_st(slide_id)
        
        expression = adata.X
        if sparse.issparse(expression):
            expression = expression.toarray()
        
        positions = adata.obsm['positions']
        
        return {
            'img': torch.FloatTensor(features),
            'target_genes': self._process_gene_expression(expression),
            'positions': torch.FloatTensor(positions),
            'slide_id': slide_id,
            'num_spots': adata.n_obs,
            'adata': adata
        }

    def get_test_slide_ids(self) -> List[str]:
        """获取测试集的slide ID列表"""
        return self.slide_splits['test'] if self.mode != 'test' else self.ids

    def _process_gene_expression(self, gene_expr) -> torch.Tensor:
        """处理基因表达数据，保持原始计数值"""
        if sparse.issparse(gene_expr):
            gene_expr = gene_expr.toarray().squeeze()
        else:
            gene_expr = np.asarray(gene_expr).squeeze()
        
        # 确保非负整数并截断
        gene_expr = np.maximum(0, gene_expr)
        gene_expr = np.round(gene_expr).astype(np.int64)
        gene_tokens = torch.clamp(torch.from_numpy(gene_expr).long(), 0, self.max_gene_count)
        
        return gene_tokens
