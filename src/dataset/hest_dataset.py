import os
import pandas as pd
import numpy as np
from scipy import sparse
import torch
import scanpy as sc
import anndata as ad
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Dict, Tuple, Optional, Union


class STDataset(Dataset):
    def __init__(self,
                 mode: str,                    # 'train', 'val', 'test'
                 data_path: str,               # 数据集根路径
                 expr_name: str,               # 数据集名称
                 slide_val: str = '',          # 验证集slides
                 slide_test: str = '',         # 测试集slides
                 encoder_name: str = 'uni',    # 编码器类型
                 use_augmented: bool = False,  # 是否使用增强
                 expand_augmented: bool = False,  # 是否展开增强为多个样本
                 normalize: bool = True,       # 数据归一化
                 cpm: bool = True,            # CPM归一化
                 smooth: bool = True):        # 高斯平滑
        """
        空间转录组学数据集
        
        Args:
            mode: 数据模式 ('train', 'val', 'test')
            data_path: 数据集根路径
            expr_name: 数据集名称 (如 'PRAD')
            slide_val: 验证集slide IDs，逗号分隔
            slide_test: 测试集slide IDs，逗号分隔
            encoder_name: 编码器类型 ('uni' 或 'conch')
            use_augmented: 是否使用增强嵌入文件
            expand_augmented: 是否将3D增强嵌入展开为7倍训练样本
                - True: 每个spot变成7个训练样本 (真正的数据增强)
                - False: 只使用第一个增强版本 (原图)
            normalize: 是否进行数据归一化
            cpm: 是否进行CPM归一化
            smooth: 是否进行高斯平滑
        """
        super(STDataset, self).__init__()
        
        # 验证输入参数
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"mode must be one of ['train', 'val', 'test'], but got {mode}")
        
        if encoder_name not in ['uni', 'conch']:
            raise ValueError(f"encoder_name must be one of ['uni', 'conch'], but got {encoder_name}")
        
        # expand_augmented只在use_augmented=True且mode='train'时有效
        if expand_augmented and not use_augmented:
            print("⚠️  警告: expand_augmented=True但use_augmented=False，将被忽略")
            expand_augmented = False
        
        if expand_augmented and mode != 'train':
            print("⚠️  警告: expand_augmented只在训练模式有效，其他模式将使用第一个增强版本")
            expand_augmented = False
        
        self.mode = mode
        self.data_path = data_path
        self.expr_name = expr_name
        self.encoder_name = encoder_name
        self.use_augmented = use_augmented
        self.expand_augmented = expand_augmented
        self.norm_param = {'normalize': normalize, 'cpm': cpm, 'smooth': smooth}
        
        # 构建路径
        self.st_dir = f"{data_path}st"
        self.processed_dir = f"{data_path}processed_data"
        
        # 构建嵌入路径
        emb_suffix = "_aug" if use_augmented else ""
        self.emb_dir = f"{self.processed_dir}/1spot_{encoder_name}_ebd{emb_suffix}"
        
        # 打印初始化信息
        print(f"🔧 初始化STDataset:")
        print(f"  - 模式: {mode}")
        print(f"  - 数据路径: {data_path}")
        print(f"  - 数据集名称: {expr_name}")
        print(f"  - 编码器: {encoder_name}")
        print(f"  - 使用增强: {use_augmented}")
        
        if self.expand_augmented:
            print(f"  - 🚀 增强模式: 7倍样本展开 (每个spot变成7个训练样本)")
        elif self.use_augmented:
            print(f"  - 📊 增强模式: 只使用第一个增强版本 (原图)")
        else:
            print(f"  - 🔧 标准模式: 使用原始2D嵌入")
        
        print(f"  - ST目录: {self.st_dir}")
        print(f"  - 嵌入目录: {self.emb_dir}")
        
        # 加载基因列表
        self.genes = self.load_gene_list()
        print(f"  - 加载基因数量: {len(self.genes)}")
        
        # 加载和划分slides
        self.slide_splits = self.load_slide_splits(slide_val, slide_test)
        self.ids = self.slide_splits[mode]
        
        print(f"  - {mode}集slide数量: {len(self.ids)}")
        print(f"  - {mode}集slides: {self.ids}")
        
        self.int2id = dict(enumerate(self.ids))
        
        # 根据模式初始化
        if self.mode == 'train':
            self._init_train_mode()
        
        print(f"✅ STDataset初始化完成")

    def load_gene_list(self) -> List[str]:
        """从selected_gene_list.txt读取基因列表"""
        gene_file = f"{self.processed_dir}/selected_gene_list.txt"
        
        if not os.path.exists(gene_file):
            raise FileNotFoundError(f"基因列表文件不存在: {gene_file}")
        
        try:
            with open(gene_file, 'r', encoding='utf-8') as f:
                genes = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(genes) == 0:
                raise ValueError(f"基因列表为空: {gene_file}")
            
            print(f"从{gene_file}加载{len(genes)}个基因")
            return genes
            
        except UnicodeDecodeError as e:
            raise ValueError(f"基因列表文件编码错误: {gene_file}, 错误: {e}")
        except PermissionError as e:
            raise PermissionError(f"没有权限读取基因列表文件: {gene_file}, 错误: {e}")
        except IOError as e:
            raise IOError(f"读取基因列表文件时发生IO错误: {gene_file}, 错误: {e}")

    def load_slide_splits(self, slide_val: str, slide_test: str) -> Dict[str, List[str]]:
        """加载和划分slides"""
        # 读取所有slide列表
        slide_file = f"{self.processed_dir}/all_slide_lst.txt"

        if not os.path.exists(slide_file):
            raise FileNotFoundError(f"Slide列表文件不存在: {slide_file}")
        
        try: 
            with open(slide_file, 'r', encoding='utf-8') as f:
                all_slides = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(all_slides) == 0:
                raise ValueError(f"Slide列表为空: {slide_file}")
            
            print(f"从{slide_file}加载{len(all_slides)}个slides")
        
        except UnicodeDecodeError as e:
            raise ValueError(f"Slide列表文件编码错误: {slide_file}, 错误: {e}")
        except PermissionError as e:
            raise PermissionError(f"没有权限读取Slide列表文件: {slide_file}, 错误: {e}")
        except IOError as e:
            raise IOError(f"读取Slide列表文件时发生IO错误: {slide_file}, 错误: {e}")
        
        # 解析验证集和测试集slides
        val_slides = [s.strip() for s in slide_val.split(',') if s.strip()] if slide_val else []
        test_slides = [s.strip() for s in slide_test.split(',') if s.strip()] if slide_test else []
        
        # 验证slide ID有效性
        all_slides_set = set(all_slides)
        for slide in val_slides + test_slides:
            if slide not in all_slides_set:
                raise ValueError(f"指定的slide ID不存在: {slide}, 可用的slides: {sorted(all_slides)}")
        
        # 检查重复
        overlap = set(val_slides) & set(test_slides)
        if overlap:
            raise ValueError(f"验证集和测试集存在重复slides: {overlap}")
        
        # 剩余slides分配给训练集
        used_slides = set(val_slides + test_slides)
        train_slides = [s for s in all_slides if s not in used_slides]
        
        splits = {
            'train': train_slides,
            'val': val_slides,
            'test': test_slides
        }
        
        print(f"Slide划分:")
        print(f"  - 训练集: {len(train_slides)} slides")
        print(f"  - 验证集: {len(val_slides)} slides")
        print(f"  - 测试集: {len(test_slides)} slides")
        
        return splits

    def _init_train_mode(self):
        """初始化训练模式"""
        print("初始化训练模式数据加载...")
        
        # 预加载ST数据
        self.adata_dict = {}
        for slide_id in self.ids:
            print(f"加载{slide_id}的ST数据...")
            self.adata_dict[slide_id] = self.load_st(slide_id, self.genes, **self.norm_param)
        
        if self.expand_augmented:
            print("🚀 启用增强样本展开模式：每个spot扩展为7个训练样本")
            
            # 在展开模式下，预加载并展开嵌入数据
            self.expanded_emb_dict = {}
            self.expanded_adata_dict = {}
            
            for slide_id in self.ids:
                print(f"展开{slide_id}的增强数据...")
                
                # 加载3D嵌入数据
                emb = self.load_emb(slide_id, None, 'all')  # [num_spots, 7, feature_dim]
                original_adata = self.adata_dict[slide_id]
                
                if len(emb.shape) == 3:
                    # 展开嵌入：[num_spots, 7, feature_dim] -> [num_spots*7, feature_dim]
                    num_spots, num_augs, feature_dim = emb.shape
                    expanded_emb = emb.reshape(-1, feature_dim)  # [num_spots*7, feature_dim]
                    
                    # 展开基因表达数据：[num_spots, num_genes] -> [num_spots*7, num_genes]
                    if sparse.issparse(original_adata.X):
                        original_X = original_adata.X.toarray()
                    else:
                        original_X = original_adata.X
                    
                    # 每个spot的表达数据重复7次
                    expanded_X = np.repeat(original_X, num_augs, axis=0)  # [num_spots*7, num_genes]
                    
                    # 展开位置信息
                    expanded_positions = np.repeat(original_adata.obsm['positions'], num_augs, axis=0)
                    
                    # 创建展开后的AnnData对象
                    expanded_adata = ad.AnnData(X=expanded_X, var=original_adata.var.copy())
                    expanded_adata.var_names = original_adata.var_names
                    expanded_adata.obsm['positions'] = expanded_positions
                    
                    # 添加增强信息到obs
                    aug_ids = np.tile(np.arange(num_augs), num_spots)  # [0,1,2,3,4,5,6, 0,1,2,3,4,5,6, ...]
                    spot_ids = np.repeat(np.arange(num_spots), num_augs)  # [0,0,0,0,0,0,0, 1,1,1,1,1,1,1, ...]
                    
                    expanded_adata.obs['original_spot_id'] = spot_ids
                    expanded_adata.obs['aug_id'] = aug_ids
                    expanded_adata.obs['array_row'] = expanded_positions[:, 0]
                    expanded_adata.obs['array_col'] = expanded_positions[:, 1]
                    
                    self.expanded_emb_dict[slide_id] = expanded_emb
                    self.expanded_adata_dict[slide_id] = expanded_adata
                    
                    print(f"  {slide_id}: {num_spots} spots -> {num_spots*num_augs} 增强样本")
                else:
                    # 如果不是3D格式，保持原样
                    print(f"  {slide_id}: 非3D格式，保持原始{emb.shape[0]}个样本")
                    self.expanded_emb_dict[slide_id] = emb
                    self.expanded_adata_dict[slide_id] = original_adata
            
            # 使用展开后的数据计算长度
            self.lengths = [len(adata) for adata in self.expanded_adata_dict.values()]
            
        else:
            # 原有模式：计算累积长度用于索引映射
            self.lengths = [len(adata) for adata in self.adata_dict.values()]
        
        self.cumlen = np.cumsum(self.lengths)
        
        print(f"训练数据统计:")
        print(f"  - 各slide样本数量: {self.lengths}")
        print(f"  - 累积长度: {self.cumlen}")
        print(f"  - 总样本数量: {self.cumlen[-1]}")
        if self.expand_augmented:
            original_total = sum(len(self.adata_dict[slide_id]) for slide_id in self.ids)
            print(f"  - 原始spot数量: {original_total}")
            print(f"  - 扩展倍数: {self.cumlen[-1] / original_total:.1f}x")

    def load_emb(self, slide_id: str, idx: Optional[int] = None, mode: str = 'first') -> torch.Tensor:
        """加载嵌入特征
        
        Args:
            slide_id: slide标识符
            idx: spot索引，如果None则返回所有spots
            mode: 3D增强嵌入的处理模式
                - 'first': 使用第一个增强版本 (原图)
                - 'all': 返回所有7个版本 (用于expand_augmented)
        """
        # 构建文件名，增强嵌入需要添加_aug后缀
        if self.use_augmented:
            emb_file = f"{self.emb_dir}/{slide_id}_{self.encoder_name}_aug.pt"
        else:
            emb_file = f"{self.emb_dir}/{slide_id}_{self.encoder_name}.pt"
        
        if not os.path.exists(emb_file):
            raise FileNotFoundError(f"嵌入文件不存在: {emb_file}")
        
        try:
            # 使用weights_only=True确保安全
            emb = torch.load(emb_file, weights_only=True)
            
            if not isinstance(emb, torch.Tensor):
                raise TypeError(f"嵌入文件格式错误，期望torch.Tensor，得到{type(emb)}")
            
            # 处理不同的tensor维度
            if len(emb.shape) == 3:
                # 3D tensor: [num_spots, num_augmentations, feature_dim]
                if mode == 'first':
                    # 使用第一个增强版本（原图）
                    print(f"检测到3D增强嵌入格式: {emb.shape} -> 使用第一个增强版本")
                    emb = emb[:, 0, :]  # [num_spots, feature_dim]
                elif mode == 'all':
                    # 返回所有增强版本 (用于expand_augmented)
                    print(f"检测到3D增强嵌入格式: {emb.shape} -> 保留所有增强版本")
                    pass  # 保持原始3D格式
                else:
                    raise ValueError(f"不支持的模式: {mode}，只支持 'first' 或 'all'")
                    
            elif len(emb.shape) == 2:
                # 2D tensor: [num_spots, feature_dim] (标准格式)
                pass
            else:
                raise ValueError(f"嵌入维度错误，期望2D或3D tensor，得到{emb.shape}")
            
            # 根据编码器类型验证特征维度
            expected_dim = 1024 if self.encoder_name == 'uni' else 512
            final_dim = emb.shape[-1]  # 获取最后一维
            if final_dim != expected_dim:
                raise ValueError(f"嵌入特征维度错误，{self.encoder_name}编码器期望{expected_dim}维，得到{final_dim}维")
            
            # 返回指定索引或全部
            if idx is not None:
                if idx >= emb.shape[0]:
                    raise IndexError(f"索引越界: {idx} >= {emb.shape[0]}")
                if mode == 'all' and len(emb.shape) == 3:
                    return emb[idx]  # [num_augmentations, feature_dim]
                else:
                    return emb[idx]  # [feature_dim]
            else:
                return emb  # [num_spots, feature_dim] 或 [num_spots, num_augmentations, feature_dim]
                
        except FileNotFoundError:
            raise FileNotFoundError(f"嵌入文件不存在: {emb_file}")
        except PermissionError as e:
            raise PermissionError(f"没有权限读取嵌入文件: {emb_file}, 错误: {e}")
        except torch.serialization.pickle.UnpicklingError as e:
            raise ValueError(f"嵌入文件损坏或格式不正确: {emb_file}, 错误: {e}")
        except RuntimeError as e:
            raise RuntimeError(f"加载嵌入文件时发生运行时错误: {emb_file}, 错误: {e}")
        except Exception as e:
            raise ValueError(f"加载嵌入文件失败 {emb_file}: {e}")

    def load_st(self, slide_id: str, genes: Optional[List[str]] = None, **kwargs) -> ad.AnnData:
        """加载ST数据"""
        st_file = f"{self.st_dir}/{slide_id}.h5ad"
        
        if not os.path.exists(st_file):
            raise FileNotFoundError(f"ST文件不存在: {st_file}")
        
        print(f"加载ST数据: {st_file}")
        
        try:
            adata = sc.read_h5ad(st_file)
            
            # 检查必要的键
            if 'spatial' not in adata.obsm:
                raise ValueError(f"ST数据缺少spatial坐标: {st_file}")
            
            # 标准化坐标到0-1范围
            coords = adata.obsm['spatial'].copy()
            coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0))
            
            # 添加array_row和array_col到obs（如果不存在）
            if 'array_row' not in adata.obs.columns:
                adata.obs['array_row'] = adata.obsm['spatial'][:, 0]
                adata.obs['array_col'] = adata.obsm['spatial'][:, 1]
            
            # 基因过滤
            if genes is not None:
                print(f"过滤基因，从{adata.n_vars}个基因中选择{len(genes)}个目标基因")
                
                common_genes = list(set(genes).intersection(set(adata.var_names)))
                if len(common_genes) < len(genes):
                    missing_genes = list(set(genes) - set(common_genes))
                    print(f"警告: {len(missing_genes)}个基因在{slide_id}中不存在: {missing_genes[:5]}...")
                
                adata = adata[:, common_genes].copy()
                print(f"过滤后保留{adata.n_vars}个基因")
            
            # 数据归一化
            if kwargs.get('normalize', True):
                print("执行数据归一化...")
                
                # 1. CPM归一化
                if kwargs.get('cpm', True):
                    print("  - CPM归一化")
                    sc.pp.normalize_total(adata, target_sum=1e6, inplace=True)
                
                # 2. 对数变换
                print("  - 对数变换")
                sc.pp.log1p(adata)
                
                # 3. Z-score标准化
                print("  - Z-score标准化")
                if sparse.issparse(adata.X):
                    X = adata.X.toarray()
                else:
                    X = adata.X
                
                gene_mean = X.mean(axis=0)
                gene_std = X.std(axis=0)
                gene_std[gene_std == 0] = 1.0
                
                X = (X - gene_mean) / gene_std
                adata.X = sparse.csr_matrix(X) if sparse.issparse(adata.X) else X
                
                # 4. 高斯平滑（可选）
                if kwargs.get('smooth', False):
                    print("  - 高斯平滑")
                    if sparse.issparse(adata.X):
                        adata.X = sparse.csr_matrix(gaussian_filter(adata.X.toarray(), sigma=1))
                    else:
                        adata.X = gaussian_filter(adata.X, sigma=1)
            
            # 保存标准化后的坐标
            adata.obsm['positions'] = coords
            
            print(f"ST数据加载完成: {adata.n_obs} spots, {adata.n_vars} genes")
            return adata
            
        except Exception as e:
            raise ValueError(f"加载ST数据失败 {st_file}: {e}")

    def __len__(self) -> int:
        if self.mode == 'train':
            return self.cumlen[-1] if len(self.cumlen) > 0 else 0
        else:
            return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.mode == 'train':
            return self._get_train_item(index)
        else:
            return self._get_eval_item(index)

    def _get_train_item(self, index: int) -> Dict[str, torch.Tensor]:
        """训练模式获取单个spot数据"""
        # 找到对应的slide和样本索引
        i = 0
        while index >= self.cumlen[i]:
            i += 1
        
        sample_idx = index
        if i > 0:
            sample_idx = index - self.cumlen[i-1]
        
        slide_id = self.int2id[i]
        
        if self.expand_augmented and hasattr(self, 'expanded_emb_dict'):
            # 使用预展开的数据
            features = self.expanded_emb_dict[slide_id][sample_idx]  # [feature_dim]
            
            # 从展开的AnnData中获取基因表达
            expanded_adata = self.expanded_adata_dict[slide_id]
            expression = expanded_adata[sample_idx].X
            
            if sparse.issparse(expression):
                expression = expression.toarray().squeeze(0)
            else:
                expression = expression.squeeze(0)
            
            # 获取位置信息
            positions = expanded_adata.obsm['positions'][sample_idx]  # [2]
            
            # 获取增强信息（可选，用于调试）
            original_spot_id = int(expanded_adata.obs['original_spot_id'].iloc[sample_idx])
            aug_id = int(expanded_adata.obs['aug_id'].iloc[sample_idx])
            
            return {
                'img': torch.FloatTensor(features),  # [feature_dim]
                'target_genes': torch.FloatTensor(expression),  # [num_genes]
                'positions': torch.FloatTensor(positions),  # [2]
                'slide_id': slide_id,
                'spot_idx': sample_idx,
                'original_spot_id': original_spot_id,  # 原始spot ID
                'aug_id': aug_id  # 增强版本ID (0-6)
            }
        else:
            # 原有模式：动态加载
            features = self.load_emb(slide_id, sample_idx, 'first')  # [feature_dim]
            
            # 加载基因表达
            adata = self.adata_dict[slide_id]
            expression = adata[sample_idx].X
            
            if sparse.issparse(expression):
                expression = expression.toarray().squeeze(0)
            else:
                expression = expression.squeeze(0)
            
            # 加载位置信息
            positions = adata.obsm['positions'][sample_idx]  # [2]
            
            return {
                'img': features,  # [feature_dim]
                'target_genes': torch.FloatTensor(expression),  # [num_genes]
                'positions': torch.FloatTensor(positions),  # [2]
                'slide_id': slide_id,
                'spot_idx': sample_idx
            }

    def _get_eval_item(self, index: int) -> Dict[str, torch.Tensor]:
        """验证/测试模式获取整个slide数据"""
        slide_id = self.int2id[index]
        
        # 加载嵌入特征
        features = self.load_emb(slide_id, None, 'first')  # [num_spots, feature_dim]
        
        # 加载ST数据
        adata = self.load_st(slide_id, self.genes, **self.norm_param)
        
        # 加载基因表达
        expression = adata.X
        if sparse.issparse(expression):
            expression = expression.toarray()
        
        # 加载位置信息
        positions = adata.obsm['positions']  # [num_spots, 2]
        
        return {
            'img': features,  # [num_spots, feature_dim]
            'target_genes': torch.FloatTensor(expression),  # [num_spots, num_genes]
            'positions': torch.FloatTensor(positions),  # [num_spots, 2]
            'slide_id': slide_id,
            'num_spots': adata.n_obs
        }

