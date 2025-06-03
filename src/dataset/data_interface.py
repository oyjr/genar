import inspect
import importlib
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import dataset


class DataInterface(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 🆕 检测VAR-ST模式 - 修复配置访问方式
        model_name = ''
        if hasattr(config, 'MODEL') and hasattr(config.MODEL, 'model_name'):
            model_name = config.MODEL.model_name
        elif hasattr(config, 'model_name'):
            model_name = config.model_name
        
        is_var_st = model_name.upper() == 'VAR_ST'
        
        # 强制VAR-ST模型使用196基因模式
        if is_var_st:
            use_var_st_genes = True
            var_st_gene_count = 196
            print(f"🧬 检测到VAR_ST模型，强制启用196基因模式")
        else:
            use_var_st_genes = getattr(config, 'use_var_st_genes', False)
            var_st_gene_count = getattr(config, 'var_st_gene_count', 196)
        
        print(f"初始化DataInterface:")
        print(f"  - 数据集名称: STDataset")
        print(f"  - 表达谱名称: {config.expr_name}")
        print(f"  - 数据路径: {config.data_path}")
        print(f"  - 编码器: {config.encoder_name}")
        print(f"  - 使用增强: {config.use_augmented}")
        print(f"  - 🧬 检测到模型名称: {model_name}")
        print(f"  - 🆕 VAR-ST基因模式: {use_var_st_genes}")
        
        if use_var_st_genes:
            print(f"  - 🧬 VAR-ST基因数量: {var_st_gene_count}")
        
        # 存储配置以便后续使用
        self.use_var_st_genes = use_var_st_genes
        self.var_st_gene_count = var_st_gene_count

    def setup(self, stage=None):
        """
        设置数据集
        Args:
            stage: 'fit', 'val', 'test' 或 None
        """
        print(f"设置数据集阶段: {stage}")
        
        # 统一使用STDataset
        dataset_class = getattr(dataset, 'STDataset')
        
        # 基础参数配置 - 🆕 添加VAR-ST基因支持
        base_params = {
            'data_path': self.config.data_path,
            'expr_name': self.config.expr_name,
            'slide_val': self.config.slide_val,
            'slide_test': self.config.slide_test,
            'encoder_name': self.config.encoder_name,
            'use_augmented': self.config.use_augmented,
            'normalize': self.config.DATA.normalize,
            'use_var_st_genes': self.use_var_st_genes,  # 🆕 VAR-ST基因模式
            'var_st_gene_count': self.var_st_gene_count,  # 🆕 VAR-ST基因数量
        }
        
        print(f"基础参数配置: {base_params}")
        
        if stage == 'fit' or stage is None:
            train_params = base_params.copy()
            train_params['mode'] = 'train'
            # 只有训练模式才传递expand_augmented参数
            train_params['expand_augmented'] = self.config.expand_augmented
            print(f"创建训练数据集...")
            self.train_dataset = dataset_class(**train_params)
            print(f"训练数据集创建成功，大小: {len(self.train_dataset)}")
        
        if stage == 'val' or stage == 'fit' or stage is None:
            val_params = base_params.copy()
            val_params['mode'] = 'val'
            # 验证模式不使用expand_augmented
            val_params['expand_augmented'] = False
            print(f"创建验证数据集...")
            self.val_dataset = dataset_class(**val_params)
            print(f"验证数据集创建成功，大小: {len(self.val_dataset)}")
        
        if stage == 'test' or stage is None:
            test_params = base_params.copy()
            test_params['mode'] = 'test'
            # 测试模式不使用expand_augmented
            test_params['expand_augmented'] = False
            print(f"创建测试数据集...")
            self.test_dataset = dataset_class(**test_params)
            print(f"测试数据集创建成功，大小: {len(self.test_dataset)}")

        print("\n=== 数据集信息总结 ===")
        if hasattr(self, 'train_dataset'):
            print(f"训练数据集: {len(self.train_dataset)} 个样本")
            if self.use_var_st_genes:
                print(f"🧬 使用VAR-ST模式: 前{self.var_st_gene_count}个基因")
        if hasattr(self, 'val_dataset'):
            print(f"验证数据集: {len(self.val_dataset)} 个样本")
        if hasattr(self, 'test_dataset'):
            print(f"测试数据集: {len(self.test_dataset)} 个样本")
        print("====================\n")

    def train_dataloader(self):
        train_config = self.config.DATA.train_dataloader
        
        # Handle both dict and Namespace types
        if isinstance(train_config, dict):
            batch_size = train_config.get('batch_size', 256)
            shuffle = train_config.get('shuffle', True)
            pin_memory = train_config.get('pin_memory', True)
            num_workers = train_config.get('num_workers', 4)
        else:
            batch_size = getattr(train_config, 'batch_size', 256)
            shuffle = getattr(train_config, 'shuffle', True)
            pin_memory = getattr(train_config, 'pin_memory', True)
            num_workers = getattr(train_config, 'num_workers', 4)
            
        print(f"创建训练DataLoader: batch_size={batch_size}")
        return DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            pin_memory=pin_memory,
            num_workers=num_workers
        )
    
    def val_dataloader(self):
        val_config = self.config.DATA.val_dataloader
        
        # Handle both dict and Namespace types
        if isinstance(val_config, dict):
            batch_size = val_config.get('batch_size', 1)
            shuffle = val_config.get('shuffle', False)
            pin_memory = val_config.get('pin_memory', True)
            num_workers = val_config.get('num_workers', 4)
        else:
            batch_size = getattr(val_config, 'batch_size', 1)
            shuffle = getattr(val_config, 'shuffle', False)
            pin_memory = getattr(val_config, 'pin_memory', True)
            num_workers = getattr(val_config, 'num_workers', 4)
            
        print(f"创建验证DataLoader: batch_size={batch_size}")
        return DataLoader(
            self.val_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            pin_memory=pin_memory, 
            num_workers=num_workers
        )

    def test_dataloader(self):
        # Check if test_dataloader config exists, otherwise use val_dataloader config
        if hasattr(self.config.DATA, 'test_dataloader'):
            test_config = self.config.DATA.test_dataloader
        else:
            test_config = self.config.DATA.val_dataloader
        
        # Handle both dict and Namespace types
        if isinstance(test_config, dict):
            batch_size = test_config.get('batch_size', 1)
            shuffle = test_config.get('shuffle', False)
            pin_memory = test_config.get('pin_memory', True)
            num_workers = test_config.get('num_workers', 4)
        else:
            batch_size = getattr(test_config, 'batch_size', 1)
            shuffle = getattr(test_config, 'shuffle', False)
            pin_memory = getattr(test_config, 'pin_memory', True)
            num_workers = getattr(test_config, 'num_workers', 4)
            
        print(f"创建测试DataLoader: batch_size={batch_size}")
        return DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            pin_memory=pin_memory, 
            num_workers=num_workers
        )

    def predict_dataloader(self):
        return self.test_dataloader()
    
    def load_data_module(self):
        pass
                    