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
        
        print(f"初始化DataInterface:")
        print(f"  - 数据集名称: STDataset")
        print(f"  - 表达谱名称: {config.expr_name}")
        print(f"  - 数据路径: {config.data_path}")
        print(f"  - 编码器: {config.encoder_name}")
        print(f"  - 使用增强: {config.use_augmented}")

    def setup(self, stage=None):
        """
        设置数据集
        Args:
            stage: 'fit', 'val', 'test' 或 None
        """
        print(f"设置数据集阶段: {stage}")
        
        # 统一使用STDataset
        dataset_class = getattr(dataset, 'STDataset')
        
        # 基础参数配置
        base_params = {
            'data_path': self.config.data_path,
            'expr_name': self.config.expr_name,
            'slide_val': self.config.slide_val,
            'slide_test': self.config.slide_test,
            'encoder_name': self.config.encoder_name,
            'use_augmented': self.config.use_augmented,
            'normalize': self.config.DATA.normalize,
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
                    