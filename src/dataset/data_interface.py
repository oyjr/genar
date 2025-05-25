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
            'cpm': self.config.DATA.cpm,
            'smooth': self.config.DATA.smooth,
        }
        
        print(f"基础参数配置: {base_params}")
        
        if stage == 'fit' or stage is None:
            train_params = base_params.copy()
            train_params['mode'] = 'train'
            print(f"创建训练数据集...")
            self.train_dataset = dataset_class(**train_params)
            print(f"训练数据集创建成功，大小: {len(self.train_dataset)}")
        
        if stage == 'val' or stage == 'fit' or stage is None:
            val_params = base_params.copy()
            val_params['mode'] = 'val'
            print(f"创建验证数据集...")
            self.val_dataset = dataset_class(**val_params)
            print(f"验证数据集创建成功，大小: {len(self.val_dataset)}")
        
        if stage == 'test' or stage is None:
            test_params = base_params.copy()
            test_params['mode'] = 'test'
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
        print(f"创建训练DataLoader: batch_size={train_config.batch_size}")
        return DataLoader(
            self.train_dataset, 
            batch_size=train_config.batch_size, 
            shuffle=train_config.shuffle, 
            pin_memory=train_config.pin_memory,
            num_workers=train_config.num_workers
        )
    
    def val_dataloader(self):
        val_config = self.config.DATA.val_dataloader
        print(f"创建验证DataLoader: batch_size={val_config.batch_size}")
        return DataLoader(
            self.val_dataset, 
            batch_size=val_config.batch_size, 
            shuffle=val_config.shuffle, 
            pin_memory=val_config.pin_memory, 
            num_workers=val_config.num_workers
        )

    def test_dataloader(self):
        test_config = self.config.DATA.test_dataloader
        print(f"创建测试DataLoader: batch_size={test_config.batch_size}")
        return DataLoader(
            self.test_dataset, 
            batch_size=test_config.batch_size, 
            shuffle=test_config.shuffle, 
            pin_memory=test_config.pin_memory, 
            num_workers=test_config.num_workers
        )

    def predict_dataloader(self):
        return self.test_dataloader()
    
    def load_data_module(self):
        pass
                    