import os
import inspect
import importlib
import numpy as np
import torch
import torchmetrics
from torchmetrics.regression import (
    PearsonCorrCoef,
    MeanAbsoluteError,
    MeanSquaredError,
    ConcordanceCorrCoef,   
)

from scipy.stats import pearsonr

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
import torch.nn.functional as F



class ModelInterface(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.save_hyperparameters()
        
        self.debug_mode = config.get('debug', False)

        self.model_name = config.MODEL.model_name if hasattr(config.MODEL, 'model_name') else None
        self.model_config = config.MODEL

        if self.debug_mode:
            print(f"Model config: {self.model_config}")
        self.model = self.load_model()

        if self.debug_mode:
            print(f"Model: {self.model}")

        self.criterion = torch.nn.MSELoss()

        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

        self._init_metrics()

        self.config = config

        if hasattr(config.DATA, 'normalize'):
            self.normalize = config.DATA.normalize
        else:
            self.normalize = True

        self.avg_pcc = None







    def training_step(self, batch, batch_idx):
         
        self._debug_print(f'============Training step============')

        batch = self._preprocess_inputs(batch)

        results_dict = self.model(**batch)

        # 计算损失
        loss = self._compute_loss(results_dict, batch)

        logits = results_dict['logits']
        target_genes = batch['target_genes']

        # 更新指标
        self._update_metrics('train', logits, target_genes)

        self.log('train_loss', loss, on_epoch=True, logger=True, sync_dist=True)
                
        return loss






    def validation_step(self, batch, batch_idx):
        self._debug_print(f"\n[DEBUG] ===== 验证步骤 {batch_idx} =====")
        
        # 预处理输入
        batch = self._preprocess_inputs(batch)
        
        # 获取模型输出
        results_dict = self.model(**batch)

        # 计算损失
        loss = self._compute_loss(results_dict, batch)
        
        # 获取预测和目标
        logits = results_dict['logits']
        target_genes = batch['target_genes']
        
        # 更新指标
        self._update_metrics('val', logits, target_genes)
        
        # 记录损失
        self.log('val_loss', loss, on_epoch=True, logger=True, sync_dist=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        self._debug_shapes(batch)
        
        batch = self._preprocess_inputs(batch)

        results_dict = self.model(**batch)
        logits = results_dict['logits']
        target_genes = batch['target_genes']
        
        # 计算损失和指标
        loss = self._compute_loss(results_dict, batch)
        self._debug_print(f"[DEBUG] 测试损失: {loss.item():.4f}")
        
        # 更新指标
        self._update_metrics('test', logits, target_genes)
        
        # 保存输出
        self._save_step_outputs('test', loss, logits, target_genes, batch_idx)
        
        return {'logits': logits, 'target_genes': target_genes}
    
    def configure_optimizers(self):
        weight_decay = float(self.config.TRAINING.get('weight_decay', 0.0))
        learning_rate = float(self.config.TRAINING.learning_rate)
        
        # 配置优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 配置学习率调度器
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.config.TRAINING.mode,
                factor=self.config.TRAINING.lr_scheduler.factor,
                patience=self.config.TRAINING.lr_scheduler.patience,
                verbose=True
            ),
            'monitor': self.config.TRAINING.monitor,
            'interval': 'epoch',
            'frequency': 1
        }
        
        # 添加梯度裁剪
        self.trainer.gradient_clip_val = self.config.TRAINING.get('gradient_clip_val', 1.0)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    

    def _init_metrics(self):
        # 从基因列表文件获取基因数量，如果不存在则使用默认值
        if hasattr(self.config, 'data_path'):
            gene_file = f"{self.config.data_path}processed_data/selected_gene_list.txt"
            try:
                with open(gene_file, 'r') as f:
                    genes = [line.strip() for line in f.readlines() if line.strip()]
                num_outputs = len(genes)
                print(f"从基因列表文件获取基因数量: {num_outputs}")
            except:
                num_outputs = 200  # 默认值
                print(f"无法读取基因列表文件，使用默认基因数量: {num_outputs}")
        else:
            num_outputs = 200  # 默认值
            print(f"配置中无数据路径，使用默认基因数量: {num_outputs}")

        metrics = {
            'mse': MeanSquaredError(num_outputs=num_outputs),
            'mae': MeanAbsoluteError(num_outputs=num_outputs),
            'pearson': PearsonCorrCoef(num_outputs=num_outputs),
            'concordance': ConcordanceCorrCoef(num_outputs=num_outputs),
        }

        self.train_metrics = torchmetrics.MetricCollection(metrics.copy())
        self.val_metrics = torchmetrics.MetricCollection(metrics.copy())
        self.test_metrics = torchmetrics.MetricCollection(metrics.copy())

        self.train_history = {
            'loss': [],
            'mse': [],
            'mae': [],
            'pearson_mean': [],
            'pearson_high': [],
        }

        self.val_history = {
            'loss': [],
            'mse': [],
            'mae': [],
            'pearson_mean': [],
            'pearson_high': [],
        }


    def _preprocess_inputs(self, inputs):
        """预处理输入数据，适配新的数据格式"""
        # 处理图像特征
        if 'img' in inputs and len(inputs['img'].shape) == 5:
            inputs['img'] = inputs['img'].squeeze(0)
        
        # 处理目标基因（新格式）
        if 'target_genes' in inputs and len(inputs['target_genes'].shape) == 3:
            inputs['target_genes'] = inputs['target_genes'].squeeze(0)
        
        # 处理位置信息
        if 'positions' in inputs and len(inputs['positions'].shape) == 3:
            inputs['positions'] = inputs['positions'].squeeze(0)
        
        return inputs


    def _debug_print(self, msg):
        if hasattr(self, 'debug_mode') and self.debug_mode:
            print(msg)

    def _debug_shapes(self, tensors_dict, prefix=""):
        if self.debug_mode:
            for name, tensor in tensors_dict.items():
                if isinstance(tensor, torch.Tensor):
                    print(f"{prefix}{name}: {tensor.shape}")

    

    def _update_metrics(self, stage, predictions, targets):
        try:
            # 获取对应阶段的指标集合
            metrics = getattr(self, f'{stage}_metrics')
            
            # 确保输入维度正确
            if predictions.dim() == 3:
                B, N, G = predictions.shape
                predictions = predictions.reshape(-1, G)  # [B*N, num_genes]
            if targets.dim() == 3:
                B, N, G = targets.shape
                targets = targets.reshape(-1, G)  # [B*N, num_genes]
            
            # 更新指标
            metrics.update(predictions, targets)

            metric_dict = metrics.compute()
            for name, value in metric_dict.items():
                if isinstance(value, torch.Tensor):
                    values = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
                    mean_value = values.mean()
                    std_value = values.std()
                
                    self.log(f'{stage}_{name}', mean_value, prog_bar=True)
                    self.log(f'{stage}_{name}_std', std_value, prog_bar=True)

                    if name == 'pearson':
                        top_k = max(1,int(len(values)*0.3))
                        high_values = torch.topk(values, top_k)[0]
                        high_mean = high_values.mean()
                        high_std = high_values.std()
                        self.log(f'{stage}_pearson_high_mean', high_mean, prog_bar=True)
                        self.log(f'{stage}_pearson_high_std', high_std, prog_bar=True)

        except Exception as e:
            self._debug_print(f"更新指标时发生错误: {e}")
            raise e
    
    
    def _save_step_outputs(self, phase, loss, preds, targets, batch_idx=None):
        output_dict = {
            'loss': loss.detach(),
            'preds': preds,
            'targets': targets,
        }
        if batch_idx is not None:
            output_dict['batch_idx'] = batch_idx

        getattr(self, f'{phase}_outputs').append(output_dict)

    def _process_epoch_end(self, phase):
        outputs = getattr(self, f'{phase}_outputs')
        if len(outputs) == 0:
            return
        
        # 清空输出列表
        outputs.clear()

    def on_train_epoch_end(self):
        self._process_epoch_end('train')
        
    def on_validation_epoch_end(self):
        self._process_epoch_end('val')
        
    def on_test_epoch_end(self):
        self._process_epoch_end('test')

    def predict_step(self, batch, batch_idx):
        batch = self._preprocess_inputs(batch)
        
        results_dict = self.model(**batch)
        
        dataset = self._trainer.predict_dataloaders.dataset
        _id = dataset.int2id[batch_idx]
        
        preds = results_dict['logits']
        
        return preds, _id

    def on_before_optimizer_step(self, optimizer):
        """在优化器步骤之前进行梯度裁剪"""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), 
            self.trainer.gradient_clip_val
        )
        
        self.log('grad_norm', grad_norm)

    def load_model(self):
        """加载模型类"""
        try:
            if '_' in self.model_name:
                camel_name = ''.join([i.capitalize() for i in self.model_name.split('_')])
            else:
                camel_name = self.model_name
                
            if self.debug_mode:
                print(f"\n加载模型类：{self.model_name}")
                print(f"转换后的名称：{camel_name}")
            
            # 根据模型名称选择相应的导入路径
            if self.model_name == 'MFBP':
                print("加载MFBP模型...")
                Model = getattr(importlib.import_module(
                    f'model.MFBP.MFBP'), 'MFBP')
            else:
                Model = getattr(importlib.import_module(
                    f'model.{self.model_name.lower()}'), camel_name)
                
            if self.debug_mode:
                print("模型类加载成功")
                
            # 实例化模型
            model = self.instancialize(Model)
            
            if self.debug_mode:
                print("模型实例化成功")
                
            return model
            
        except Exception as e:
            print(f"加载模型时出错：{str(e)}")
            raise ValueError('Invalid Module File Name or Invalid Class Name!')

    def instancialize(self, Model, **other_args):
        try:
            # 获取模型初始化参数
            class_args = inspect.getfullargspec(Model.__init__).args[1:]
            inkeys = self.model_config.keys()
            args1 = {}
            
            # 从配置中获取参数
            for arg in class_args:
                if arg in inkeys:
                    args1[arg] = getattr(self.model_config, arg)
                elif arg == 'config':  # 如果需要config参数，传入完整配置
                    args1[arg] = self.config
                    
            # 添加其他参数
            args1.update(other_args)
            
            if self.debug_mode:
                print(f"模型参数：{args1}")
                
            # 实例化模型
            return Model(**args1)
            
        except Exception as e:
            print(f"模型实例化失败：{str(e)}")
            print(f"模型参数：{args1 if 'args1' in locals() else 'Not available'}")
            raise
    
    def on_fit_end(self):
        print("训练完成")

    def _compute_loss(self, outputs, batch):
        """简化的损失计算 - 只计算基因表达预测损失"""
        logits = outputs['logits']
        target_genes = batch['target_genes']
        
        # 确保维度匹配
        if logits.dim() != target_genes.dim():
            if logits.dim() == 3 and target_genes.dim() == 2:
                logits = logits.squeeze(1)
            elif logits.dim() == 2 and target_genes.dim() == 3:
                target_genes = target_genes.squeeze(1)
        
        # 计算MSE损失
        loss = self.criterion(logits, target_genes)
        
        self._debug_print(f"基因表达预测损失: {loss.item():.4f}")
        
        return loss
