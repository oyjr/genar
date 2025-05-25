import os
import inspect
import importlib
import logging
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


# 设置日志记录器
logger = logging.getLogger(__name__)


class ModelInterface(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.save_hyperparameters()

        self.model_name = config.MODEL.model_name if hasattr(config.MODEL, 'model_name') else None
        self.model_config = config.MODEL

        logger.debug(f"Model config: {self.model_config}")
        self.model = self.load_model()

        logger.debug(f"Model: {self.model}")

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
        logger.debug('Training step started')

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
        logger.debug(f"Validation step {batch_idx} started")
        
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
        self._log_tensor_shapes(batch, "Test batch")
        
        batch = self._preprocess_inputs(batch)

        results_dict = self.model(**batch)
        logits = results_dict['logits']
        target_genes = batch['target_genes']
        
        # 计算损失和指标
        loss = self._compute_loss(results_dict, batch)
        logger.debug(f"Test loss: {loss.item():.4f}")
        
        # 更新指标
        self._update_metrics('test', logits, target_genes)
        
        # 保存输出
        self._save_step_outputs('test', loss, logits, target_genes, batch_idx)
        
        return {'logits': logits, 'target_genes': target_genes}
    
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler with multi-GPU support.
        
        This method implements learning rate scaling strategies for multi-GPU training
        and sets up the AdamW optimizer with ReduceLROnPlateau scheduler.
        
        Returns:
            dict: Dictionary containing optimizer and lr_scheduler configurations
        """
        weight_decay = float(getattr(self.config.TRAINING, 'weight_decay', 0.0))
        learning_rate = float(self.config.TRAINING.learning_rate)
        
        # Apply learning rate scaling for multi-GPU training
        # When training with multiple GPUs, the effective batch size increases proportionally
        # Different scaling strategies help maintain training stability and convergence
        if hasattr(self.config, 'devices') and self.config.devices > 1:
            if hasattr(self.config, 'MULTI_GPU') and hasattr(self.config.MULTI_GPU, 'lr_scaling'):
                lr_scaling = self.config.MULTI_GPU.lr_scaling
                if lr_scaling == 'linear':
                    # Linear scaling: lr = base_lr * num_gpus
                    # Commonly used rule: scale learning rate linearly with batch size
                    learning_rate = learning_rate * self.config.devices
                    logger.info(f"多卡训练线性缩放学习率: {learning_rate} (原始: {self.config.TRAINING.learning_rate}, 设备数: {self.config.devices})")
                elif lr_scaling == 'sqrt':
                    # Square root scaling: lr = base_lr * sqrt(num_gpus)
                    # More conservative scaling, often used for very large batch sizes
                    learning_rate = learning_rate * (self.config.devices ** 0.5)
                    logger.info(f"多卡训练平方根缩放学习率: {learning_rate} (原始: {self.config.TRAINING.learning_rate}, 设备数: {self.config.devices})")
                else:
                    # No scaling: keep original learning rate
                    # Useful when batch size scaling is handled elsewhere or not needed
                    logger.info(f"多卡训练不缩放学习率: {learning_rate}")
        
        # Initialize AdamW optimizer with weight decay for regularization
        # AdamW decouples weight decay from gradient-based update, improving generalization
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Configure ReduceLROnPlateau scheduler for adaptive learning rate adjustment
        # Reduces learning rate when validation loss plateaus, helping fine-tune convergence
        lr_scheduler_config = self.config.TRAINING.lr_scheduler
        
        # Handle both dict and Namespace types for lr_scheduler configuration
        if isinstance(lr_scheduler_config, dict):
            factor = lr_scheduler_config.get('factor', 0.5)
            patience = lr_scheduler_config.get('patience', 5)
            mode = lr_scheduler_config.get('mode', 'min')
        else:
            factor = getattr(lr_scheduler_config, 'factor', 0.5)
            patience = getattr(lr_scheduler_config, 'patience', 5)
            mode = getattr(lr_scheduler_config, 'mode', 'min')
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.config.TRAINING.mode,  # 'min' for loss, 'max' for accuracy
                factor=factor,  # Reduction factor
                patience=patience,  # Epochs to wait before reduction
                verbose=True  # Log learning rate changes
            ),
            'monitor': self.config.TRAINING.monitor,  # Metric to monitor (e.g., 'val_loss')
            'interval': 'epoch',  # Check at the end of each epoch
            'frequency': 1  # Check every epoch
        }
        
        # Set gradient clipping value for training stability
        # Prevents exploding gradients by clipping gradient norms above threshold
        self.trainer.gradient_clip_val = getattr(self.config.TRAINING, 'gradient_clip_val', 1.0)
        
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
                logger.info(f"从基因列表文件获取基因数量: {num_outputs}")
            except FileNotFoundError:
                num_outputs = 200  # 默认值
                logger.warning(f"无法读取基因列表文件，使用默认基因数量: {num_outputs}")
            except Exception as e:
                num_outputs = 200  # 默认值
                logger.error(f"读取基因列表文件时出错: {e}，使用默认基因数量: {num_outputs}")
        else:
            num_outputs = 200  # 默认值
            logger.warning(f"配置中无数据路径，使用默认基因数量: {num_outputs}")

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
        """预处理输入数据，简化维度处理"""
        processed_inputs = {}
        
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # 统一处理：如果有多余的batch维度就移除
                if key in ['img', 'target_genes', 'positions'] and value.dim() > 2:
                    # 只在确实有多余维度时才squeeze
                    while value.dim() > 2 and value.size(0) == 1:
                        value = value.squeeze(0)
                processed_inputs[key] = value
            else:
                processed_inputs[key] = value
        
        return processed_inputs


    def _log_tensor_shapes(self, tensors_dict, prefix=""):
        """记录张量形状信息到日志"""
        if logger.isEnabledFor(logging.DEBUG):
            for name, tensor in tensors_dict.items():
                if isinstance(tensor, torch.Tensor):
                    logger.debug(f"{prefix}{name}: {tensor.shape}")

    

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
            logger.error(f"更新指标时发生错误: {e}")
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
                
            logger.debug(f"加载模型类：{self.model_name}")
            logger.debug(f"转换后的名称：{camel_name}")
            
            # 根据模型名称选择相应的导入路径
            if self.model_name == 'MFBP':
                logger.info("加载MFBP模型...")
                Model = getattr(importlib.import_module(
                    f'model.MFBP.MFBP'), 'MFBP')
            else:
                Model = getattr(importlib.import_module(
                    f'model.{self.model_name.lower()}'), camel_name)
                
            logger.debug("模型类加载成功")
                
            # 实例化模型
            model = self.instancialize(Model)
            
            logger.debug("模型实例化成功")
                
            return model
            
        except Exception as e:
            logger.error(f"加载模型时出错：{str(e)}")
            raise ValueError('Invalid Module File Name or Invalid Class Name!')

    def instancialize(self, Model, **other_args):
        try:
            # 获取模型初始化参数
            class_args = inspect.getfullargspec(Model.__init__).args[1:]
            
            # 处理model_config，支持Namespace和dict两种类型
            if hasattr(self.model_config, '__dict__'):
                # Namespace对象，转换为字典
                model_config_dict = vars(self.model_config)
                inkeys = model_config_dict.keys()
            else:
                # 字典对象
                model_config_dict = self.model_config
                inkeys = model_config_dict.keys()
            
            args1 = {}
            
            # 从配置中获取参数
            for arg in class_args:
                if arg in inkeys:
                    args1[arg] = model_config_dict[arg]
                elif arg == 'config':  # 如果需要config参数，传入完整配置
                    args1[arg] = self.config
                    
            # 添加其他参数
            args1.update(other_args)
            
            logger.debug(f"模型参数：{args1}")
                
            # 实例化模型
            return Model(**args1)
            
        except Exception as e:
            logger.error(f"模型实例化失败：{str(e)}")
            logger.error(f"模型参数：{args1 if 'args1' in locals() else 'Not available'}")
            raise
    
    def on_fit_end(self):
        logger.info("训练完成")

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
        
        logger.debug(f"基因表达预测损失: {loss.item():.4f}")
        
        return loss
