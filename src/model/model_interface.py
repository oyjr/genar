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
from datetime import datetime
from typing import Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter
import torch.nn.functional as F
from addict import Dict as AddictDict

# 设置日志记录器
logger = logging.getLogger(__name__)

# Import visualization module
try:
    # Try relative import first
    from ..visualization import GeneVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    try:
        # Fallback to absolute import
        from visualization import GeneVisualizer
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False
        logger.warning("Visualization module not available. Install matplotlib, seaborn, and PIL to enable visualization.")


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

        original_batch = batch.copy()  # 保存原始batch用于后处理
        batch = self._preprocess_inputs(batch)

        results_dict = self.model(**batch)
        
        # 如果是VAR_ST模型，需要后处理输出
        if hasattr(self, 'model_name') and self.model_name == 'VAR_ST':
            results_dict = self._postprocess_outputs_var_st(results_dict, original_batch)

        # 计算损失
        loss = self._compute_loss(results_dict, original_batch)

        # 获取预测和目标用于指标计算
        logits, target_genes = self._extract_predictions_and_targets(results_dict, original_batch)

        # 更新指标
        self._update_metrics('train', logits, target_genes)

        self.log('train_loss', loss, on_epoch=True, logger=True, sync_dist=True)
                
        # 🔇 Debug: 显示预测形状（简化输出，只在主进程显示）
        is_main_process = int(os.environ.get('LOCAL_RANK', 0)) == 0
        if is_main_process and hasattr(logits, 'shape'):
            pred_shape = logits.shape
            # 🔄 大幅减少输出频率：每100步输出一次
            if not hasattr(self, '_debug_step_count'):
                self._debug_step_count = 0
            self._debug_step_count += 1
            if self._debug_step_count % 100 == 0:
                print(f"🔄 单spot训练: {pred_shape} (Step {self._debug_step_count})")
        
        return loss


    def validation_step(self, batch, batch_idx):
        logger.debug(f"Validation step {batch_idx} started")
        
        original_batch = batch.copy()  # 保存原始batch用于后处理
        # 预处理输入
        batch = self._preprocess_inputs(batch)
        
        # 获取模型输出
        results_dict = self.model(**batch)
        
        # 如果是VAR_ST模型，需要后处理输出
        if hasattr(self, 'model_name') and self.model_name == 'VAR_ST':
            results_dict = self._postprocess_outputs_var_st(results_dict, original_batch)

        # 计算损失
        loss = self._compute_loss(results_dict, original_batch)
        
        # 获取预测和目标
        logits, target_genes = self._extract_predictions_and_targets(results_dict, original_batch)
        
        # 更新指标
        self._update_metrics('val', logits, target_genes)
        
        # 保存输出用于详细评估
        self._save_step_outputs('val', loss, logits, target_genes, batch_idx)
        
        # 记录损失
        self.log('val_loss', loss, on_epoch=True, logger=True, sync_dist=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        self._log_tensor_shapes(batch, "Test batch")
        
        original_batch = batch.copy()  # 保存原始batch用于后处理
        batch = self._preprocess_inputs(batch)

        results_dict = self.model(**batch)
        
        # 如果是VAR_ST模型，需要后处理输出
        if hasattr(self, 'model_name') and self.model_name == 'VAR_ST':
            results_dict = self._postprocess_outputs_var_st(results_dict, original_batch)
        
        # 获取预测和目标
        logits, target_genes = self._extract_predictions_and_targets(results_dict, original_batch)
        
        # 计算损失和指标
        loss = self._compute_loss(results_dict, original_batch)
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
        
        # Set gradient clipping value for training stability
        # Prevents exploding gradients by clipping gradient norms above threshold
        self.trainer.gradient_clip_val = getattr(self.config.TRAINING, 'gradient_clip_val', 1.0)
        
        # Check if learning rate scheduler should be disabled
        if patience == 0:
            logger.info("学习率调度器已禁用 (patience=0)，将使用固定学习率")
            return {'optimizer': optimizer}
        
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
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }
    

    def _init_metrics(self):
        # 检测是否为VAR_ST模型
        model_name = getattr(self.config, 'model_name', '') or getattr(self.config.MODEL, 'model_name', '') if hasattr(self.config, 'MODEL') else ''
        is_var_st = model_name.upper() == 'VAR_ST'
        
        if is_var_st:
            # VAR-ST模型使用196个基因
            num_outputs = 196
            logger.info(f"VAR_ST模型使用固定基因数量: {num_outputs}")
        else:
            # 其他模型从基因列表文件获取基因数量
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
        """
        预处理输入数据，使其与模型期望的格式匹配
        
        Args:
            inputs: 原始输入数据批次
            
        Returns:
            处理后的输入数据
        """
        
        # 记录输入张量的形状用于调试
        logger.debug("Original input shapes:")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                logger.debug(f"  {key}: {value.shape}")
        
        # 如果是VAR_ST模型，需要特殊的输入格式
        if hasattr(self, 'model_name') and self.model_name == 'VAR_ST':
            return self._preprocess_inputs_var_st(inputs)
        else:
            # 原有的MFBP输入格式
            return inputs

    def _preprocess_inputs_var_st(self, inputs):
        """
        VAR_ST模型的输入预处理（基因多尺度版本）
        
        关键变化：
        - 196基因 → 14×14伪图像
        - 基因多尺度 (1,2,3,4,5) → 55 tokens
        
        不需要positions参数，因为基因多尺度在基因维度而非空间维度
        """
        processed = {}
        
        # 基因表达数据处理 - 直接使用，不做维度转换
        if 'target_genes' in inputs:
            target_genes = inputs['target_genes']
            processed['gene_expression'] = target_genes
            # 🔇 简化输出：只在debug模式显示
            # print(f"📊 基因表达数据: {target_genes.shape}")
            
            # 验证维度
            if target_genes.dim() not in [2, 3]:
                raise ValueError(f"不支持的target_genes维度: {target_genes.shape}")
        
        # 组织学特征处理 - 直接使用，不做维度转换
        if 'img' in inputs:
            img_features = inputs['img']
            processed['histology_features'] = img_features
            # 🔇 简化输出：只在debug模式显示
            # print(f"🖼️  组织学特征: {img_features.shape}")
            
            # 验证维度
            if img_features.dim() not in [2, 3]:
                raise ValueError(f"不支持的img_features维度: {img_features.shape}")
        else:
            raise ValueError("Missing histology features ('img' key not found in inputs)")
        
        # 空间位置处理 - 基因多尺度模式下可选
        if 'positions' in inputs:
            processed['positions'] = inputs['positions']
            # 🔇 简化输出：只在debug模式显示
            # print(f"📍 空间位置: {inputs['positions'].shape} (基因模式下不使用)")
        
        # 设置模式
        processed['mode'] = 'training' if 'target_genes' in inputs else 'inference'
        
        # 🔇 大幅简化预处理完成输出
        # print(f"✅ VAR_ST预处理完成 (基因多尺度模式):")
        # print(f"   - 模式: {processed['mode']}")
        # for key, value in processed.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"   - {key}: {value.shape}")
        
        return processed


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
            'preds': preds.detach().cpu(),
            'targets': targets.detach().cpu(),
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

    def _compute_and_log_evaluation_metrics(self, phase):
        """计算并记录详细的评估指标"""
        outputs = getattr(self, f'{phase}_outputs')
        if len(outputs) == 0:
            return
        
        # 收集所有预测和目标
        all_preds = []
        all_targets = []
        
        for output in outputs:
            preds = output['preds']
            targets = output['targets']
            
            # 确保维度正确
            if preds.dim() == 3:
                preds = preds.reshape(-1, preds.size(-1))
            if targets.dim() == 3:
                targets = targets.reshape(-1, targets.size(-1))
                
            all_preds.append(preds)
            all_targets.append(targets)
        
        if len(all_preds) == 0:
            return
            
        # 合并所有批次的结果
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 计算评估指标
        metrics = self.calculate_evaluation_metrics(all_targets.numpy(), all_preds.numpy())
        
        # 打印结果
        self.print_evaluation_results(metrics, prefix=phase.capitalize())
        
        # 记录到tensorboard
        for key, value in metrics.items():
            if key != 'correlations':  # 不记录相关性数组
                self.log(f'{phase}_detailed_{key.replace("-", "_")}', value, logger=True)
        
        # 保存到文件 - 每10个epoch保存一次，或者是最后一个epoch
        save_metrics = (self.current_epoch % 10 == 0) or (self.current_epoch == self.trainer.max_epochs - 1)
        
        if save_metrics:
            if hasattr(self.config, 'GENERAL') and hasattr(self.config.GENERAL, 'log_path'):
                log_dir = self.config.GENERAL.log_path
            else:
                log_dir = './logs'
                
            # 创建保存路径
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = os.path.join(log_dir, 'evaluation_results', 
                                    f'{phase}_metrics_epoch_{self.current_epoch}_{timestamp}.txt')
            
            self.save_evaluation_results(metrics, save_path, 
                                       slide_id=f"epoch_{self.current_epoch}", 
                                       model_name="MFBP")
            
            logger.info(f"{phase.capitalize()} 评估指标已保存到: {save_path}")
        else:
            logger.debug(f"{phase.capitalize()} epoch {self.current_epoch}: 评估指标计算完成，跳过文件保存")
        
        # 注意：可视化现在只在训练完成后生成，不在每个epoch生成
        # 这样可以避免产生大量中间可视化文件

    def on_train_epoch_end(self):
        self._process_epoch_end('train')
        
    def on_validation_epoch_end(self):
        self._compute_and_log_evaluation_metrics('val')
        # 只有在非最后一个epoch时才清空数据，保留最后一个epoch的数据用于可视化
        if self.current_epoch < self.trainer.max_epochs - 1:
            self._process_epoch_end('val')
        
    def on_test_epoch_end(self):
        self._compute_and_log_evaluation_metrics('test')
        # 只有在非最后一个epoch时才清空数据，保留最后一个epoch的数据用于可视化
        if self.current_epoch < self.trainer.max_epochs - 1:
            self._process_epoch_end('test')
    
    def on_fit_end(self):
        """训练完成时的回调 - 生成最终可视化"""
        # 多GPU环境下只在主进程（rank 0）执行可视化
        if self.trainer.is_global_zero:
            print("=" * 60)
            print("🎉 训练完成！开始生成最终可视化...")
            print("=" * 60)
            logger.info("训练完成，开始生成最终可视化...")
            logger.info(f"验证数据输出数量: {len(self.val_outputs)}")
            logger.info(f"测试数据输出数量: {len(self.test_outputs)}")
            print(f"📊 验证数据输出数量: {len(self.val_outputs)}")
            print(f"📊 测试数据输出数量: {len(self.test_outputs)}")
            
            # 智能获取可视化设置
            enable_vis = self._get_visualization_setting()
            print(f"🔍 enable_visualization: {enable_vis}")
            print(f"🔍 VISUALIZATION_AVAILABLE: {VISUALIZATION_AVAILABLE}")
            
            if enable_vis:
                try:
                    # 如果有验证数据，使用验证数据生成可视化
                    if len(self.val_outputs) > 0:
                        print("🎨 开始使用验证数据生成最终可视化...")
                        logger.info("使用验证数据生成最终可视化...")
                        self._generate_final_visualization('val')
                        print("🎨 验证数据可视化完成")
                    elif len(self.test_outputs) > 0:
                        print("🎨 开始使用测试数据生成最终可视化...")
                        logger.info("使用测试数据生成最终可视化...")
                        self._generate_final_visualization('test')
                        print("🎨 测试数据可视化完成")
                    else:
                        print("❌ 没有可用的验证或测试数据用于生成可视化")
                        logger.warning("没有可用的验证或测试数据用于生成可视化")
                        
                except Exception as e:
                    print(f"❌ 可视化生成异常: {e}")
                    logger.error(f"最终可视化生成失败: {e}")
                    import traceback
                    traceback.print_exc()
                    logger.warning("训练已完成，但跳过可视化生成")
            else:
                print("❌ 可视化已禁用")
                logger.info("可视化已禁用，跳过可视化生成")
            
            logger.info("训练和可视化生成完成")
        else:
            # 非主进程只记录信息
            logger.info(f"GPU进程 {self.trainer.global_rank}: 训练完成，跳过可视化生成（只在主进程生成）")

    def _get_visualization_setting(self):
        """智能获取可视化设置"""
        # 尝试多个可能的配置位置
        possible_paths = [
            'enable_visualization',
            'GENERAL.enable_visualization', 
            'TRAINING.enable_visualization',
            'visualization.enable',
            'vis_enable'
        ]
        
        for attr_path in possible_paths:
            try:
                value = self.config
                for part in attr_path.split('.'):
                    value = getattr(value, part)
                # 如果找到了布尔值，直接返回
                if isinstance(value, bool):
                    logger.info(f"Found visualization setting at {attr_path}: {value}")
                    return value
                # 如果是字符串，尝试转换
                elif isinstance(value, str):
                    if value.lower() in ['true', '1', 'yes', 'on']:
                        logger.info(f"Found visualization setting at {attr_path}: {value} -> True")
                        return True
                    elif value.lower() in ['false', '0', 'no', 'off']:
                        logger.info(f"Found visualization setting at {attr_path}: {value} -> False")
                        return False
            except AttributeError:
                continue
        
        # 检查命令行参数或环境变量
        if hasattr(self.config, '__dict__'):
            config_dict = vars(self.config)
            logger.debug(f"Config attributes: {list(config_dict.keys())}")
            
            # 查找任何包含 'visual' 的属性
            for key, value in config_dict.items():
                if 'visual' in key.lower():
                    logger.info(f"Found visualization-related config: {key} = {value}")
                    if isinstance(value, bool):
                        return value
        
        # 默认启用可视化
        logger.info("No explicit visualization setting found, defaulting to True")
        return True

    def _load_gene_names(self):
        """加载基因名称列表"""
        try:
            # 尝试从配置的数据路径加载基因列表
            if hasattr(self.config, 'data_path'):
                gene_file = f"{self.config.data_path}processed_data/selected_gene_list.txt"
                if os.path.exists(gene_file):
                    with open(gene_file, 'r') as f:
                        gene_names = [line.strip() for line in f.readlines() if line.strip()]
                    logger.info(f"Loaded {len(gene_names)} gene names from {gene_file}")
                    return gene_names
            
            # 如果基因列表文件不存在，尝试从训练器的数据模块获取
            if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'gene_names'):
                gene_names = self.trainer.datamodule.gene_names
                logger.info(f"Loaded {len(gene_names)} gene names from datamodule")
                return gene_names
                
            logger.warning("Could not load gene names, spatial visualization may be limited")
            return None
            
        except Exception as e:
            logger.error(f"Error loading gene names: {e}")
            return None

    def _load_adata_for_visualization(self, phase):
        """加载用于可视化的AnnData对象，同时返回对应的slide_id"""
        try:
            # 尝试从trainer的数据模块获取相应阶段的数据集
            if hasattr(self.trainer, 'datamodule'):
                datamodule = self.trainer.datamodule
                
                # 根据阶段选择相应的数据集
                if phase == 'val' and hasattr(datamodule, 'val_dataloader'):
                    dataset = datamodule.val_dataloader().dataset
                elif phase == 'test' and hasattr(datamodule, 'test_dataloader'):
                    dataset = datamodule.test_dataloader().dataset
                else:
                    logger.warning(f"No {phase} dataloader found")
                    return None, None
                
                # 方法1：尝试获取预存储的AnnData对象
                if hasattr(dataset, 'adata'):
                    adata = dataset.adata
                    # 尝试获取对应的slide_id
                    slide_id = dataset.ids[0] if hasattr(dataset, 'ids') and len(dataset.ids) > 0 else 'unknown_slide'
                    logger.info(f"Loaded AnnData for {phase} phase with {adata.n_obs} spots from slide: {slide_id}")
                    return adata, slide_id
                elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'adata'):
                    adata = dataset.dataset.dataset.adata
                    # 尝试获取对应的slide_id
                    slide_id = dataset.dataset.ids[0] if hasattr(dataset.dataset, 'ids') and len(dataset.dataset.ids) > 0 else 'unknown_slide'
                    logger.info(f"Loaded AnnData for {phase} phase with {adata.n_obs} spots from slide: {slide_id}")
                    return adata, slide_id
                
                # 方法2：如果没有预存储的，尝试动态加载（像eval模式那样）
                elif hasattr(dataset, 'load_st') and hasattr(dataset, 'ids'):
                    # 获取第一个slide的ID（用于可视化）
                    if len(dataset.ids) > 0:
                        slide_id = dataset.ids[0]  # 取第一个slide用于可视化
                        logger.info(f"Dynamically loading AnnData for slide: {slide_id}")
                        
                        # 使用数据集的load_st方法动态加载
                        adata = dataset.load_st(slide_id, dataset.genes if hasattr(dataset, 'genes') else None)
                        logger.info(f"Dynamically loaded AnnData for {phase} phase with {adata.n_obs} spots from slide: {slide_id}")
                        return adata, slide_id
                    else:
                        logger.warning(f"No slides found in {phase} dataset")
                        return None, None
                
                # 方法3：如果是包装类，尝试深度查找
                else:
                    logger.warning(f"Trying to find AnnData in nested dataset structure...")
                    current_dataset = dataset
                    for i in range(3):  # 最多查找3层
                        if hasattr(current_dataset, 'dataset'):
                            current_dataset = current_dataset.dataset
                            if hasattr(current_dataset, 'adata'):
                                adata = current_dataset.adata
                                slide_id = current_dataset.ids[0] if hasattr(current_dataset, 'ids') and len(current_dataset.ids) > 0 else 'unknown_slide'
                                logger.info(f"Found AnnData at depth {i+1} for {phase} phase with {adata.n_obs} spots from slide: {slide_id}")
                                return adata, slide_id
                            elif hasattr(current_dataset, 'load_st') and hasattr(current_dataset, 'ids'):
                                if len(current_dataset.ids) > 0:
                                    slide_id = current_dataset.ids[0]
                                    logger.info(f"Dynamically loading AnnData for slide: {slide_id} at depth {i+1}")
                                    adata = current_dataset.load_st(slide_id, getattr(current_dataset, 'genes', None))
                                    logger.info(f"Dynamically loaded AnnData for {phase} phase with {adata.n_obs} spots from slide: {slide_id}")
                                    return adata, slide_id
                        else:
                            break
                    
                    logger.warning(f"No AnnData object found in {phase} dataset after deep search")
                    return None, None
            else:
                logger.warning("No datamodule found in trainer")
                return None, None
                
        except Exception as e:
            logger.error(f"Error loading AnnData for visualization: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _generate_final_visualization(self, phase):
        """生成最终的可视化报告"""
        print(f"📊 _generate_final_visualization called with phase: {phase}")
        outputs = getattr(self, f'{phase}_outputs')
        print(f"📊 Found {len(outputs)} outputs for {phase}")
        if len(outputs) == 0:
            print(f"❌ 没有{phase}数据用于生成可视化")
            logger.warning(f"没有{phase}数据用于生成可视化")
            return
        
        print(f"🎨 开始处理{phase}阶段的最终可视化...")
        logger.info(f"开始生成{phase}阶段的最终可视化...")
        
        # 获取AnnData对象和对应的slide_id用于空间可视化
        adata, slide_id = self._load_adata_for_visualization(phase)
        
        # 收集所有预测和目标
        all_preds = []
        all_targets = []
        
        for output in outputs:
            preds = output['preds']
            targets = output['targets']
            
            # 确保维度正确
            if preds.dim() == 3:
                preds = preds.reshape(-1, preds.size(-1))
            if targets.dim() == 3:
                targets = targets.reshape(-1, targets.size(-1))
                
            all_preds.append(preds)
            all_targets.append(targets)
        
        if len(all_preds) == 0:
            logger.warning(f"没有有效的{phase}预测数据")
            return
            
        # 合并所有批次的结果
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # 如果有AnnData，确保预测数据与空间坐标维度匹配
        if adata is not None:
            n_spots = adata.n_obs
            print(f"🔍 AnnData spots: {n_spots}, Prediction spots: {all_preds.shape[0]}")
            
            # 如果预测数据比空间点多，只取前n_spots个（通常是第一个slide的数据）
            if all_preds.shape[0] > n_spots:
                print(f"📏 Truncating prediction data from {all_preds.shape[0]} to {n_spots} to match spatial coordinates")
                all_preds = all_preds[:n_spots]
                all_targets = all_targets[:n_spots]
            elif all_preds.shape[0] < n_spots:
                print(f"⚠️ Warning: Prediction data ({all_preds.shape[0]}) is less than spatial coordinates ({n_spots})")
        
        # 计算评估指标
        metrics = self.calculate_evaluation_metrics(all_targets.numpy(), all_preds.numpy())
        
        try:
            # 获取数据集名称和标记基因
            dataset_name = getattr(self.config, 'expr_name', 'default')
            marker_genes = self.get_marker_genes_for_dataset(dataset_name)
            
            # 获取基因名称列表
            gene_names = self._load_gene_names()
            
            print(f"🧬 Dataset: {dataset_name}")
            print(f"🎯 Marker genes: {marker_genes}")
            print(f"📝 Gene names loaded: {len(gene_names) if gene_names else 0}")
            print(f"🗺️ AnnData available: {adata is not None}")
            
            # 创建最终可视化
            self.create_visualizations(
                phase=f"{phase}_final",  # 添加"final"标识
                y_true=all_targets.numpy(),
                y_pred=all_preds.numpy(),
                metrics=metrics,
                gene_names=gene_names,  # 从配置中加载的基因名称
                marker_genes=marker_genes,
                adata=adata,  # 从数据集加载的AnnData对象
                slide_id=slide_id,  # 从数据集获取的实际slide_id
                img_path=None  # 如果需要可以配置
            )
            
            logger.info(f"{phase}阶段最终可视化生成完成")
            
        except Exception as e:
            logger.error(f"生成{phase}最终可视化时出错: {e}")
            import traceback
            traceback.print_exc()

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
            elif self.model_name == 'VAR_ST':
                logger.info("加载VAR_ST模型...")
                Model = getattr(importlib.import_module(
                    f'model.VAR.VAR_ST_Complete'), 'VAR_ST_Complete')
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
            
            # 🔧 修复：正确处理addict.Dict对象
            if isinstance(self.model_config, AddictDict):
                # 对于addict.Dict，直接使用dict()转换
                model_config_dict = dict(self.model_config)
                inkeys = model_config_dict.keys()
            elif hasattr(self.model_config, '__dict__'):
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
    


    def _compute_loss(self, outputs, batch):
        """
        计算损失 - 支持不同模型的损失计算方式
        
        Args:
            outputs: 模型输出
            batch: 输入批次数据
            
        Returns:
            loss: 计算得到的损失值
        """
        
        # 如果是VAR_ST模型，使用其特殊的损失计算
        if hasattr(self, 'model_name') and self.model_name == 'VAR_ST':
            return self._compute_loss_var_st(outputs, batch)
        else:
            # 原有的MFBP损失计算
            return self._compute_loss_mfbp(outputs, batch)

    def _compute_loss_var_st(self, outputs, batch):
        """
        VAR_ST模型的损失计算
        
        VAR_ST返回的输出包含多个损失组件：
        - loss: 总损失 (已经在模型内部计算好)
        - vq_loss: VQ量化损失
        - recon_loss: 重建损失  
        - ar_loss: 自回归损失
        - spot_recon_loss: spots重建损失
        """
        if 'loss' in outputs:
            # 如果模型已经计算好总损失，直接使用
            total_loss = outputs['loss']
            
            # 记录各个损失组件用于监控
            if 'vq_loss' in outputs:
                self.log('train_vq_loss', outputs['vq_loss'], on_epoch=True, logger=True, sync_dist=True)
            if 'recon_loss' in outputs:
                self.log('train_recon_loss', outputs['recon_loss'], on_epoch=True, logger=True, sync_dist=True)
            if 'ar_loss' in outputs:
                self.log('train_ar_loss', outputs['ar_loss'], on_epoch=True, logger=True, sync_dist=True)
            if 'spot_recon_loss' in outputs:
                self.log('train_spot_recon_loss', outputs['spot_recon_loss'], on_epoch=True, logger=True, sync_dist=True)
            
            logger.debug(f"VAR_ST总损失: {total_loss.item():.4f}")
            
            return total_loss
        else:
            # 如果模型没有返回损失，手动计算
            if 'predicted_expression' in outputs and 'gene_expression' in batch:
                logits = outputs['predicted_expression']
                target_genes = batch['gene_expression']
                loss = self.criterion(logits, target_genes)
                logger.debug(f"VAR_ST预测损失: {loss.item():.4f}")
                return loss
            else:
                raise ValueError("VAR_ST模型输出格式不正确，缺少损失信息")

    def _compute_loss_mfbp(self, outputs, batch):
        """原有的MFBP损失计算"""
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
        
        logger.debug(f"MFBP基因表达预测损失: {loss.item():.4f}")
        
        return loss

    def calculate_gene_correlations(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate gene-wise Pearson correlation coefficients.
        
        Args:
            y_true: Ground truth gene expression [num_spots, num_genes]
            y_pred: Predicted gene expression [num_spots, num_genes]
            
        Returns:
            Array of correlation coefficients for each gene [num_genes]
        """
        num_genes = y_true.shape[1]
        correlations = []
        
        for i in range(num_genes):
            # Extract gene expression for all spots
            true_gene = y_true[:, i]
            pred_gene = y_pred[:, i]
            
            # Calculate Pearson correlation
            if np.std(true_gene) == 0 or np.std(pred_gene) == 0:
                # Handle constant values (no variation)
                corr = 0.0
            else:
                corr = np.corrcoef(true_gene, pred_gene)[0, 1]
                # Handle NaN values
                if np.isnan(corr):
                    corr = 0.0
            
            correlations.append(corr)
        
        return np.array(correlations)

    def calculate_evaluation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Calculate comprehensive evaluation metrics for spatial transcriptomics.
        
        Args:
            y_true: Ground truth gene expression [num_spots, num_genes]
            y_pred: Predicted gene expression [num_spots, num_genes]
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        # Ensure inputs are numpy arrays
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        # Calculate gene-wise correlations
        correlations = self.calculate_gene_correlations(y_true, y_pred)
        
        # Sort correlations in descending order for PCC metrics
        sorted_corr = np.sort(correlations)[::-1]
        
        # Calculate PCC metrics (top-k correlations)
        pcc_10 = np.mean(sorted_corr[:10]) if len(sorted_corr) >= 10 else np.mean(sorted_corr)
        pcc_50 = np.mean(sorted_corr[:50]) if len(sorted_corr) >= 50 else np.mean(sorted_corr)
        pcc_200 = np.mean(sorted_corr[:200]) if len(sorted_corr) >= 200 else np.mean(sorted_corr)
        
        # Calculate MSE and MAE
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Calculate RVD (Relative Variance Difference)
        pred_var = np.var(y_pred, axis=0)  # Variance across spots for each gene
        true_var = np.var(y_true, axis=0)  # Variance across spots for each gene
        
        # Avoid division by zero
        valid_mask = true_var > 1e-8
        if np.sum(valid_mask) > 0:
            rvd = np.mean(((pred_var[valid_mask] - true_var[valid_mask]) ** 2) / (true_var[valid_mask] ** 2))
        else:
            rvd = 0.0
        
        return {
            'PCC-10': float(pcc_10),
            'PCC-50': float(pcc_50), 
            'PCC-200': float(pcc_200),
            'MSE': float(mse),
            'MAE': float(mae),
            'RVD': float(rvd),
            'correlations': correlations  # Keep for detailed analysis
        }

    def print_evaluation_results(self, metrics: dict, prefix: str = "") -> None:
        """
        Print evaluation metrics in a formatted way.
        
        Args:
            metrics: Dictionary containing evaluation metrics
            prefix: Optional prefix for the output (e.g., "Val", "Test")
        """
        # 🔧 在分布式训练中，只在主进程输出评估结果
        import os
        is_main_process = int(os.environ.get('LOCAL_RANK', 0)) == 0
        
        if not is_main_process:
            return  # 非主进程直接返回，不输出
        
        if prefix:
            print(f"\n========== {prefix} 评估结果 ==========")
        else:
            print(f"\n========== 评估结果 ==========")
        
        print(f"PCC-10: {metrics['PCC-10']:.4f}")
        print(f"PCC-50: {metrics['PCC-50']:.4f}")
        print(f"PCC-200: {metrics['PCC-200']:.4f}")
        print(f"MSE: {metrics['MSE']:.4f}")
        print(f"MAE: {metrics['MAE']:.4f}")
        print(f"RVD: {metrics['RVD']:.4f}")

    def save_evaluation_results(self, metrics: dict, save_path: str, 
                               slide_id: str = "", model_name: str = "MFBP") -> None:
        """
        Save evaluation metrics to file.
        
        Args:
            metrics: Dictionary containing evaluation metrics
            save_path: Path to save the results
            slide_id: Optional slide identifier
            model_name: Optional model name
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            f.write(f"Model: {model_name}\n")
            if slide_id:
                f.write(f"Slide: {slide_id}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            f.write(f"PCC-10: {metrics['PCC-10']:.4f}\n")
            f.write(f"PCC-50: {metrics['PCC-50']:.4f}\n")
            f.write(f"PCC-200: {metrics['PCC-200']:.4f}\n")
            f.write(f"MSE: {metrics['MSE']:.4f}\n")
            f.write(f"MAE: {metrics['MAE']:.4f}\n")
            f.write(f"RVD: {metrics['RVD']:.4f}\n")

    def create_visualizations(self, phase: str, y_true: np.ndarray, y_pred: np.ndarray, 
                            metrics: dict, gene_names: list = None, 
                            marker_genes: list = None, adata=None, slide_id: str = None, img_path: str = None) -> None:
        """
        Create comprehensive visualizations for model evaluation.
        
        Args:
            phase: Training phase ('val', 'test', etc.)
            y_true: Ground truth gene expression [num_spots, num_genes]
            y_pred: Predicted gene expression [num_spots, num_genes]
            metrics: Dictionary containing evaluation metrics
            gene_names: Optional list of gene names
            marker_genes: Optional list of marker genes for spatial visualization
            adata: Optional AnnData object for spatial coordinates
            img_path: Optional path to tissue image
        """
        if not VISUALIZATION_AVAILABLE:
            logger.warning("Visualization module not available. Skipping visualization creation.")
            return
        
        try:
            # Create visualization directory
            if hasattr(self.config, 'GENERAL') and hasattr(self.config.GENERAL, 'log_path'):
                log_dir = self.config.GENERAL.log_path
            else:
                log_dir = './logs'
            
            # 处理不同的阶段格式
            if phase.endswith('_final'):
                vis_dir = os.path.join(log_dir, 'vis', phase)
            else:
                vis_dir = os.path.join(log_dir, 'vis', f'{phase}_epoch_{self.current_epoch}')
            
            # Initialize visualizer
            visualizer = GeneVisualizer(save_dir=vis_dir)
            
            # 1. Create gene variation curves
            logger.info(f"Creating gene variation curves for {phase}...")
            visualizer.plot_gene_variation_curves(
                y_true, y_pred, 
                save_name=f"{phase}_gene_variation_curves",
                show_plots=False
            )
            
            # 2. Create correlation analysis
            logger.info(f"Creating correlation analysis for {phase}...")
            correlations = visualizer.plot_correlation_analysis(
                y_true, y_pred,
                gene_names=gene_names,
                save_name=f"{phase}_correlation_analysis", 
                show_plots=False
            )
            
            # 3. Create spatial gene expression maps (if data available)
            if marker_genes and adata is not None:
                logger.info(f"Creating spatial gene expression maps for {phase}...")
                # Get dataset path from config
                data_path = getattr(self.config, 'data_path', '')
                
                # Use the passed slide_id if available, otherwise fall back to config
                if slide_id is None:
                    slide_id = getattr(self.config, 'slide_test', 'unknown_slide')
                    if phase == 'val':
                        slide_id = getattr(self.config, 'slide_val', slide_id)
                
                logger.info(f"Using slide_id for WSI visualization: {slide_id}")
                
                visualizer.plot_spatial_gene_expression(
                    adata, y_pred, gene_names or [], marker_genes,
                    data_path=data_path,
                    slide_id=slide_id,
                    save_name=f"{phase}_spatial_expression",
                    show_plots=False
                )
            
            # 4. Create summary report
            logger.info(f"Creating summary report for {phase}...")
            visualizer.create_summary_report(
                metrics, correlations,
                save_name=f"{phase}_summary_report"
            )
            
            logger.info(f"All visualizations for {phase} saved to: {vis_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations for {phase}: {e}")
            logger.error(f"Visualization will be skipped for this epoch.")

    def get_marker_genes_for_dataset(self, dataset_name: str) -> list:
        """
        Get default marker genes for different datasets.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of marker gene names
        """
        # Default marker genes for different datasets
        marker_genes_dict = {
            'PRAD': ['KLK3', 'AR', 'FOLH1', 'ACPP', 'KLK2', 'STEAP2', 'PSMA', 'NKX3-1'],
            'her2st': ['ERBB2', 'ESR1', 'PGR', 'MKI67', 'TP53', 'BRCA1', 'BRCA2'],
            'default': ['CD3E', 'CD4', 'CD8A', 'CD19', 'CD68', 'PTPRC', 'VIM', 'KRT19']
        }
        
        dataset_key = dataset_name.upper() if dataset_name else 'default'
        return marker_genes_dict.get(dataset_key, marker_genes_dict['default'])

    def _extract_predictions_and_targets(self, results_dict, batch):
        """
        从模型输出和批次数据中提取预测和目标
        
        Args:
            results_dict: 模型输出
            batch: 输入批次数据
            
        Returns:
            tuple: (logits, target_genes)
        """
        if hasattr(self, 'model_name') and self.model_name == 'VAR_ST':
            # VAR_ST模型输出格式
            if 'predicted_expression' in results_dict:
                logits = results_dict['predicted_expression']
            elif 'logits' in results_dict:
                logits = results_dict['logits']
            else:
                raise ValueError("VAR_ST模型输出中找不到predicted_expression或logits")
            
            # 目标数据
            if 'gene_expression' in batch:
                target_genes = batch['gene_expression']
            elif 'target_genes' in batch:
                target_genes = batch['target_genes']
            else:
                raise ValueError("批次数据中找不到gene_expression或target_genes")
        else:
            # MFBP模型输出格式
            logits = results_dict['logits']
            target_genes = batch['target_genes']
            
            # 确保维度匹配
            if logits.dim() != target_genes.dim():
                if logits.dim() == 3 and target_genes.dim() == 2:
                    logits = logits.squeeze(1)
                elif logits.dim() == 2 and target_genes.dim() == 3:
                    target_genes = target_genes.squeeze(1)
        
        return logits, target_genes

    def _postprocess_outputs_var_st(self, outputs, original_inputs):
        """
        为VAR_ST模型后处理输出数据 - 基因维度多尺度模式
        
        支持单spot和多spot输出：
        - 单spot: [B, num_genes] 
        - 多spot: [B, N, num_genes]
        """
        processed = {}
        
        if 'target_genes' in original_inputs:
            # 训练模式：直接使用模型输出
            processed['loss'] = outputs['loss']
            processed['var_loss'] = outputs.get('var_loss', outputs['loss'])
            processed['vqvae_loss'] = outputs.get('vqvae_loss', torch.tensor(0.0, requires_grad=True))
            processed['gene_recon_loss'] = outputs.get('gene_recon_loss', torch.tensor(0.0, requires_grad=True))
            
            # 预测和目标数据 - 确保所有必要字段都被传递
            processed['predictions'] = outputs.get('predictions', outputs.get('predicted_expression'))
            processed['predicted_expression'] = outputs.get('predicted_expression', outputs.get('predictions'))
            processed['logits'] = outputs.get('logits', outputs.get('predicted_expression', outputs.get('predictions')))
            processed['targets'] = outputs.get('targets', original_inputs['target_genes'])
            
            # 只在每100个step时显示详细信息，其余时候保持简洁
            if not hasattr(self, '_step_count'):
                self._step_count = 0
            self._step_count += 1
            
            # 🔧 在分布式训练中，只在主进程输出，并大幅减少输出频率
            import os
            is_main_process = int(os.environ.get('LOCAL_RANK', 0)) == 0
            
            if processed['predictions'] is not None and is_main_process and self._step_count % 500 == 1:
                pred_shape = processed['predictions'].shape
                if len(pred_shape) == 2:
                    print(f"🔄 单spot训练: {pred_shape} (Step {self._step_count})")
                elif len(pred_shape) == 3:
                    B, N, G = pred_shape
                    print(f"🔄 多spot训练: Batch={B}, Spots={N}, Genes={G} (Step {self._step_count})")
        else:
            # 推理模式：处理预测输出
            processed['predictions'] = outputs.get('predictions', outputs.get('predicted_expression', outputs.get('logits')))
            processed['predicted_expression'] = processed['predictions']
            processed['logits'] = processed['predictions']
        
        return processed

    def test_full_slide(self, slide_data: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        对整个slide进行测试，逐spot预测后整合结果
        
        Args:
            slide_data: 完整slide数据，包含：
                - img: [num_spots, feature_dim]
                - target_genes: [num_spots, num_genes]
                - positions: [num_spots, 2]
                - slide_id: str
                - num_spots: int
                
        Returns:
            包含预测结果和评价指标的字典
        """
        self.eval()  # 设置为评估模式
        
        # 获取slide信息
        slide_id = slide_data['slide_id']
        num_spots = slide_data['num_spots']
        
        print(f"🔬 开始测试slide: {slide_id}，共{num_spots}个spots")
        logger.info(f"Testing full slide: {slide_id} with {num_spots} spots")
        
        # 准备结果容器
        all_predictions = []
        all_targets = []
        
        # 逐spot进行预测
        with torch.no_grad():
            for spot_idx in range(num_spots):
                # 构造单个spot的batch数据
                single_spot_batch = {
                    'img': slide_data['img'][spot_idx:spot_idx+1],  # [1, feature_dim]
                    'target_genes': slide_data['target_genes'][spot_idx:spot_idx+1],  # [1, num_genes]
                    'positions': slide_data['positions'][spot_idx:spot_idx+1],  # [1, 2]
                    'slide_id': slide_id,
                    'spot_idx': spot_idx
                }
                
                # 移动到正确的设备
                for key, value in single_spot_batch.items():
                    if isinstance(value, torch.Tensor):
                        single_spot_batch[key] = value.to(self.device)
                
                # 预处理输入
                processed_batch = self._preprocess_inputs(single_spot_batch)
                
                # 模型预测
                results_dict = self.model(**processed_batch)
                
                # 后处理输出（如果是VAR_ST）
                if hasattr(self, 'model_name') and self.model_name == 'VAR_ST':
                    results_dict = self._postprocess_outputs_var_st(results_dict, single_spot_batch)
                
                # 提取预测和目标
                prediction, target = self._extract_predictions_and_targets(results_dict, single_spot_batch)
                
                # 收集结果
                all_predictions.append(prediction.cpu().numpy())
                all_targets.append(target.cpu().numpy())
                
                # 每100个spot显示一次进度
                if (spot_idx + 1) % 100 == 0 or spot_idx == num_spots - 1:
                    print(f"  📈 已处理 {spot_idx + 1}/{num_spots} spots")
        
        # 整合所有预测结果
        predictions_array = np.vstack(all_predictions)  # [num_spots, num_genes]
        targets_array = np.vstack(all_targets)  # [num_spots, num_genes]
        
        print(f"✅ Slide {slide_id} 测试完成")
        print(f"   预测结果形状: {predictions_array.shape}")
        print(f"   目标数据形状: {targets_array.shape}")
        
        # 计算完整的评价指标
        metrics = self.calculate_evaluation_metrics(targets_array, predictions_array)
        
        # 打印评价结果
        self.print_evaluation_results(metrics, f"Slide {slide_id}")
        
        return {
            'predictions': predictions_array,
            'targets': targets_array,
            'metrics': metrics,
            'slide_id': slide_id,
            'num_spots': num_spots
        }

    def run_full_slide_testing(self) -> Dict[str, Any]:
        """
        运行完整的slide测试流程
        
        Returns:
            包含所有slide测试结果的字典
        """
        print("🎯 开始整slide测试模式...")
        logger.info("Starting full slide testing mode...")
        
        # 获取测试数据集
        if not hasattr(self.trainer, 'datamodule'):
            raise ValueError("No datamodule found in trainer")
        
        datamodule = self.trainer.datamodule
        if not hasattr(datamodule, 'test_dataloader'):
            raise ValueError("No test dataloader found")
        
        test_dataset = datamodule.test_dataloader().dataset
        
        # 获取原始dataset（可能被包装了）
        original_dataset = test_dataset
        while hasattr(original_dataset, 'dataset'):
            original_dataset = original_dataset.dataset
        
        # 获取测试slide列表
        test_slide_ids = original_dataset.get_test_slide_ids()
        
        if not test_slide_ids:
            raise ValueError("No test slides found")
        
        print(f"📋 找到 {len(test_slide_ids)} 个测试slides: {test_slide_ids}")
        
        # 存储所有slide的结果
        all_slide_results = {}
        aggregated_predictions = []
        aggregated_targets = []
        
        # 逐个测试每个slide
        for slide_id in test_slide_ids:
            print(f"\n{'='*60}")
            print(f"🔬 测试Slide: {slide_id}")
            print(f"{'='*60}")
            
            # 获取完整slide数据
            slide_data = original_dataset.get_full_slide_for_testing(slide_id)
            
            # 进行测试
            slide_results = self.test_full_slide(slide_data)
            
            # 保存结果
            all_slide_results[slide_id] = slide_results
            
            # 累积所有数据用于总体评估
            aggregated_predictions.append(slide_results['predictions'])
            aggregated_targets.append(slide_results['targets'])
            
            # 保存单个slide的结果
            if hasattr(self.config, 'GENERAL') and hasattr(self.config.GENERAL, 'log_path'):
                save_dir = os.path.join(self.config.GENERAL.log_path, 'test_results')
            else:
                save_dir = './logs/test_results'
            
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{slide_id}_results.txt")
            self.save_evaluation_results(slide_results['metrics'], save_path, slide_id, "VAR_ST")
        
        # 计算所有slide的聚合结果
        print(f"\n{'='*60}")
        print("📊 计算聚合评价指标...")
        print(f"{'='*60}")
        
        all_predictions = np.vstack(aggregated_predictions)
        all_targets = np.vstack(aggregated_targets)
        
        overall_metrics = self.calculate_evaluation_metrics(all_targets, all_predictions)
        self.print_evaluation_results(overall_metrics, "整体测试结果")
        
        # 保存整体结果
        if hasattr(self.config, 'GENERAL') and hasattr(self.config.GENERAL, 'log_path'):
            save_dir = os.path.join(self.config.GENERAL.log_path, 'test_results')
        else:
            save_dir = './logs/test_results'
        
        save_path = os.path.join(save_dir, "overall_results.txt")
        self.save_evaluation_results(overall_metrics, save_path, "ALL_SLIDES", "VAR_ST")
        
        # 生成可视化（如果启用）
        enable_vis = self._get_visualization_setting()
        if enable_vis and VISUALIZATION_AVAILABLE:
            print("🎨 生成测试结果可视化...")
            
            # 获取基因名称和marker基因
            gene_names = self._load_gene_names()
            marker_genes = self.get_marker_genes_for_dataset(getattr(self.config, 'expr_name', 'default'))
            
            # 为每个slide生成可视化
            for slide_id, slide_results in all_slide_results.items():
                try:
                    # 获取对应的adata
                    slide_data = original_dataset.get_full_slide_for_testing(slide_id)
                    adata = slide_data.get('adata', None)
                    
                    self.create_visualizations(
                        phase=f"test_{slide_id}",
                        y_true=slide_results['targets'],
                        y_pred=slide_results['predictions'],
                        metrics=slide_results['metrics'],
                        gene_names=gene_names,
                        marker_genes=marker_genes,
                        adata=adata,
                        slide_id=slide_id
                    )
                except Exception as e:
                    logger.warning(f"Failed to create visualization for slide {slide_id}: {e}")
            
            # 生成整体可视化
            try:
                self.create_visualizations(
                    phase="test_overall",
                    y_true=all_targets,
                    y_pred=all_predictions,
                    metrics=overall_metrics,
                    gene_names=gene_names,
                    marker_genes=marker_genes
                )
            except Exception as e:
                logger.warning(f"Failed to create overall visualization: {e}")
        
        print(f"\n🎉 整slide测试完成!")
        print(f"  测试slides数量: {len(test_slide_ids)}")
        print(f"  总spots数量: {all_predictions.shape[0]}")
        print(f"  基因数量: {all_predictions.shape[1]}")
        print(f"  整体PCC-10: {overall_metrics['PCC-10']:.4f}")
        print(f"  整体MSE: {overall_metrics['MSE']:.4f}")
        
        return {
            'slide_results': all_slide_results,
            'overall_metrics': overall_metrics,
            'overall_predictions': all_predictions,
            'overall_targets': all_targets,
            'test_slide_ids': test_slide_ids
        }
