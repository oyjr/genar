"""
VAR-ST模型的PyTorch Lightning接口
精简版本：保留核心功能，删除冗余代码
"""

# 标准库导入
import os
import inspect
import importlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# 第三方库导入
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch Lightning相关
import pytorch_lightning as pl

# Metrics
import torchmetrics
from torchmetrics.regression import (
    PearsonCorrCoef,
    MeanAbsoluteError,
    MeanSquaredError,
    ConcordanceCorrCoef,   
    R2Score,
)

# 项目内部导入
from addict import Dict as AddictDict

# 设置日志记录器
logger = logging.getLogger(__name__)

# 默认常量
DEFAULT_NUM_GENES = 200
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_GRADIENT_CLIP = 1.0
MAX_SAVED_SAMPLES = 10000
LOG_FREQUENCY = 100
TOP_GENE_RATIO = 0.3
MIN_VARIANCE_THRESHOLD = 1e-8


class ModelInterface(pl.LightningModule):
    """VAR-ST模型的PyTorch Lightning接口"""

    def __init__(self, config):
        super().__init__()
        
        # 创建专用logger
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 保存配置
        self.config = config
        self.save_hyperparameters()

        # 加载模型配置和模型
        self.model_config = config.MODEL
        self._logger.info("初始化VAR-ST模型接口")
        self.model = self._load_model()
        
        # 初始化损失函数（实际在_compute_loss中实现期望值损失）
        self.criterion = torch.nn.MSELoss()  # 保留作为备用
        self._logger.info("使用期望值回归损失（在_compute_loss中实现）")

        # 初始化输出缓存（只用于验证和测试）
        self.val_outputs = []
        self.test_outputs = []
        self.validation_step_outputs = []

        # 初始化指标
        self._init_metrics()

        # 获取标准化设置
        self.normalize = self._get_config('DATA.normalize', True)

    def _get_config(self, path: str, default=None):
        """安全地获取配置值"""
        parts = path.split('.')
        value = self.config
        
        try:
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, default)
                elif hasattr(value, part):
                    value = getattr(value, part)
                else:
                    return default
            return value
        except Exception:
            return default

    def _load_model(self):
        """加载Multi-Scale Gene VAR模型"""
        try:
            self._logger.info("加载Multi-Scale Gene VAR模型...")
            Model = getattr(importlib.import_module(
                'model.VAR.two_stage_var_st'), 'MultiScaleGeneVAR')
            
            # 实例化模型
            model = self._instancialize(Model)
            self._logger.info("Multi-Scale Gene VAR模型加载成功")
            
            return model
            
        except Exception as e:
            self._logger.error(f"加载Multi-Scale Gene VAR模型时出错：{str(e)}")
            raise ValueError(f'Multi-Scale Gene VAR模型加载失败: {str(e)}')

    def _instancialize(self, Model, **other_args):
        """实例化模型"""
        try:
            # 获取模型初始化参数
            class_args = inspect.getfullargspec(Model.__init__).args[1:]
            
            # 处理不同类型的配置对象
            if isinstance(self.model_config, AddictDict):
                model_config_dict = dict(self.model_config)
            elif hasattr(self.model_config, '__dict__'):
                model_config_dict = vars(self.model_config)
            else:
                model_config_dict = self.model_config
            
            args = {}
            
            # 从配置中获取参数
            for arg in class_args:
                if arg in model_config_dict:
                    args[arg] = model_config_dict[arg]
                elif arg == 'config':
                    args[arg] = self.config
                elif arg == 'histology_feature_dim' and 'feature_dim' in model_config_dict:
                    args[arg] = model_config_dict['feature_dim']
                    
            # 添加其他参数
            args.update(other_args)
            
            return Model(**args)
            
        except Exception as e:
            self._logger.error(f"模型实例化失败：{str(e)}")
            raise

    def _init_metrics(self):
        """初始化评估指标"""
        num_genes = self._get_config('MODEL.num_genes', DEFAULT_NUM_GENES)
        self._logger.info(f"VAR_ST模型使用基因数量: {num_genes}")
        
        # 创建指标集合
        metrics = {
            'mse': MeanSquaredError(),
            'mae': MeanAbsoluteError(),
            'pearson': PearsonCorrCoef(num_outputs=num_genes),
            'concordance': ConcordanceCorrCoef(num_outputs=num_genes),
            'r2': R2Score(multioutput='uniform_average')
        }

        self.train_metrics = torchmetrics.MetricCollection(metrics.copy())
        self.val_metrics = torchmetrics.MetricCollection(metrics.copy())
        self.test_metrics = torchmetrics.MetricCollection(metrics.copy())

        # 创建详细指标
        self.val_detailed_metrics = self._create_detailed_metrics(num_genes)
        self.test_detailed_metrics = self._create_detailed_metrics(num_genes)

    def _common_step(self, batch, batch_idx, phase: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """通用的step处理逻辑"""
        # 预处理
        original_batch = batch.copy() if isinstance(batch, dict) else batch
        processed_batch = self._preprocess_inputs(batch)
        
        # 🔧 关键修复：验证和测试时移除target_genes以启用真正的推理模式
        if phase in ['val', 'test'] and 'target_genes' in processed_batch:
            # 保存target_genes用于损失计算，但从模型输入中移除
            _ = processed_batch.pop('target_genes')
        
        # 前向传播
        results_dict = self.model(**processed_batch)
        # 计算损失
        loss = self._compute_loss(results_dict, original_batch)
        # 提取预测和目标
        logits, target_genes = self._extract_predictions_and_targets(results_dict, original_batch)
        # 🔧 谨慎处理指标更新，避免验证时的多GPU同步问题
        should_log = (phase == 'train' and batch_idx % LOG_FREQUENCY == 0)
        if should_log:
            # 只在训练时更新指标，验证时避免调用复杂的指标计算
            self._update_metrics(phase, logits, target_genes)
        
        # 记录损失 - 避免验证时同步
        # 🔧 关键修复：验证时不同步，避免死锁
        sync_dist = False  # 完全禁用同步，让Lightning在epoch end处理
        batch_size = original_batch.get('target_genes', torch.empty(1)).size(0) if isinstance(original_batch, dict) else 1
        
        # 只在training_step中记录，validation_step自己处理
        if phase == 'train':
            self.log(f'{phase}_loss', loss, 
                    on_step=True, 
                    on_epoch=True, 
                    prog_bar=True,
                    batch_size=batch_size,
                    sync_dist=sync_dist)
        
        # 记录模型特定指标
        self._log_model_specific_metrics(phase, results_dict)
        
        return loss, logits, target_genes

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        loss, _, _ = self._common_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤 - 修复多GPU同步问题"""
        # 执行完整的验证步骤
        loss, predictions, targets = self._common_step(batch, batch_idx, 'val')
        
        # 🔧 关键修复：在多GPU环境下正确同步val_loss
        # 记录验证损失（启用同步以确保ModelCheckpoint能获取正确的值）
        self.log('val_loss', loss, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True,
                batch_size=targets.size(0) if hasattr(targets, 'size') else 1,
                sync_dist=True,  # 🔧 关键修复：启用同步确保ModelCheckpoint正确工作
                reduce_fx='mean')  # 明确指定reduce函数
        
        # 🔧 收集验证输出用于PCC计算
        output = {
            'val_loss': loss,
            'predictions': predictions.detach().cpu(),  # 移到CPU减少GPU内存
            'targets': targets.detach().cpu()
        }
        
        # 添加到验证输出列表
        self.validation_step_outputs.append(output)
        
        return output

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        loss, logits, target_genes = self._common_step(batch, batch_idx, 'test')
        # 保存输出
        self._save_step_outputs('test', loss, logits, target_genes, batch_idx)
        return {'logits': logits, 'target_genes': target_genes}
    
    def _preprocess_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """预处理输入数据"""
        # 验证输入
        self._validate_inputs(inputs)
        processed = {}
        # 组织学特征处理
        if 'img' in inputs:
            processed['histology_features'] = inputs['img']
        # 空间坐标处理
        if 'positions' in inputs:
            processed['spatial_coords'] = inputs['positions']
        # 基因表达数据处理 - 保留原始逻辑，让_common_step处理推理逻辑
        if 'target_genes' in inputs:
            processed['target_genes'] = inputs['target_genes']
        return processed

    def _validate_inputs(self, inputs: Dict[str, torch.Tensor]):
        """验证输入数据"""
        # 检查必需的键
        if 'img' not in inputs:
            raise ValueError("缺少必需的输入: img")
        
        # 定义不同字段的期望维度
        expected_dims = {
            'img': [2, 3],           # 图像特征: (batch, features) 或 (batch, seq, features)
            'target_genes': [2, 3],   # 基因表达: (batch, genes) 或 (batch, seq, genes)
            'positions': [2, 3],      # 空间坐标: (batch, coords) 或 (batch, seq, coords)
            'spot_idx': [1, 2],       # spot索引: (batch,) 或 (batch, seq)
            'slide_id': [1],          # slide标识: (batch,)
            'gene_ids': [1, 2],       # 基因ID: (batch,) 或 (batch, genes)
        }
        
        # 验证张量形状
        for key, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                # 获取该字段的期望维度，如果未定义则允许1-3维
                allowed_dims = expected_dims.get(key, [1, 2, 3])
                
                if tensor.dim() not in allowed_dims:
                    raise ValueError(f"{key}维度错误: {tensor.shape}，期望维度: {allowed_dims}")
        
        # 验证数值范围
        if 'target_genes' in inputs:
            targets = inputs['target_genes']
            if (targets < 0).any():
                raise ValueError("目标基因表达值包含负数")

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """使用期望值回归损失替代交叉熵损失"""
        try:
            # 检查是否有logits输出（用于期望值损失）
            if 'logits' in outputs and ('full_target' in outputs or 'target_genes' in batch):
                # 使用期望值回归损失
                logits = outputs['logits']  # [B, seq_len, vocab_size] 或 [B*seq_len, vocab_size]
                
                # 优先使用full_target（VAR模型的多尺度目标），否则使用target_genes
                if 'full_target' in outputs:
                    targets = outputs['full_target']  # [B, seq_len] - VAR模型的多尺度目标
                else:
                    targets = batch['target_genes']   # [B, seq_len] - 标准目标
                
                # 确保维度匹配
                if logits.dim() == 3:
                    B, seq_len, V = logits.shape
                    logits = logits.view(-1, V)
                    targets = targets.view(-1)
                
                # 创建token到log2连续值的映射
                vocab_size = logits.shape[-1]
                token_values = torch.log2(torch.arange(vocab_size, dtype=torch.float32, device=logits.device) + 1.0)
                
                # 计算真实连续值
                true_continuous = token_values[targets]
                
                # 计算期望连续值
                probs = F.softmax(logits, dim=-1)
                expected_continuous = torch.sum(probs * token_values[None, :], dim=-1)
                
                # 使用MSE损失（与评估指标一致）
                total_loss = F.mse_loss(expected_continuous, true_continuous)
                
                # 记录额外指标
                with torch.no_grad():
                    token_acc = (logits.argmax(dim=-1) == targets).float().mean()
                    self.log('train_token_accuracy', token_acc, prog_bar=False, sync_dist=False)
                    self.log('train_expected_mse', total_loss.detach(), prog_bar=True, sync_dist=False)
                
                self._logger.debug(f"期望值回归损失={total_loss:.4f}")
                
            elif 'predictions' in outputs and 'target_genes' in batch:
                # 推理模式：直接使用预测的token IDs
                predictions = outputs['predictions']  # [B, 200] token IDs
                targets = batch['target_genes']       # [B, 200] token IDs
                predictions_log2 = torch.log2(predictions.float() + 1.0)
                targets_log2 = torch.log2(targets.float() + 1.0)
                total_loss = F.mse_loss(predictions_log2, targets_log2)
                self._logger.debug(f"推理模式：使用log2(x+1)变换后的MSE损失={total_loss:.4f}")
                
            elif 'loss' in outputs:
                # 备用方案：使用模型内部损失
                total_loss = outputs['loss']
                self._logger.warning("使用模型内部交叉熵损失（备用方案）")
                
            else:
                raise KeyError("无法计算损失：缺少必要的输出")
            
            # 验证损失值
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self._logger.error(f"损失值异常: {total_loss.item()}")
                raise ValueError("损失值为NaN或Inf")
                
            return total_loss
            
        except Exception as e:
            self._logger.error(f"计算损失时出错: {str(e)}")
            self._logger.error(f"输出键: {list(outputs.keys())}")
            raise

    def _extract_predictions_and_targets(self, results_dict: Dict[str, torch.Tensor], 
                                       batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """提取预测和目标"""
        # 获取预测
        if 'predictions' in results_dict:
            logits = results_dict['predictions']
        else:
            logits = results_dict.get('generated_sequence', None)
            if logits is None:
                raise ValueError("Multi-Scale Gene VAR模型应该有predictions或generated_sequence输出")
        
        # 获取目标
        if 'target_genes' not in batch:
            raise ValueError("批次数据中找不到target_genes")
        target_genes = batch['target_genes']
        
        # 🔧 关键修复：VAR_ST模型应该直接返回200个基因的预测
        # 如果形状不匹配，说明模型实现有问题，直接报错
        num_genes = target_genes.shape[-1]  # 通常是200
        
        if logits.shape[-1] != num_genes:
            raise ValueError(
                f"模型预测维度({logits.shape[-1]})与目标基因数量({num_genes})不匹配！"
                f"这表明训练和推理的模型配置不一致。"
                f"预测形状: {logits.shape}, 目标形状: {target_genes.shape}"
            )
        
        # 训练时直接使用原始计数值，不进行log2变换
        # 评估指标计算时会在需要的地方进行log2变换
        return logits.float(), target_genes.float()

    def _apply_log2_normalization_for_evaluation(self, predictions: torch.Tensor, 
                                                targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """为评估指标应用log2(x+1)标准化 - 只在评估时使用"""
        # 确保数据类型为float
        predictions = predictions.float()
        targets = targets.float()
        
        # 应用log2(x+1)标准化
        predictions_log2 = torch.log2(predictions + 1.0)
        targets_log2 = torch.log2(targets + 1.0)
        
        # 验证结果
        if torch.isnan(predictions_log2).any() or torch.isnan(targets_log2).any():
            self._logger.warning("Log2标准化后发现NaN值")
        
        return predictions_log2, targets_log2

    def _update_metrics(self, stage: str, predictions: torch.Tensor, targets: torch.Tensor):
        """更新评估指标"""
        try:
            # 获取对应阶段的指标集合
            metrics = getattr(self, f'{stage}_metrics')
            
            # 确保输入维度正确
            if predictions.dim() == 3:
                B, N, G = predictions.shape
                predictions = predictions.reshape(-1, G)
            if targets.dim() == 3:
                B, N, G = targets.shape
                targets = targets.reshape(-1, G)
            
            # 为了与其他模型对比，在计算评估指标时应用log2变换
            predictions_log2, targets_log2 = self._apply_log2_normalization_for_evaluation(predictions, targets)
            
            # 训练阶段每次都重置指标
            if stage == 'train':
                metrics.reset()
            
            # 使用log2变换后的值更新指标
            metrics.update(predictions_log2, targets_log2)

            # 计算并记录指标
            if stage == 'train' or self.trainer.global_step % LOG_FREQUENCY == 0:
                self._log_metrics(stage, metrics, predictions.size(0))

        except Exception as e:
            self._logger.error(f"更新指标时发生错误: {e}")
            raise

    def _log_metrics(self, stage: str, metrics: torchmetrics.MetricCollection, batch_size: int):
        """记录指标"""
        metric_dict = metrics.compute()
        
        for name, value in metric_dict.items():
            if isinstance(value, torch.Tensor):
                if value.numel() > 1:
                    # 多元素张量
                    values = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
                    mean_value = values.mean()
                    std_value = values.std() if values.numel() > 1 else torch.tensor(0.0, device=value.device)
                    
                    # 只在进度条显示重要指标
                    show_in_prog_bar = name in ['mse', 'mae']
                    
                    self.log(f'{stage}_{name}', mean_value, 
                            prog_bar=show_in_prog_bar, 
                            batch_size=batch_size, 
                            sync_dist=True)
                    self.log(f'{stage}_{name}_std', std_value, 
                            prog_bar=False, 
                            batch_size=batch_size, 
                            sync_dist=True)
                    
                    # 记录高相关性基因
                    if name in ['pearson', 'concordance']:
                        self._log_high_correlation_genes(stage, name, values, batch_size)
                else:
                    # 单元素张量
                    self.log(f'{stage}_{name}', value.item(), 
                            prog_bar=(name in ['mse', 'mae']), 
                            batch_size=batch_size, 
                            sync_dist=True)

    def _log_high_correlation_genes(self, stage: str, metric_name: str, 
                                   values: torch.Tensor, batch_size: int):
        """记录高相关性基因的统计"""
        top_k = max(1, int(len(values) * TOP_GENE_RATIO))
        high_values = torch.topk(values, top_k)[0]
        high_mean = high_values.mean()
        high_std = high_values.std() if high_values.numel() > 1 else torch.tensor(0.0, device=values.device)
        
        self.log(f'{stage}_{metric_name}_high_mean', high_mean, 
                prog_bar=False, 
                batch_size=batch_size, 
                sync_dist=True)
        self.log(f'{stage}_{metric_name}_high_std', high_std, 
                prog_bar=False, 
                batch_size=batch_size, 
                sync_dist=True)

    def _log_model_specific_metrics(self, phase: str, results_dict: Dict[str, Any]):
        """记录模型特定的指标"""
        # 🔧 减少同步，只在训练时记录详细指标
        if phase == 'train':
            # VAR-ST的特定指标
            if 'accuracy' in results_dict:
                self.log(f'{phase}_accuracy', results_dict['accuracy'], 
                        on_epoch=True, 
                        sync_dist=False)
            
            if 'perplexity' in results_dict:
                self.log(f'{phase}_perplexity', results_dict['perplexity'], 
                        on_epoch=True, 
                        sync_dist=False)
            
            if 'top5_accuracy' in results_dict:
                self.log(f'{phase}_top5_accuracy', results_dict['top5_accuracy'], 
                        on_epoch=True, 
                        sync_dist=False)

    def _save_step_outputs(self, phase: str, loss: torch.Tensor, 
                          preds: torch.Tensor, targets: torch.Tensor, 
                          batch_idx: Optional[int] = None):
        """保存步骤输出"""
        if phase == 'train':
            return  # 训练阶段不保存
        
        # 检查内存限制
        current_samples = sum(out['preds'].shape[0] for out in getattr(self, f'{phase}_outputs'))
        if current_samples >= MAX_SAVED_SAMPLES:
            if batch_idx == 0:  # 只在第一个batch时警告
                self._logger.warning(f"{phase}阶段已保存{current_samples}个样本，达到上限")
            return
        
        output_dict = {
            'loss': loss.detach().cpu().item(),
            'preds': preds.detach().cpu(),
            'targets': targets.detach().cpu(),
        }
        if batch_idx is not None:
            output_dict['batch_idx'] = batch_idx

        getattr(self, f'{phase}_outputs').append(output_dict)

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        weight_decay = float(self._get_config('TRAINING.weight_decay', DEFAULT_WEIGHT_DECAY))
        learning_rate = float(self._get_config('TRAINING.learning_rate', DEFAULT_LEARNING_RATE))
        
        # 多GPU学习率缩放
        learning_rate = self._scale_learning_rate(learning_rate)
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 设置梯度裁剪
        self.trainer.gradient_clip_val = self._get_config('TRAINING.gradient_clip_val', DEFAULT_GRADIENT_CLIP)
        
        # 配置学习率调度器
        scheduler_config = self._get_scheduler_config(optimizer)
        
        if scheduler_config:
            return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
        else:
            return {'optimizer': optimizer}

    def _scale_learning_rate(self, base_lr: float) -> float:
        """根据GPU数量缩放学习率"""
        num_devices = self._get_config('devices', 1)
        if num_devices <= 1:
            return base_lr
        
        scaling_strategy = self._get_config('MULTI_GPU.lr_scaling', 'none')
        
        if scaling_strategy == 'linear':
            scaled_lr = base_lr * num_devices
            self._logger.info(f"线性缩放学习率: {scaled_lr} (原始: {base_lr}, 设备数: {num_devices})")
        elif scaling_strategy == 'sqrt':
            scaled_lr = base_lr * (num_devices ** 0.5)
            self._logger.info(f"平方根缩放学习率: {scaled_lr} (原始: {base_lr}, 设备数: {num_devices})")
        else:
            scaled_lr = base_lr
            self._logger.info(f"不缩放学习率: {scaled_lr}")
        
        return scaled_lr

    def _get_scheduler_config(self, optimizer):
        """获取学习率调度器配置"""
        # 获取配置参数
        factor = self._get_config('TRAINING.lr_scheduler.factor', 0.5)
        patience = self._get_config('TRAINING.lr_scheduler.patience', 5)
        mode = self._get_config('TRAINING.lr_scheduler.mode', 'min')
        
        if patience == 0:
            self._logger.info("学习率调度器已禁用")
            return None
        
        return {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                verbose=True
            ),
            'monitor': self._get_config('TRAINING.monitor', 'val_loss'),
            'interval': 'epoch',
            'frequency': 1
        }

    def on_train_epoch_end(self):
        """训练epoch结束时的回调"""
        pass  # 训练数据不再累积
    
    def on_validation_epoch_end(self):
        """验证epoch结束时的回调 - 计算和打印PCC指标"""
        
        # 🔧 收集验证数据并计算PCC指标
        if hasattr(self, 'validation_step_outputs') and self.validation_step_outputs:
            try:
                # 收集所有验证数据
                all_predictions = []
                all_targets = []
                
                for output in self.validation_step_outputs:
                    all_predictions.append(output['predictions'])
                    all_targets.append(output['targets'])
                
                # 合并数据
                predictions = torch.cat(all_predictions, dim=0)  # [N, genes]
                targets = torch.cat(all_targets, dim=0)  # [N, genes]
                
                # 计算PCC指标
                pcc_metrics = self._calculate_comprehensive_pcc_metrics(predictions, targets)
                
                # 记录PCC指标到wandb
                for metric_name, value in pcc_metrics.items():
                    self.log(f'val_{metric_name}', value, on_epoch=True, prog_bar=False, sync_dist=True)
                
                # 在主进程打印详细结果
                if self.trainer.is_global_zero:
                    val_loss = self.trainer.callback_metrics.get('val_loss', 0.0)
                    print(f"\n🎯 Epoch {self.current_epoch} 验证结果:")
                    print(f"   Loss: {val_loss:.6f}")
                    print(f"   PCC-10:  {pcc_metrics['pcc_10']:.4f}")
                    print(f"   PCC-50:  {pcc_metrics['pcc_50']:.4f}")
                    print(f"   PCC-200: {pcc_metrics['pcc_200']:.4f}")
                    print(f"   MSE:     {pcc_metrics['mse']:.6f}")
                    print(f"   MAE:     {pcc_metrics['mae']:.6f}")
                    print(f"   RVD:     {pcc_metrics['rvd']:.6f}")
                    print()
                
                # 清理验证输出数据
                self.validation_step_outputs.clear()
                
            except Exception as e:
                self._logger.error(f"计算PCC指标时出错: {e}")
                import traceback
                traceback.print_exc()
        
        # 清理验证数据（安全操作）
        self._cleanup_validation_data()
            
        # 🔧 重置验证指标以释放内存
    
    def _cleanup_validation_data(self):
        """安全地清理验证相关数据"""
        # 清空验证输出
        if hasattr(self, 'val_outputs'):
            self.val_outputs.clear()
        if hasattr(self, '_collected_predictions'):
            self._collected_predictions.clear()
        if hasattr(self, '_collected_targets'):
            self._collected_targets.clear()
            
        # 重置验证指标（这个操作是安全的）
        if hasattr(self, 'val_metrics'):
            try:
                self.val_metrics.reset()
            except Exception:
                pass  # 如果重置失败就忽略
        
        # 🔧 确保验证指标正确重置
        try:
            if hasattr(self, 'val_metrics'):
                self.val_metrics.reset()
        except Exception:
            pass  # 忽略重置错误

    def on_test_epoch_end(self):
        """测试epoch结束时的回调"""
        self._compute_and_log_evaluation_metrics('test')
        
        # 清空测试输出
        if self.current_epoch < self.trainer.max_epochs - 1:
            self.test_outputs.clear()
    
    def on_fit_end(self):
        """训练完成时的回调"""
        if not self.trainer.is_global_zero:
            self._logger.info(f"GPU进程 {self.trainer.global_rank}: 训练完成")
            return
        
        self._logger.info("训练完成！")

    def _compute_and_log_evaluation_metrics(self, phase: str):
        """计算并记录评估指标"""
        outputs = getattr(self, f'{phase}_outputs', [])
        
        if not outputs:
            self._logger.warning(f"没有{phase}阶段的输出数据")
            return
        
        # 检查是否为sanity check
        if hasattr(self.trainer, 'sanity_checking') and self.trainer.sanity_checking:
            self._logger.info("跳过sanity check阶段的详细评估")
            return
       
        self._logger.info(f"开始计算{phase}阶段的评估指标...")
        
        # 收集所有输出
        all_predictions, all_targets = self._collect_outputs(outputs)
        
        # 确保数据在正确的设备上
        predictions = all_predictions.to(self.device)
        targets = all_targets.to(self.device)
        
        # 🔧 关键修复：测试阶段也需要应用log2变换来计算指标
        # 为了与其他模型对比，在计算评估指标时应用log2变换
        predictions_log2, targets_log2 = self._apply_log2_normalization_for_evaluation(predictions, targets)
        
        # 使用TorchMetrics计算标准指标（使用log2变换后的值）
        metrics = getattr(self, f'{phase}_metrics')
        metrics.reset()
        metrics.update(predictions_log2, targets_log2)
        metric_dict = metrics.compute()
        
        # 记录标准指标
        self._log_evaluation_metrics(phase, metric_dict)
        
        # 计算详细指标（也使用log2变换后的值）
        if hasattr(self, f'{phase}_detailed_metrics'):
            detailed_metrics = getattr(self, f'{phase}_detailed_metrics')
            detailed_metrics.reset()
            detailed_metrics.update(predictions_log2, targets_log2)
            detailed_dict = detailed_metrics.compute()
            self._log_detailed_metrics(phase, detailed_dict)
        
        # 在主进程上生成评估报告
        if self.trainer.is_global_zero:
            self._generate_simple_evaluation_report(phase, metric_dict)

    def _log_evaluation_metrics(self, phase: str, metric_dict: Dict[str, torch.Tensor]):
        """记录评估指标"""
        processed_metrics = {}
        
        for name, value in metric_dict.items():
            if isinstance(value, torch.Tensor):
                if value.numel() > 1:
                    # 多元素张量
                    values = torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6)
                    mean_value = values.mean()
                    std_value = values.std() if values.numel() > 1 else torch.tensor(0.0, device=value.device)
                    
                    # 记录指标，确保多GPU同步
                    show_in_prog_bar = name in ['mse', 'mae']
                    self.log(f'{phase}_{name}', mean_value, on_epoch=True, prog_bar=show_in_prog_bar, sync_dist=True, reduce_fx='mean')
                    self.log(f'{phase}_{name}_std', std_value, on_epoch=True, prog_bar=False, sync_dist=True, reduce_fx='mean')
                    
                    processed_metrics[name] = mean_value.item()
                    
                    # 高相关性基因统计
                    if name in ['pearson', 'concordance']:
                        self._log_high_correlation_genes(phase, name, values, 1)
                else:
                    # 单元素张量
                    scalar_value = value.item()
                    self.log(f'{phase}_{name}', scalar_value, on_epoch=True, 
                            prog_bar=(name in ['mse', 'mae']), sync_dist=True, reduce_fx='mean')
                    processed_metrics[name] = scalar_value
        
        return processed_metrics

    def _log_detailed_metrics(self, phase: str, detailed_dict: Dict[str, torch.Tensor]):
        """记录详细指标"""
        for name, value in detailed_dict.items():
            if isinstance(value, torch.Tensor):
                scalar_value = value.item() if value.numel() == 1 else value.mean().item()
                self.log(f'{phase}_{name}', scalar_value, on_epoch=True, prog_bar=False, sync_dist=True)

    def _generate_simple_evaluation_report(self, phase: str, metric_dict: Dict[str, torch.Tensor]):
        """生成简化的评估报告"""
        self._logger.info("=" * 40)
        self._logger.info(f"{phase.upper()} 评估结果")
        self._logger.info("=" * 40)
        
        for name, value in metric_dict.items():
            if isinstance(value, torch.Tensor):
                if value.numel() > 1:
                    mean_val = torch.nan_to_num(value, nan=0.0).mean().item()
                    self._logger.info(f"  {name}: {mean_val:.4f}")
                else:
                    self._logger.info(f"  {name}: {value.item():.4f}")
        
        self._logger.info("=" * 40)

    def _collect_outputs(self, outputs: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """收集并合并输出"""
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
        
        return torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0)

    def _create_detailed_metrics(self, num_genes: int):
        """创建详细指标计算器"""
        from torchmetrics import Metric
        
        class DetailedMetrics(Metric):
            def __init__(self, num_genes):
                super().__init__()
                self.num_genes = num_genes
                # 添加状态张量
                self.add_state("preds_sum", default=torch.zeros(num_genes), dist_reduce_fx="sum")
                self.add_state("targets_sum", default=torch.zeros(num_genes), dist_reduce_fx="sum")
                self.add_state("preds_sq_sum", default=torch.zeros(num_genes), dist_reduce_fx="sum")
                self.add_state("targets_sq_sum", default=torch.zeros(num_genes), dist_reduce_fx="sum")
                self.add_state("preds_targets_sum", default=torch.zeros(num_genes), dist_reduce_fx="sum")
                self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
            
            def update(self, preds: torch.Tensor, targets: torch.Tensor):
                # 确保维度正确
                if preds.dim() == 3:
                    preds = preds.reshape(-1, preds.size(-1))
                if targets.dim() == 3:
                    targets = targets.reshape(-1, targets.size(-1))
                
                batch_size = preds.size(0)
                
                # 累积统计量
                self.preds_sum += preds.sum(dim=0)
                self.targets_sum += targets.sum(dim=0)
                self.preds_sq_sum += (preds ** 2).sum(dim=0)
                self.targets_sq_sum += (targets ** 2).sum(dim=0)
                self.preds_targets_sum += (preds * targets).sum(dim=0)
                self.total += batch_size
            
            def compute(self):
                # 计算每个基因的相关系数
                n = self.total.float()
                
                # 计算均值
                preds_mean = self.preds_sum / n
                targets_mean = self.targets_sum / n
                
                # 计算协方差和方差
                covariance = (self.preds_targets_sum / n) - (preds_mean * targets_mean)
                preds_var = (self.preds_sq_sum / n) - (preds_mean ** 2)
                targets_var = (self.targets_sq_sum / n) - (targets_mean ** 2)
                
                # 计算相关系数
                correlations = covariance / torch.sqrt(preds_var * targets_var + 1e-8)
                correlations = torch.nan_to_num(correlations, nan=0.0)
                
                # 计算PCC指标
                sorted_corr, _ = torch.sort(correlations, descending=True)
                
                pcc_10 = sorted_corr[:10].mean() if self.num_genes >= 10 else sorted_corr.mean()
                pcc_50 = sorted_corr[:50].mean() if self.num_genes >= 50 else sorted_corr.mean()
                pcc_200 = sorted_corr[:200].mean() if self.num_genes >= 200 else sorted_corr.mean()
                
                return {
                    'pcc_10': pcc_10,
                    'pcc_50': pcc_50,
                    'pcc_200': pcc_200,
                    'correlations_mean': correlations.mean()
                }
        
        return DetailedMetrics(num_genes)

    def calculate_gene_correlations(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """计算基因级别的相关系数"""
        num_genes = y_true.shape[1]
        correlations = np.zeros(num_genes)
        
        for i in range(num_genes):
            true_gene = y_true[:, i]
            pred_gene = y_pred[:, i]
            
            # 处理常数值
            if np.std(true_gene) == 0 or np.std(pred_gene) == 0:
                correlations[i] = 0.0
            else:
                corr = np.corrcoef(true_gene, pred_gene)[0, 1]
                correlations[i] = 0.0 if np.isnan(corr) else corr
        
        return correlations

    def calculate_evaluation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算综合评估指标"""
        # 确保输入是numpy数组
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().numpy()
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().numpy()
        
        # 计算基因相关性
        correlations = self.calculate_gene_correlations(y_true, y_pred)
        
        # 排序相关性
        sorted_corr = np.sort(correlations)[::-1]
        
        # 计算PCC指标
        pcc_10 = np.mean(sorted_corr[:10]) if len(sorted_corr) >= 10 else np.mean(sorted_corr)
        pcc_50 = np.mean(sorted_corr[:50]) if len(sorted_corr) >= 50 else np.mean(sorted_corr)
        pcc_200 = np.mean(sorted_corr[:200]) if len(sorted_corr) >= 200 else np.mean(sorted_corr)
        
        # 计算MSE和MAE
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # 计算RVD
        pred_var = np.var(y_pred, axis=0)
        true_var = np.var(y_true, axis=0)
        
        valid_mask = true_var > MIN_VARIANCE_THRESHOLD
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
            'correlations': correlations
        }

    def on_before_optimizer_step(self, optimizer):
        """优化器步骤前的回调"""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), 
            self.trainer.gradient_clip_val
        )
        
        self.log('grad_norm', grad_norm, sync_dist=True)

    def _calculate_pcc_metrics(self, val_metrics):
        """从验证期间收集的数据计算PCC-10, PCC-50, PCC-200 - 安全版本"""
        pcc_metrics = {}
        
        # 🔧 暂时禁用PCC计算以避免死锁
        # 在多GPU环境下torch.corrcoef可能导致死锁
        self._logger.info("PCC计算已暂时禁用以避免死锁问题")
        
        # 清理收集的数据
        if hasattr(self, '_collected_predictions'):
            self._collected_predictions.clear()
        if hasattr(self, '_collected_targets'):
            self._collected_targets.clear()
        
        return pcc_metrics

    def _print_simple_validation_summary(self):
        """打印简化的验证结果摘要"""
        if not self.trainer.is_global_zero:
            return
            
        try:
            val_metrics = self.val_metrics.compute()
            
            # 计算PCC指标
            pcc_metrics = self._calculate_pcc_metrics(val_metrics)
            
            # 提取关键指标
            key_metrics = {}
            for name, value in val_metrics.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() > 1:
                        mean_val = torch.nan_to_num(value, nan=0.0).mean().item()
                        key_metrics[name] = mean_val
                    else:
                        key_metrics[name] = value.item()
            
            # 简洁格式打印
            print(f"\n🎯 Epoch {self.current_epoch} 验证结果:")
            
            # 优先显示PCC指标
            if pcc_metrics:
                print("   📊 PCC指标:")
                for pcc_name, pcc_value in pcc_metrics.items():
                    print(f"      {pcc_name}: {pcc_value:.4f}")
            
            # 显示基础指标
            basic_metrics = ['mse', 'mae', 'r2']
            print("   📈 基础指标:")
            for metric in basic_metrics:
                if metric in key_metrics:
                    print(f"      {metric.upper()}: {key_metrics[metric]:.4f}")
            
            print()  # 空行分隔
            
        except Exception as e:
            print(f"❌ 验证结果打印失败: {e}")
            self._logger.error(f"简化验证摘要打印出错: {e}")

    def on_validation_epoch_start(self):
        """验证epoch开始时重置指标"""
        try:
            # 安全地重置验证指标
            self.val_metrics.reset()
            self._logger.debug(f"开始验证Epoch {self.current_epoch}")
        except Exception as e:
            self._logger.warning(f"重置验证指标时出现警告: {e}")
        
        # 🔧 清理之前可能残留的数据
        self._cleanup_validation_data()

    def _print_simple_validation_summary_safe(self):
        """安全的验证结果摘要打印 - 避免死锁"""
        if not self.trainer.is_global_zero:
            return
            
        try:
            val_metrics = self.val_metrics.compute()
            
            # 提取关键指标
            key_metrics = {}
            for name, value in val_metrics.items():
                if isinstance(value, torch.Tensor):
                    if value.numel() > 1:
                        mean_val = torch.nan_to_num(value, nan=0.0).mean().item()
                        key_metrics[name] = mean_val
                    else:
                        key_metrics[name] = value.item()
            
            # 简洁格式打印 - 避免复杂的PCC计算
            print(f"\n🎯 Epoch {self.current_epoch} 验证结果:")
            
            # 只显示基础指标
            basic_metrics = ['mse', 'mae', 'r2']
            print("   📈 基础指标:")
            for metric in basic_metrics:
                if metric in key_metrics:
                    print(f"      {metric.upper()}: {key_metrics[metric]:.4f}")
            
            print()  # 空行分隔
            
        except Exception as e:
            print(f"❌ 安全验证结果打印失败: {e}")
            self._logger.error(f"安全验证摘要打印出错: {e}")

    def _calculate_comprehensive_pcc_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """计算综合PCC指标 - 与推理脚本保持一致"""
        import numpy as np
        
        # 转换为numpy数组
        if torch.is_tensor(predictions):
            predictions = predictions.cpu().numpy()
        if torch.is_tensor(targets):
            targets = targets.cpu().numpy()
        
        # 应用log2(x+1)变换用于评估指标计算（与推理脚本保持一致）
        y_true_log2 = np.log2(targets + 1.0)
        y_pred_log2 = np.log2(predictions + 1.0)
        
        # 检查NaN值
        if np.isnan(y_true_log2).any() or np.isnan(y_pred_log2).any():
            self._logger.warning("⚠️ Log2变换后发现NaN值，将使用原始值")
            y_true_log2 = targets
            y_pred_log2 = predictions
        
        # 计算基因级别的相关性
        num_genes = y_true_log2.shape[1]
        correlations = np.zeros(num_genes)
        
        for i in range(num_genes):
            true_gene = y_true_log2[:, i]
            pred_gene = y_pred_log2[:, i]
            
            # 处理常数值
            if np.std(true_gene) == 0 or np.std(pred_gene) == 0:
                correlations[i] = 0.0
            else:
                corr = np.corrcoef(true_gene, pred_gene)[0, 1]
                correlations[i] = 0.0 if np.isnan(corr) else corr
        
        # 排序相关性
        sorted_corr = np.sort(correlations)[::-1]
        
        # 计算PCC指标
        pcc_10 = np.mean(sorted_corr[:10]) if len(sorted_corr) >= 10 else np.mean(sorted_corr)
        pcc_50 = np.mean(sorted_corr[:50]) if len(sorted_corr) >= 50 else np.mean(sorted_corr)
        pcc_200 = np.mean(sorted_corr[:200]) if len(sorted_corr) >= 200 else np.mean(sorted_corr)
        
        # 计算MSE和MAE（使用log2变换后的值）
        mse = np.mean((y_true_log2 - y_pred_log2) ** 2)
        mae = np.mean(np.abs(y_true_log2 - y_pred_log2))
        
        # 计算RVD (Relative Variance Difference)（使用log2变换后的值）
        MIN_VARIANCE_THRESHOLD = 1e-8
        pred_var = np.var(y_pred_log2, axis=0)
        true_var = np.var(y_true_log2, axis=0)
        
        valid_mask = true_var > MIN_VARIANCE_THRESHOLD
        if np.sum(valid_mask) > 0:
            rvd = np.mean(((pred_var[valid_mask] - true_var[valid_mask]) ** 2) / (true_var[valid_mask] ** 2))
        else:
            rvd = 0.0
        
        return {
            'pcc_10': float(pcc_10),
            'pcc_50': float(pcc_50), 
            'pcc_200': float(pcc_200),
            'mse': float(mse),
            'mae': float(mae),
            'rvd': float(rvd)
        }