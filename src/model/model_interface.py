"""
VAR-ST模型的PyTorch Lightning接口
重构版本：核心Lightning接口，委托具体功能给专门的工具类
"""

import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# 导入工具类
from .model_metrics import ModelMetrics
from .model_utils import ModelUtils

# 设置日志记录器
logging.basicConfig(level=logging.INFO)

# 默认超参数
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_GRADIENT_CLIP = 1.0



class ModelInterface(pl.LightningModule):
    """VAR-ST模型的PyTorch Lightning接口"""

    def __init__(self, config):
        super().__init__()
        
        # 创建专用logger
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 保存配置
        self.config = config
        self.save_hyperparameters()

        # 初始化工具类
        self.model_utils = ModelUtils(config, self)
        self.model_metrics = ModelMetrics(config, self)
        
        # 加载模型
        self._logger.info("初始化VAR-ST模型接口")
        self.model = self.model_utils.load_model()
        
        # 初始化验证和测试输出缓存
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # 从配置中获取推理参数
        self.inference_top_k = self.model_utils.get_config('INFERENCE.top_k', 1)

    def _common_step(self, batch, batch_idx, phase: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """通用的step处理逻辑"""
        original_batch = batch.copy() if isinstance(batch, dict) else batch
        processed_batch = self.model_utils.preprocess_inputs(batch)
        
        loss_final = None

        # Validation and Test steps require special handling to avoid data leakage from teacher forcing
        if phase in ['val', 'test']:
            # Pass 1: Get realistic predictions without teacher forcing for metrics
            inference_batch = processed_batch.copy()
            if 'target_genes' in inference_batch:
                _ = inference_batch.pop('target_genes')
            
            with torch.no_grad():
                # Manually set model to eval mode for this pass
                self.model.eval()
                # Use top-k sampling for realistic, non-deterministic predictions
                inference_results = self.model(**inference_batch, top_k=self.inference_top_k)
            
            # Pass 2: Get the loss using ground truth targets
            # The model is already in eval mode from the trainer, so forward_training will be called
            # This pass will use teacher forcing, but only for loss calculation
            loss_results = self.model(**processed_batch)
            loss = self._compute_loss(loss_results, original_batch)
            loss_final = loss_results.get('loss_final', loss)

            # Extract predictions from the inference pass and targets from the original batch
            predictions, targets = self._extract_predictions_and_targets(inference_results, original_batch)

        else: # Training step
            # Single pass for training
            results_dict = self.model(**processed_batch)
            loss = self._compute_loss(results_dict, original_batch)
            loss_final = results_dict.get('loss_final', loss)
            predictions, targets = self._extract_predictions_and_targets(results_dict, original_batch)
        
        # 记录模型特定指标
        # For val/test, this uses loss_results which contains more metrics than inference_results
        # final_results_for_logging = loss_results if phase in ['val', 'test'] else results_dict
        # self.model_metrics.log_model_specific_metrics(phase, final_results_for_logging) # ✅ FIX: 禁用辅助模块的自动日志，避免重复记录
        
        return loss, loss_final, predictions, targets

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        loss, loss_final, _, _ = self._common_step(batch, batch_idx, 'train')
        self.log('train_loss_final', loss_final, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        # 执行完整的验证步骤
        loss, loss_final, predictions, targets = self._common_step(batch, batch_idx, 'val')
        
        # 获取实际的batch_size
        batch_size = targets.size(0) if hasattr(targets, 'size') else 1
        
        # 记录复合验证损失 (信息性)
        self.log('val_loss', loss, 
                on_step=False, 
                on_epoch=True, 
                prog_bar=True,
                batch_size=batch_size,
                sync_dist=True,
                reduce_fx='mean')
        
        # 记录最终尺度损失 (新的关键监控指标)
        self.log('val_loss_final', loss_final,
                on_step=False, 
                on_epoch=True, 
                prog_bar=True,
                batch_size=batch_size,
                sync_dist=True,
                reduce_fx='mean')
        
        # 暂时移除即时PCC计算，因为现在使用val_loss作为监控指标
        # TODO: 如果后续需要使用val_pcc_50作为监控指标，可以重新启用这部分代码
        
        # 收集验证输出用于详细PCC计算 - 但要避免sanity check阶段
        if not (hasattr(self.trainer, 'sanity_checking') and self.trainer.sanity_checking):
            output = {
                'val_loss': loss_final,  # 使用最终尺度损失
                'predictions': predictions.detach().cpu(),  # 移到CPU减少GPU内存
                'targets': targets.detach().cpu()
            }
            
            # 添加到验证输出列表
            self.validation_step_outputs.append(output)
        
    def test_step(self, batch, batch_idx):
        """测试步骤"""
        loss, loss_final, predictions, targets = self._common_step(batch, batch_idx, 'test')
        
        # 记录最终尺度测试损失
        self.log('test_loss_final', loss_final, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # 收集测试输出用于PCC计算
        output = {
            'test_loss': loss_final, # 使用最终尺度损失
            'predictions': predictions.detach().cpu(),  # 移到CPU减少GPU内存
            'targets': targets.detach().cpu()
        }
        
        # 添加到测试输出列表
        if not hasattr(self, 'test_step_outputs'):
            self.test_step_outputs = []
        self.test_step_outputs.append(output)
        
        return output

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算损失函数"""
        try:
            # 直接使用模型返回的损失，不再重复计算
            if 'loss' in outputs:
                total_loss = outputs['loss']
                
                # 记录额外指标（仅在训练时）
                if self.training and 'logits' in outputs:
                    with torch.no_grad():
                        logits = outputs['logits']
                        if 'full_target' in outputs:
                            targets = outputs['full_target']
                        elif 'target_genes' in batch:
                            targets = batch['target_genes']
                        else:
                            targets = None
                        
                        if targets is not None:
                            # 计算token准确率
                            if logits.dim() == 3:
                                pred_tokens = logits.argmax(dim=-1)
                                targets_flat = targets.view(-1)
                                pred_flat = pred_tokens.view(-1)
                            else:
                                pred_tokens = logits.argmax(dim=-1)
                                targets_flat = targets.view(-1)
                                pred_flat = pred_tokens
                            
                            token_acc = (pred_flat == targets_flat).float().mean()
                            self.log('train_token_accuracy', token_acc, prog_bar=False, sync_dist=False)
                
                self._logger.debug(f"使用模型内部损失={total_loss:.4f}")
                
            else:
                # Fallback for models that don't return 'loss' but 'logits'
                # This part is now less likely to be used with the hierarchical model
                logits = outputs.get('logits')
                if logits is None:
                    raise KeyError("模型输出中缺少'loss'或'logits'键")
                
                targets = batch.get('target_genes')
                if targets is None:
                    raise KeyError("批次数据中缺少'target_genes'键")

                total_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                self._logger.debug(f"手动计算损失={total_loss:.4f}")
            
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
        # The hierarchical model returns 'predictions' for the final scale during training/validation,
        # and 'generated_sequence' during pure inference.
        if 'predictions' in results_dict:
            predictions = results_dict['predictions']
        elif 'generated_sequence' in results_dict:
            predictions = results_dict['generated_sequence']
        else:
            raise ValueError("模型输出中必须包含 'predictions' 或 'generated_sequence'")
        
        # 获取目标
        if 'target_genes' not in batch:
            raise ValueError("批次数据中找不到target_genes")
        targets = batch['target_genes']
        
        # 验证最终预测的维度是否为200
        num_genes = self.model.num_genes
        if predictions.shape[-1] != num_genes:
            raise ValueError(
                f"最终预测维度({predictions.shape[-1]})与目标基因数量({num_genes})不匹配！"
            )
        
        return predictions.float(), targets.float()

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        weight_decay = float(self.model_utils.get_config('TRAINING.weight_decay', DEFAULT_WEIGHT_DECAY))
        learning_rate = float(self.model_utils.get_config('TRAINING.learning_rate', DEFAULT_LEARNING_RATE))
        
        # 多GPU学习率缩放
        learning_rate = self.model_utils.scale_learning_rate(learning_rate)
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 设置梯度裁剪
        self.trainer.gradient_clip_val = self.model_utils.get_config('TRAINING.gradient_clip_val', DEFAULT_GRADIENT_CLIP)
        
        # 配置学习率调度器
        scheduler_config = self.model_utils.get_scheduler_config(optimizer)
        
        if scheduler_config:
            return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}
        else:
            return {'optimizer': optimizer}

    def on_train_epoch_end(self):
        """训练epoch结束时的回调"""
        pass  # 训练数据不再累积
    
    def on_validation_epoch_end(self):
        """验证epoch结束时的回调"""
        self._compute_and_log_pcc_metrics('val')
    
    def on_test_epoch_end(self):
        """测试epoch结束时的回调"""
        self._compute_and_log_pcc_metrics('test')
    
    def _compute_and_log_pcc_metrics(self, phase: str):
        """统一的PCC指标计算和记录方法"""
        # 修复属性名映射
        if phase == 'val':
            outputs_attr = 'validation_step_outputs'
        elif phase == 'test':
            outputs_attr = 'test_step_outputs'
        else:
            if self.trainer.is_global_zero:
                print(f"⚠️ 不支持的阶段: {phase}")
            return
        
        if not hasattr(self, outputs_attr):
            if self.trainer.is_global_zero:
                print(f"⚠️ 没有{phase}阶段的输出数据属性: {outputs_attr}")
            return
            
        outputs = getattr(self, outputs_attr)
        if not outputs:
            if self.trainer.is_global_zero:
                print(f"⚠️ {phase}阶段输出列表为空 (可能是sanity check阶段)")
            return
        
        try:
            # 收集所有数据
            all_predictions = []
            all_targets = []
            
            for output in outputs:
                all_predictions.append(output['predictions'])
                all_targets.append(output['targets'])
            
            # 合并数据
            predictions = torch.cat(all_predictions, dim=0)  # [N, genes]
            targets = torch.cat(all_targets, dim=0)  # [N, genes]
            
            self._logger.info(f"{phase}阶段收集到 {predictions.shape[0]} 个样本，{predictions.shape[1]} 个基因")
            
            # 计算PCC指标 - 数据是原始token计数值，需要应用log2变换
            pcc_metrics = self.model_metrics.calculate_comprehensive_pcc_metrics(predictions, targets, apply_log2=True)
            
            # 记录PCC指标到wandb
            total_samples = predictions.shape[0]
            for metric_name, value in pcc_metrics.items():
                self.log(f'{phase}_{metric_name}', value, 
                        on_epoch=True, 
                        prog_bar=False, 
                        batch_size=total_samples,
                        sync_dist=True)
            
            # 在主进程打印详细结果
            if self.trainer.is_global_zero:
                phase_loss = self.trainer.callback_metrics.get(f'{phase}_loss', 0.0)
                
                print(f"\n🎯 Epoch {self.current_epoch} {phase.upper()}结果:")
                print(f"   Loss: {phase_loss:.6f}")
                print(f"   PCC-10:  {pcc_metrics['pcc_10']:.4f}")
                print(f"   PCC-50:  {pcc_metrics['pcc_50']:.4f}")
                print(f"   PCC-200: {pcc_metrics['pcc_200']:.4f}")
                print(f"   MSE:     {pcc_metrics['mse']:.6f}")
                print(f"   MAE:     {pcc_metrics['mae']:.6f}")
                print(f"   RVD:     {pcc_metrics['rvd']:.6f}")
                print()
            
            # 清理输出数据
            outputs.clear()
            
        except Exception as e:
            self._logger.error(f"计算{phase}阶段PCC指标时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def on_fit_end(self):
        """训练完成时的回调"""
        if not self.trainer.is_global_zero:
            self._logger.info(f"GPU进程 {self.trainer.global_rank}: 训练完成")
            return
        
        self._logger.info("训练完成！")





    def on_before_optimizer_step(self, optimizer):
        """优化器步骤前的回调"""
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.parameters(), 
            self.trainer.gradient_clip_val
        )
        
        self.log('grad_norm', grad_norm, sync_dist=True)