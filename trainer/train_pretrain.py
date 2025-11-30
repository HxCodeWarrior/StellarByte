"""
StellarByte LLM 预训练脚本

功能特性：
- 支持分布式训练 (DDP)
- 完整的检查点保存与恢复
- 可配置的优化器和学习率调度
- 详细的训练日志和进度跟踪
- 内存优化和训练稳定性保障
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.extend([project_root, os.path.join(project_root, 'model')])

import yaml
import argparse
import math
import time
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from contextlib import nullcontext

from model.config import StellarByteConfig
from model.StellarByteLm import StellarByteForCausalLM
from trainer.datasets import DatasetConfig, DatasetFactory
from trainer.utils.modelutils import build_optimizer, build_scheduler, init_model_and_tokenizer, init_swanlab, count_parameters
from trainer.utils.checkpoint import CheckpointManager
from trainer.utils.logger import Logger, is_main_process
from trainer.utils.distributed import init_distributed_mode, get_rank, get_world_size, broadcast_object
from trainer.utils.seed import setup_seed
from trainer.utils.samplers import SkipBatchSampler


class PretrainTrainer:
    """预训练器类"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.local_rank = init_distributed_mode()
        self.device = torch.device(
            self.config['device'] if self.config['device'] != "cuda" or torch.cuda.is_available() else "cpu"
        )
        
        # 设置混合精度
        self.dtype = torch.bfloat16 if self.config.get('dtype', 'bfloat16') == "bfloat16" else torch.float16
        self.autocast_ctx = nullcontext() if self.device.type == "cpu" else torch.amp.autocast(dtype=self.dtype)
        self.scaler = torch.amp.GradScaler(enabled=(self.dtype == torch.float16))
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        
        # 恢复训练
        if self.config['resume_training']:
            self._resume_training()
            
        # 设置随机种子
        setup_seed(self.config['seed'] + get_rank())
        
        # 初始化日志
        self.logger = Logger(
            self.config['log_file'] if get_rank() == 0 else None
        )
        self.logger.info(f"初始化预训练器 (rank {get_rank()}, world_size {get_world_size()})")
        
        # 初始化检查点管理器
        self.checkpoint_manager = CheckpointManager(self.config['checkpoint_dir'], self.logger)
        
        # 初始化模型和配置
        self._init_model_and_tokenizer()
        
        # 初始化优化器和调度器
        self._init_optimizer_and_scheduler()
        
        # 初始化数据集
        self._init_datasets()

        # 初始化SwanLab
        self.use_swanlab = self.config['use_swanlab']
        self.swanlab = None
        if self.config['use_swanlab'] and is_main_process():
            self.swanlab = init_swanlab(
                project=self.config['project'],
                workspace=self.config['workspace'],
                experiment_name=self.config['experiment_name'],
                api_key=self.config['api_key'],
                config=config,
            )

            if self.swanlab is None and self.config['use_swanlab']:
                print("警告: SwanLab初始化失败，将继续训练但不记录指标")
    
    def _init_model_and_tokenizer(self):
        """初始化模型和分词器"""
        self.logger.info("初始化模型和分词器...")
        
        # 创建模型配置
        model_config = StellarByteConfig(
            vocab_size=self.config['vocab_size'],
            hidden_size=self.config['hidden_size'],
            hidden_act=self.config['hidden_act'],
            num_hidden_layers=self.config['num_hidden_layers'],
            intermediate_size=self.config['intermediate_size'],
            num_attention_heads=self.config['num_attention_heads'],
            num_key_value_heads=self.config['num_key_value_heads'],
            flash_attn=self.config['flash_attn'],
            max_position_embeddings=self.config['max_length'],
            inference_rope_scaling=self.config['inference_rope_scaling'],
            dropout=float(self.config['dropout']),
            rms_norm_eps=float(self.config['rms_norm_eps']),
            use_moe=self.config['use_moe'],
            num_experts_per_tok=self.config['num_experts_per_tok'] if self.config['use_moe'] else 0,
            n_routed_experts=self.config['num_router_experts'] if self.config['use_moe'] else 0,
            n_shared_experts=self.config['num_shared_experts'] if self.config['use_moe'] else 0,
            scoring_func=self.config['scoring_func'] if self.config['use_moe'] else None,
            aux_loss_alpha=float(self.config['aux_loss_alpha']),
            seq_au=self.config['seq_au'] if self.config['use_moe'] else None,
            norm_topk_prob=self.config['norm_topk_prob'] if self.config['use_moe'] else True,
            initializer_range=float(self.config['initializer_range']),
            use_cache=self.config['use_cache'],
            pad_token_id=self.config['pad_token_id'],
            bos_token_id=self.config['bos_token_id'],
            eos_token_id=self.config['eos_token_id'],
            tie_word_embeddings=self.config['tie_word_embeddings'],
        )
        
        # 加载模型和分词器
        from_weight = "pretrain" if self.config['resume_training'] else "none"
        self.model, self.tokenizer = init_model_and_tokenizer(
            lm_config=model_config,
            tokenizer_path=self.config['tokenizer_path'],
            model_class=StellarByteForCausalLM,
            from_weight=from_weight,
            checkpoint_dir=self.config['checkpoint_dir'],
            device=self.device,
            strict=False,
            logger=self.logger
        )
        
        # 包装为 DDP
        if get_world_size() > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if self.local_rank != -1 else None,
                output_device=self.local_rank if self.local_rank != -1 else None,
                find_unused_parameters=self.config['use_moe']  # MoE 需要此参数
            )
        
        self.logger.info(f"模型参数量: {count_parameters(self.model)['total_parameters']:,}")
    
    def _init_optimizer_and_scheduler(self):
        """初始化优化器和调度器"""
        self.logger.info("初始化优化器和调度器...")
        
        # 优化器配置
        optim_cfg = {
            'type': 'AdamW',
            'lr': float(self.config['learning_rate']),
            'weight_decay': float(self.config['weight_decay']),
        }
        
        # 调度器配置
        scheduler_cfg = {
            'type': self.config['scheduler_type'],
        }
        
        # 构建优化器和调度器
        model_for_opt = self.model.module if hasattr(self.model, 'module') else self.model
        self.optimizer = build_optimizer(model_for_opt, optim_cfg)
        self.scheduler = build_scheduler(
            self.optimizer, 
            scheduler_cfg, 
            num_training_steps=self.config['total_steps'],
            num_warmup_steps=self.config['warmup_steps']
        )
    
    def _init_datasets(self):
        """初始化数据集"""
        self.logger.info("初始化数据集...")
        
        # 数据集配置
        dataset_config = DatasetConfig(
            max_length=self.config['max_length'],
            shuffle=True,
            seed=self.config['seed'] + get_rank()
        )
        
        # 训练数据集
        self.train_dataset = DatasetFactory.create_dataset(
            'pretrain',
            self.config['data_path'],
            self.tokenizer,
            config=dataset_config
        )
        
        self.logger.info(f"训练数据集大小: {len(self.train_dataset):,}")
        
        # 数据加载器
        if get_world_size() > 1:
            # 分布式采样器
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=True
            )
        else:
            from torch.utils.data import RandomSampler
            train_sampler = RandomSampler(self.train_dataset)
        
        # 包装为 SkipBatchSampler 以支持恢复训练
        self.train_sampler = SkipBatchSampler(
            train_sampler,
            batch_size=self.config['batch_size'],
            skip_batches=self.global_step // self.config['gradient_accumulation_steps']
        )
        
        self.train_loader = DatasetFactory.create_dataloader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            sampler=self.train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
    
    def _resume_training(self):
        """恢复训练状态"""
        self.logger.info("尝试恢复训练...")
        
        resume_data = self.checkpoint_manager.load(
            "pretrain",
            self.model.module.config if hasattr(self.model, 'module') else self.model.config,
            device=self.device
        )
        
        if resume_data is not None:
            # 加载优化器状态
            if resume_data.get('optimizer') is not None:
                self.optimizer.load_state_dict(resume_data['optimizer'])
            
            # 加载scaler状态
            if resume_data.get('scaler') is not None and self.scaler:
                self.scaler.load_state_dict(resume_data['scaler'])

            # 恢复训练状态
            self.global_step = resume_data.get('step', 0)
            self.current_epoch = resume_data.get('epoch', 0)
            self.best_loss = resume_data.get('best_loss', float('inf'))
            
            # 调整调度器步数
            if self.scheduler and 'scheduler' in resume_data:
                self.scheduler.load_state_dict(resume_data['scheduler'])
            
            self.logger.info(f"恢复训练成功: epoch={self.current_epoch}, step={self.global_step}")
        else:
            self.logger.info("未找到检查点，从头开始训练")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        epoch_start_time = time.time()
        
        # 进度条（仅主进程显示）
        if get_rank() == 0:
            from tqdm import tqdm
            pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {self.current_epoch}")
        else:
            pbar = None
        
        for batch_idx, (X, Y, loss_mask) in enumerate(self.train_loader):
            # 移动到设备
            X = X.to(self.device, non_blocking=True)
            Y = Y.to(self.device, non_blocking=True)
            loss_mask = loss_mask.to(self.device, non_blocking=True)
            
            # 梯度累积
            with self.autocast_ctx:
                outputs = self.model(X, labels=Y)
                loss = nn.CrossEntropyLoss(reduction='none')(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())

                # 应用损失掩码
                loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)

                # MoE 辅助损失
                if hasattr(outputs, 'aux_loss') and outputs.aux_loss is not None:
                    loss = loss + outputs.aux_loss * self.config.get('aux_loss_alpha', 0.01)

                loss = loss / self.config['gradient_accumulation_steps']
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 梯度累积步骤
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['max_grad_norm']
                )
                
                # 优化器步进
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                # 更新全局步数
                self.global_step += 1
                
                # 记录日志
                if self.global_step % self.config['log_steps'] == 0 and get_rank() == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    current_loss = loss.item() * self.config['gradient_accumulation_steps']
                    spend_time = time.time() - epoch_start_time
                    eta_min = spend_time / (batch_idx + 1) * len(self.train_loader) // 60 - spend_time // 60

                    self.logger.info(
                        f"Epoch:[{self.current_epoch+1}/{self.config['epochs']}]"
                        f"({self.global_step}/{self.config['total_steps']}) "
                        f"loss:{current_loss:.6f} lr:{current_lr:.12f} eta:{eta_min}min"
                    )

                    # SwanLab记录
                    if self.use_swanlab and self.swanlab:
                        self.swanlab.log({
                            "train/loss": current_loss,
                            "train/learning_rate": current_lr,
                            "train/step": self.global_step,
                            "train/epoch": self.current_epoch + 1,
                            "train/tokens_per_step": X.numel() * get_world_size(),
                        })
                
                # 保存检查点
                if self.global_step % self.config['save_steps'] == 0 and get_rank() == 0:
                    self._save_checkpoint()
            
            # 统计信息
            total_loss += loss.item() * self.config['gradient_accumulation_steps']
            total_tokens += X.numel()
            
            # 更新进度条
            if pbar:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{loss.item()*self.config["gradient_accumulation_steps"]:.4f}',
                    'step': self.global_step
                })
            
            # 提前结束（用于调试）
            if os.environ.get('DEBUG') and batch_idx > 10:
                break
        
        if pbar:
            pbar.close()
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / len(self.train_loader)
        
        self.logger.info(
            f"Epoch {self.current_epoch} 完成: "
            f"平均损失={avg_loss:.4f}, "
            f"耗时={epoch_time:.2f}s, "
            f"处理tokens={total_tokens:,}"
        )
        
        return avg_loss
    
    def _save_checkpoint(self):
        """保存检查点"""
        self.logger.info(f"保存检查点 (step={self.global_step})...")
        
        # 准备额外数据
        extras = {
            'best_loss': self.best_loss,
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict(),
        }
        
        # 保存检查点
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        self.checkpoint_manager.save(
            prefix="pretrain",
            lm_config=model_to_save.config,
            model=model_to_save,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            step=self.global_step,
            extras=extras
        )
    
    def train(self):
        """主训练循环"""
        self.logger.info("开始训练...")
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config['epochs']):
                self.current_epoch = epoch
                
                # 设置采样器epoch（用于分布式训练）
                if hasattr(self.train_sampler, 'set_epoch'):
                    self.train_sampler.set_epoch(epoch)
                
                # 训练一个epoch
                avg_loss = self.train_epoch()
                
                # 更新最佳损失
                if avg_loss < self.best_loss and get_rank() == 0:
                    self.best_loss = avg_loss
                    self.logger.info(f"新的最佳损失: {self.best_loss:.4f}")
                    if self.use_swanlab and self.swanlab:
                        self.swanlab.log({"train/best_loss": self.best_loss})
                
                # 检查是否达到总步数限制
                if self.global_step >= self.config['total_steps']:
                    self.logger.info(f"达到总步数限制: {self.config['total_steps']}")
                    break
            
            # 训练完成，保存最终模型
            if get_rank() == 0:
                self._save_checkpoint()
                self.logger.info("训练完成!")
        
        except KeyboardInterrupt:
            self.logger.info("训练被中断")
            if get_rank() == 0:
                self._save_checkpoint()
        
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {e}")
            raise
        
        finally:
            total_time = time.time() - start_time
            self.logger.info(f"总训练时间: {total_time:.2f}s")
            
            # 清理分布式训练
            if get_world_size() > 1:
                torch.distributed.destroy_process_group()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="StellarByte LLM 预训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--resume_training", action="store_true", help="恢复训练")
    args = parser.parse_args()
    
    # 加载配置文件
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建训练器并开始训练
    trainer = PretrainTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()