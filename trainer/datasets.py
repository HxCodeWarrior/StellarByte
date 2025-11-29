import json
import random
import re
from typing import Dict, List, Tuple, Optional, Union
import logging

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import mmap
from tqdm import tqdm

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """数据集配置类"""
    max_length: int = 512
    buffer_size: int = 8192  # 预读取缓冲区大小
    num_workers: int = min(4, os.cpu_count() or 1)  # 数据加载工作进程数
    shuffle: bool = True
    seed: int = 42
    truncation_side: str = "right"  # 截断方向

class BaseDataset(Dataset):
    """基础数据集类，提供通用功能"""
    
    def __init__(self, tokenizer, config: DatasetConfig):
        self.tokenizer = tokenizer
        self.config = config
        self._setup_tokenizer_special_tokens()
    
    def _setup_tokenizer_special_tokens(self):
        """确保tokenizer有必要的特殊token"""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = "<bos>"
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = "<eos>"
    
    def load_jsonl_mmap(self, path: str) -> List[Dict]:
        """使用内存映射高效加载大文件"""
        samples = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                # 使用mmap提高大文件读取性能
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmap_obj:
                    for line_num, line in enumerate(iter(mmap_obj.readline, b""), 1):
                        try:
                            line_str = line.decode('utf-8').strip()
                            if not line_str:
                                continue
                            data = json.loads(line_str)
                            samples.append(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"JSON解析错误 行 {line_num}: {e}")
                        except Exception as e:
                            logger.warning(f"行 {line_num} 处理错误: {e}")
                            
        except FileNotFoundError:
            logger.error(f"文件未找到: {path}")
            raise
        except Exception as e:
            logger.error(f"加载文件 {path} 时出错: {e}")
            raise
            
        logger.info(f"从 {path} 加载了 {len(samples)} 个样本")
        return samples
    
    def load_data(self, path: str) -> List[Dict]:
        """加载数据，支持单个文件或目录"""
        if os.path.isdir(path):
            # 如果是目录，加载所有jsonl文件
            all_samples = []
            for filename in os.listdir(path):
                if filename.endswith('.jsonl') or filename.endswith('.json'):
                    file_path = os.path.join(path, filename)
                    all_samples.extend(self.load_jsonl_mmap(file_path))
            return all_samples
        else:
            return self.load_jsonl_mmap(path)
    
    def safe_tokenize(self, text: str, **kwargs) -> Dict:
        """安全的tokenize方法，处理异常"""
        try:
            return self.tokenizer(
                text,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                **kwargs
            )
        except Exception as e:
            logger.error(f"Tokenization错误: {e}")
            # 返回空tensor作为fallback
            empty_tensor = torch.zeros((self.config.max_length,), dtype=torch.long)
            return {'input_ids': empty_tensor.unsqueeze(0)}

class PretrainDataset(BaseDataset):
    """预训练数据集"""
    
    def __init__(self, data_path: str, tokenizer, config: DatasetConfig = None):
        config = config or DatasetConfig()
        super().__init__(tokenizer, config)
        self.samples = self.load_data(data_path)
        self._precompute_samples()
    
    def _precompute_samples(self):
        """预计算样本以提高性能"""
        self.processed_samples = []
        for sample in tqdm(self.samples, desc="预处理预训练数据"):
            try:
                encoding = self.safe_tokenize(str(sample.get('text', '')))
                input_ids = encoding.input_ids.squeeze()
                
                if len(input_ids) < 2:  # 跳过过短的样本
                    continue
                    
                loss_mask = (input_ids != self.tokenizer.pad_token_id)
                
                self.processed_samples.append({
                    'input_ids': input_ids,
                    'loss_mask': loss_mask
                })
            except Exception as e:
                logger.warning(f"预处理样本时出错: {e}")
                continue
    
    def __len__(self):
        return len(self.processed_samples)
    
    def __getitem__(self, index):
        sample = self.processed_samples[index]
        input_ids = sample['input_ids']
        loss_mask = sample['loss_mask']
        
        X = input_ids[:-1].clone().detach().long()
        Y = input_ids[1:].clone().detach().long()
        loss_mask = loss_mask[1:].clone().detach().long()
        
        return X, Y, loss_mask

class ChatDataset(BaseDataset):
    """对话数据集基类"""
    
    def __init__(self, tokenizer, config: DatasetConfig = None):
        config = config or DatasetConfig(max_length=1024)
        super().__init__(tokenizer, config)
        self._setup_special_token_ids()
    
    def _setup_special_token_ids(self):
        """设置特殊token ID"""
        try:
            self.bos_id = self.tokenizer(
                f'{self.tokenizer.bos_token}assistant', 
                add_special_tokens=False
            ).input_ids
            self.eos_id = self.tokenizer(
                f'{self.tokenizer.eos_token}', 
                add_special_tokens=False
            ).input_ids
        except Exception as e:
            logger.warning(f"设置特殊token ID失败: {e}")
            # 使用fallback值
            self.bos_id = [self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id else [1]
            self.eos_id = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [2]
    
    def _create_chat_prompt(self, conversations: List[Dict], tools: Optional[List] = None) -> str:
        """创建聊天提示"""
        try:
            return self.tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False,
                tools=tools
            )
        except Exception as e:
            logger.error(f"创建聊天提示时出错: {e}")
            # Fallback: 手动构建提示
            return self._build_fallback_prompt(conversations)
    
    def _build_fallback_prompt(self, conversations: List[Dict]) -> str:
        """构建fallback提示"""
        prompt = ""
        for msg in conversations:
            role = msg.get('role', '')
            content = msg.get('content', '')
            prompt += f"<|{role}|>{content}</s>"
        return prompt
    
    def _generate_attention_mask(self, input_ids: List[int]) -> List[int]:
        """生成注意力掩码"""
        return [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]
    
    def _generate_loss_mask(self, input_ids: List[int]) -> List[int]:
        """生成动态损失掩码 - 优化版本"""
        loss_mask = [0] * len(input_ids)
        bos_len = len(self.bos_id)
        eos_len = len(self.eos_id)
        
        i = 0
        while i < len(input_ids) - bos_len:
            # 检查是否匹配bos_id
            if input_ids[i:i + bos_len] == self.bos_id:
                start_idx = i + bos_len
                end_idx = start_idx
                
                # 寻找eos_id
                while end_idx <= len(input_ids) - eos_len:
                    if input_ids[end_idx:end_idx + eos_len] == self.eos_id:
                        break
                    end_idx += 1
                else:
                    # 没找到eos_id，使用序列结尾
                    end_idx = len(input_ids)
                
                # 设置损失掩码 (从bos后开始到eos前结束)
                mask_start = start_idx
                mask_end = min(end_idx, len(input_ids))
                
                for j in range(mask_start, mask_end):
                    if j < len(loss_mask):
                        loss_mask[j] = 1
                
                i = end_idx + eos_len if end_idx < len(input_ids) else len(input_ids)
            else:
                i += 1
        
        return loss_mask

class SFTDataset(ChatDataset):
    """监督微调数据集"""
    
    def __init__(self, jsonl_path: str, tokenizer, config: DatasetConfig = None):
        super().__init__(tokenizer, config)
        self.samples = self.load_data(jsonl_path)
        self._precompute_prompts()
    
    def _precompute_prompts(self):
        """预计算提示以加速训练"""
        self.processed_samples = []
        
        for sample in tqdm(self.samples, desc="预处理SFT数据"):
            try:
                conversations = sample.get('conversations', [])
                if not conversations:
                    continue
                
                tools = None
                if conversations and conversations[0].get("role") == "system" and conversations[0].get("functions"):
                    tools = conversations[0]["functions"]
                
                prompt = self._create_chat_prompt(conversations, tools)
                
                # Tokenize
                encoding = self.tokenizer(
                    prompt,
                    max_length=self.config.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoding.input_ids.squeeze().tolist()
                loss_mask = self._generate_loss_mask(input_ids)
                attention_mask = self._generate_attention_mask(input_ids)
                
                self.processed_samples.append({
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'loss_mask': torch.tensor(loss_mask, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
                })
                
            except Exception as e:
                logger.warning(f"预处理SFT样本时出错: {e}")
                continue
    
    def __len__(self):
        return len(self.processed_samples)
    
    def __getitem__(self, index):
        sample = self.processed_samples[index]
        input_ids = sample['input_ids']
        loss_mask = sample['loss_mask']
        
        X = input_ids[:-1]
        Y = input_ids[1:]
        loss_mask = loss_mask[1:]
        
        return X, Y, loss_mask

class DPODataset(ChatDataset):
    """DPO训练数据集"""
    
    def __init__(self, file_path: str, tokenizer, config: DatasetConfig = None):
        config = config or DatasetConfig(max_length=4096)
        super().__init__(tokenizer, config)
        self.data = self.load_data(file_path)
        self._precompute_pairs()
    
    def _precompute_pairs(self):
        """预计算DPO对"""
        self.processed_pairs = []
        
        for item in tqdm(self.data, desc="预处理DPO数据"):
            try:
                chosen = item.get('chosen', [])
                rejected = item.get('rejected', [])
                
                if not chosen or not rejected:
                    continue
                
                # 处理chosen
                chosen_prompt = self._create_chat_prompt(chosen)
                chosen_encoding = self.safe_tokenize(chosen_prompt)
                chosen_input_ids = chosen_encoding.input_ids.squeeze().tolist()
                chosen_loss_mask = self._generate_loss_mask(chosen_input_ids)
                
                # 处理rejected
                rejected_prompt = self._create_chat_prompt(rejected)
                rejected_encoding = self.safe_tokenize(rejected_prompt)
                rejected_input_ids = rejected_encoding.input_ids.squeeze().tolist()
                rejected_loss_mask = self._generate_loss_mask(rejected_input_ids)
                
                self.processed_pairs.append({
                    'chosen': {
                        'input_ids': torch.tensor(chosen_input_ids, dtype=torch.long),
                        'loss_mask': torch.tensor(chosen_loss_mask, dtype=torch.long)
                    },
                    'rejected': {
                        'input_ids': torch.tensor(rejected_input_ids, dtype=torch.long),
                        'loss_mask': torch.tensor(rejected_loss_mask, dtype=torch.long)
                    }
                })
                
            except Exception as e:
                logger.warning(f"预处理DPO样本时出错: {e}")
                continue
    
    def __len__(self):
        return len(self.processed_pairs)
    
    def __getitem__(self, index):
        pair = self.processed_pairs[index]
        
        chosen_ids = pair['chosen']['input_ids']
        chosen_mask = pair['chosen']['loss_mask']
        rejected_ids = pair['rejected']['input_ids']
        rejected_mask = pair['rejected']['loss_mask']
        
        return {
            'x_chosen': chosen_ids[:-1],
            'y_chosen': chosen_ids[1:],
            'mask_chosen': chosen_mask[1:],
            'x_rejected': rejected_ids[:-1],
            'y_rejected': rejected_ids[1:],
            'mask_rejected': rejected_mask[1:]
        }

class RLAIFDataset(ChatDataset):
    """RLAIF训练数据集"""
    
    def __init__(self, jsonl_path: str, tokenizer, config: DatasetConfig = None):
        super().__init__(tokenizer, config)
        self.samples = self.load_data(jsonl_path)
    
    def _create_chat_prompt(self, conversations: List[Dict]) -> Tuple[str, str]:
        """构建对话提示并返回答案"""
        messages = []
        answer = ''
        
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            content = turn.get('content', '')
            messages.append({"role": role, "content": content})
            
            if i == len(conversations) - 1:  # 最后一个turn是答案
                answer = content
        
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"创建RLAIF提示时出错: {e}")
            prompt = self._build_fallback_prompt(messages[:-1])
        
        return prompt, answer
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample.get('conversations', [])
        
        if len(conversations) < 2:
            # 返回空数据作为fallback
            return {'prompt': '', 'answer': ''}
        
        prompt, answer = self._create_chat_prompt(conversations)
        
        return {
            'prompt': prompt,
            'answer': answer,
            'conversations': conversations  # 保留原始对话用于调试
        }

class DatasetFactory:
    """数据集工厂类"""
    
    @staticmethod
    def create_dataset(dataset_type: str, data_path: str, tokenizer, **kwargs) -> Dataset:
        """创建数据集实例"""
        config = kwargs.get('config', DatasetConfig())
        
        dataset_map = {
            'pretrain': PretrainDataset,
            'sft': SFTDataset,
            'dpo': DPODataset,
            'rlaif': RLAIFDataset
        }
        
        if dataset_type not in dataset_map:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
        
        return dataset_map[dataset_type](data_path, tokenizer, config)
    
    @staticmethod
    def create_dataloader(dataset: Dataset, batch_size: int = 32, **kwargs) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=kwargs.get('shuffle', True),
            num_workers=kwargs.get('num_workers', 0),
            pin_memory=kwargs.get('pin_memory', True),
            drop_last=kwargs.get('drop_last', False),
            persistent_workers=kwargs.get('persistent_workers', True) if kwargs.get('num_workers', 0) > 0 else False
        )

# 使用示例
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
    
    # 创建配置
    config = DatasetConfig(
        max_length=512,
        num_workers=4,
        shuffle=True
    )
    
    # 使用工厂创建数据集
    try:
        sft_dataset = DatasetFactory.create_dataset(
            'pretrain', 
            './datasets/test/test_train.jsonl', 
            tokenizer, 
            config=config
        )
        
        print(f"数据集大小: {len(sft_dataset)}")
        
        # 输出第一条数据
        if len(sft_dataset) > 0:
            print("\n=== 第一条数据样本 ===")
            X, Y, loss_mask = sft_dataset[0]
            print(f"输入形状: {X.shape}")
            print(f"目标形状: {Y.shape}")
            print(f"损失掩码形状: {loss_mask.shape}")
            
            # 解码输入文本
            input_text = tokenizer.decode(X, skip_special_tokens=False)
            print(f"\n输入文本: {input_text}")
            
            # 解码目标文本
            target_text = tokenizer.decode(Y, skip_special_tokens=False)
            print(f"目标文本: {target_text}")
            
            # 显示损失掩码统计
            print(f"损失掩码中1的数量: {loss_mask.sum().item()}/{len(loss_mask)}")
            print(f"损失掩码比例: {loss_mask.sum().item()/len(loss_mask):.3f}")
            
            # 显示前20个token的详细信息
            print(f"\n前20个token的详细信息:")
            for i in range(min(20, len(X))):
                token_id = X[i].item()
                target_id = Y[i].item()
                mask_val = loss_mask[i].item()
                token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                target_str = tokenizer.decode([target_id], skip_special_tokens=False)
                print(f"位置{i:3d}: 输入ID={token_id:5d}('{token_str:10s}') -> 目标ID={target_id:5d}('{target_str:10s}') | 损失掩码={mask_val}")
        
        # 创建数据加载器
        dataloader = DatasetFactory.create_dataloader(
            sft_dataset,
            batch_size=16,
            num_workers=4,
            pin_memory=True
        )
        
        # 测试一个batch
        print(f"\n=== 测试第一个Batch ===")
        for batch_idx, batch in enumerate(dataloader):
            X, Y, loss_mask = batch
            print(f"Batch {batch_idx} - X: {X.shape}, Y: {Y.shape}, mask: {loss_mask.shape}")
            
            # 显示batch统计信息
            print(f"Batch中有效token比例: {loss_mask.float().mean().item():.3f}")
            
            # 只显示第一个batch的详细信息
            if batch_idx == 0:
                # 显示batch中第一个样本的部分信息
                sample_idx = 0
                print(f"\nBatch中第一个样本的前10个token:")
                for i in range(min(10, X.shape[1])):
                    token_id = X[sample_idx, i].item()
                    target_id = Y[sample_idx, i].item()
                    mask_val = loss_mask[sample_idx, i].item()
                    token_str = tokenizer.decode([token_id], skip_special_tokens=False)
                    print(f"位置{i:3d}: 输入='{token_str:10s}' | 目标ID={target_id:5d} | 掩码={mask_val}")
            break
            
    except Exception as e:
        logger.error(f"数据集创建失败: {e}")
        import traceback
        traceback.print_exc()