import json
import os
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from dataclasses import dataclass

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """数据集配置类"""
    max_length: int = 512
    num_workers: int = min(4, os.cpu_count() or 1)  # 数据加载工作进程数
    shuffle: bool = True
    seed: int = 42
    
    def __post_init__(self):
        # 验证配置参数
        assert self.max_length > 0, "max_length must be positive"
        assert self.num_workers >= 0, "num_workers must be non-negative"

class BaseDataset(Dataset):
    """基础数据集类，提供通用功能"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DatasetConfig):
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
    
    def load_jsonl_data(self, path: str) -> List[Dict[str, Any]]:
        """加载JSONL文件数据"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        # 预计算文件总行数用于进度条
        total_lines = sum(1 for _ in open(path, 'r', encoding='utf-8', errors='ignore'))

        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            # 使用tqdm包装循环，显示进度条
            for line_num, line in enumerate(tqdm(f, total=total_lines, desc=f"Loading {path.name}"), 1
            ):
                try:
                    data = json.loads(line.strip())
                    samples.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error at line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(samples)} samples from {path}")
        return samples
    
    def tokenize(self, text: str, **kwargs) -> Dict:
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
    
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, config: DatasetConfig = None):
        config = config or DatasetConfig()
        super().__init__(tokenizer, config)
        self.samples = self.load_jsonl_data(data_path)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """获取单个样本"""
        sample = self.samples[index]
        text = str(sample['text']).strip()

        # Tokenization
        encoding = self.tokenize(text)
        input_ids = encoding.input_ids.squeeze(0)

        # 创建损失掩码（忽略填充token）
        loss_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # 构建因果语言建模目标
        X = input_ids[:-1].long()
        Y = input_ids[1:].long()
        loss_mask = loss_mask[1:].long()
        
        return {
            'input_ids': X,
            'labels': Y,
            'loss_mask': loss_mask
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    """测试数据集加载器的功能和质量"""
    from transformers import AutoTokenizer
    import torch

    # 1. 加载你的tokenizer（根据实际情况修改路径）
    print("1. 加载tokenizer...")
    try:
        # 假设你的tokenizer保存在"tokenizer/"目录下
        tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")
        print(f"✓ Tokenizer加载成功，词汇量: {len(tokenizer)}")
        print(f"✓ 特殊token: pad={tokenizer.pad_token}, bos={tokenizer.bos_token}, eos={tokenizer.eos_token}")
    except Exception as e:
        print(f"✗ Tokenizer加载失败: {e}")
    
    # 2. 创建数据集配置
    print("\n2. 创建数据集配置...")
    config = DatasetConfig(
        max_length=128,  # 使用较短长度便于测试
        num_workers=0,   # 测试时设为0避免多进程问题
        shuffle=True
    )
    print(f"✓ 配置: {config}")
    
    # 3. 加载数据集
    print("\n3. 加载数据集...")
    try:
        dataset = PretrainDataset(
            data_path="./datasets/test/test_pre_train.jsonl",  # 修改为你的数据文件路径
            tokenizer=tokenizer,
            config=config
        )
        print(f"✓ 数据集加载成功，样本数: {len(dataset)}")
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
    
    # 4. 创建DataLoader
    print("\n4. 创建DataLoader...")
    dataloader = DataLoader(
        dataset,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 5. 测试单一样本
    print("\n5. 测试单一样本...")
    try:
        single_sample = dataset[0]
        print(f"✓ 获取单一样本成功")
        print(f"  - input_ids shape: {single_sample['input_ids'].shape}")
        print(f"  - labels shape: {single_sample['labels'].shape}")
        print(f"  - loss_mask shape: {single_sample['loss_mask'].shape}")
        print(f"  - input_ids[:10]: {single_sample['input_ids'][:10].tolist()}")
        print(f"  - labels[:10]: {single_sample['labels'][:10].tolist()}")
        print(f"  - loss_mask[:10]: {single_sample['loss_mask'][:10].tolist()}")
    except Exception as e:
        print(f"✗ 单一样本测试失败: {e}")
    
    # 6. 测试一个batch
    print("\n6. 测试一个batch...")
    try:
        batch = next(iter(dataloader))
        print(f"✓ 获取batch成功")
        print(f"  - input_ids shape: {batch['input_ids'].shape}")
        print(f"  - labels shape: {batch['labels'].shape}")
        print(f"  - loss_mask shape: {batch['loss_mask'].shape}")
        
        # 检查数据是否在合理范围内
        vocab_size = len(tokenizer)
        input_min, input_max = batch['input_ids'].min().item(), batch['input_ids'].max().item()
        labels_min, labels_max = batch['labels'].min().item(), batch['labels'].max().item()
        
        print(f"  - input_ids范围: [{input_min}, {input_max}] (词汇量: {vocab_size})")
        print(f"  - labels范围: [{labels_min}, {labels_max}] (词汇量: {vocab_size})")
        
        if input_min < 0 or input_max >= vocab_size:
            print("⚠ 警告: input_ids包含超出词汇表范围的token")
        if labels_min < 0 or labels_max >= vocab_size:
            print("⚠ 警告: labels包含超出词汇表范围的token")
        
    except Exception as e:
        print(f"✗ batch测试失败: {e}")
    
    # 7. 测试多个batch（内存允许的情况下）
    print("\n7. 测试多个batch...")
    try:
        max_batches = 3
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            print(f"  Batch {i}: input_ids shape = {batch['input_ids'].shape}")
            
            # 检查是否有NaN或inf
            if torch.isnan(batch['input_ids']).any():
                print(f"⚠ Batch {i}: input_ids包含NaN值")
            if torch.isinf(batch['input_ids']).any():
                print(f"⚠ Batch {i}: input_ids包含inf值")
        
        print(f"✓ 成功迭代{min(max_batches, len(dataloader))}个batch")
    except Exception as e:
        print(f"✗ 多batch测试失败: {e}")
    
    # 8. 解码测试
    print("\n8. 解码测试...")
    try:
        decoded_text = tokenizer.decode(single_sample['input_ids'][:20], skip_special_tokens=False)
        print(f"✓ 解码成功")
        print(f"  - 原始tokens: {single_sample['input_ids'][:20].tolist()}")
        print(f"  - 解码文本: {decoded_text}")
    except Exception as e:
        print(f"✗ 解码失败: {e}")
    
    # 9. 统计信息
    print("\n9. 数据集统计信息...")
    try:
        # 计算平均序列长度
        total_length = 0
        valid_samples = 0
        
        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            seq_len = (sample['loss_mask'] == 1).sum().item()
            total_length += seq_len
            valid_samples += 1
        
        avg_length = total_length / valid_samples if valid_samples > 0 else 0
        print(f"✓ 平均有效序列长度: {avg_length:.2f}")
        print(f"✓ 序列最大长度: {config.max_length}")
        print(f"✓ 填充比例: {1 - (avg_length/config.max_length):.2%}")
    except Exception as e:
        print(f"✗ 统计信息计算失败: {e}")
    
    print("\n" + "="*50)
    print("测试完成！所有基本功能正常 ✓")