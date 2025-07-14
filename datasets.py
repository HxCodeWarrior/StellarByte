import os
import json
import torch
import logging
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from typing import List, Optional, Callable

class BaseDataset(Dataset):
    """
    通用数据集基类：提供序列填充与损失掩码生成逻辑
    """
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id or 0

    def _pad_and_mask(self, input_ids: List[int]) -> tuple[list[int], list[int]]:
        """
        填充序列并生成损失掩码
        """
        seq_len = len(input_ids)
        pad_len = self.max_length - seq_len
        input_ids += [self.pad_token_id] * pad_len
        loss_mask = [1] * seq_len + [0] * pad_len
        return input_ids, loss_mask


class PretrainDataset(BaseDataset):
    """
    预训练数据集：支持多字段拼接、自定义模板、自动加载 json/jsonl/csv
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        fields: Optional[List[str]] = None,
        template: Optional[str] = None,
        add_bos: bool = True
    ):
        """
        :param data_path: 数据文件路径（支持 .json / .jsonl / .csv）
        :param tokenizer: 分词器
        :param max_length: 最大长度（token级别）
        :param fields: 需要拼接的字段名，例如 ['input', 'thinking', 'output']
        :param template: 拼接模板，例如 "问：{input}\n想：{thinking}\n答：{output}"
        :param add_bos: 是否添加 BOS token
        """
        super().__init__(tokenizer, max_length)
        self.data_path = data_path
        self.fields = fields or ["input", "output"]
        self.template = template
        self.add_bos = add_bos
        self.bos_token = tokenizer.bos_token or tokenizer.eos_token or ""

        self.data = self._load_data()
        logging.info(f"[Dataset] Loaded {len(self)} samples from {data_path}")

    def _load_data(self) -> List[dict]:
        """
        加载 json/jsonl/csv 格式数据
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found: {self.data_path}")

        ext = os.path.splitext(self.data_path)[-1]
        if ext == ".jsonl":
            with open(self.data_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        elif ext == ".json":
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else list(data.values())
        elif ext == ".csv":
            df = pd.read_csv(self.data_path)
            return df.to_dict(orient="records")
        else:
            raise ValueError(f"Unsupported format: {ext}")

    def _format_sample(self, sample: dict) -> str:
        """
        将多个字段合并为单个文本
        """
        # 字段缺失降级为空字符串
        field_values = {k: str(sample.get(k, "")).strip() for k in self.fields}

        if self.template:
            try:
                text = self.template.format(**field_values)
            except KeyError as e:
                raise ValueError(f"Template字段缺失：{e}")
        else:
            # 默认拼接方式：input: xxx\noutput: xxx
            text = "\n".join([f"{k}: {v}" for k, v in field_values.items()])
        
        return text.strip()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.data[index]
        text = self._format_sample(sample)

        # 处理空文本
        if not text:
            text = "[EMPTY]"

        # 添加 BOS（例如GPT风格）
        if self.add_bos:
            text = self.bos_token + text

        # 编码 & 截断
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        input_ids = input_ids[:self.max_length]

        # 生成输入、标签、loss_mask
        input_ids, loss_mask = self._pad_and_mask(input_ids)

        input_tensor = torch.tensor(input_ids[:-1], dtype=torch.long)
        label_tensor = torch.tensor(input_ids[1:], dtype=torch.long)
        mask_tensor  = torch.tensor(loss_mask[1:], dtype=torch.long)

        return {
            "input_ids": input_tensor,     # [seq_len-1]
            "labels":    label_tensor,     # [seq_len-1]
            "loss_mask": mask_tensor,      # [seq_len-1]
        }

class SFTDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__(tokenizer, max_length)
        self.data_path = data_path
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        
        # 动态生成特殊标记序列
        self._init_special_tokens()
        logging.info(f"Loaded {len(self)} SFT samples")
        logging.info(f"Assistant start tokens: {self.assistant_start_tokens}")
        logging.info(f"Im_end token: {self.im_end_token}")

    def _init_special_tokens(self):
        """动态生成特殊标记序列，增强鲁棒性"""
        # 生成assistant起始标记
        assistant_start_str = "<|im_start|>assistant\n"
        self.assistant_start_tokens = self.tokenizer.encode(
            assistant_start_str, 
            add_special_tokens=False
        )
        
        # 生成消息结束标记
        im_end_str = "<|im_end|>"
        self.im_end_token = self.tokenizer.encode(
            im_end_str,
            add_special_tokens=False
        )

        # 验证特殊标记
        if not self.assistant_start_tokens:
            logging.warning("Assistant start tokens are empty!")
        if not self.im_end_token:
            logging.warning("Im_end token is empty!")

    def _find_token_sequence(self, sequence, input_ids):
        """高效查找所有序列出现位置"""
        if not sequence:
            return []
        
        # 转换为字符串匹配（更鲁棒）
        seq_str = self.tokenizer.decode(sequence)
        full_str = self.tokenizer.decode(input_ids)
        
        positions = []
        start_idx = 0
        while start_idx < len(full_str):
            pos = full_str.find(seq_str, start_idx)
            if pos == -1:
                break
            # 将字符位置转换为token位置
            token_pos = len(self.tokenizer.encode(full_str[:pos], add_special_tokens=False))
            positions.append(token_pos)
            start_idx = pos + len(seq_str)
        
        return positions

    def generate_loss_mask(self, input_ids):
        """生成精确的损失掩码"""
        mask = [0] * len(input_ids)
        
        # 查找所有assistant起始位置
        start_positions = self._find_token_sequence(
            self.assistant_start_tokens, 
            input_ids
        )
        
        # 查找所有结束标记位置
        end_positions = self._find_token_sequence(
            self.im_end_token,
            input_ids
        )
        
        # 为每个assistant片段生成掩码
        for start_idx in start_positions:
            content_start = start_idx + len(self.assistant_start_tokens)
            
            # 查找最近的结束标记
            end_idx = None
            for pos in end_positions:
                if pos > content_start:
                    end_idx = pos
                    break
            
            # 确定内容结束位置
            content_end = end_idx if end_idx is not None else len(input_ids)
            
            # 检查内容区域是否有效（非空）
            if content_start < content_end:
                # 排除特殊标记
                start = min(content_start, len(mask))
                end = min(content_end, len(mask))

                # 标记有效区域
                for i in range(start, end):
                    mask[i] = 1
                
        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        try:
            sample = self.data[index]
            text = sample.get('text', '')
    
            # 空文本处理（与PretrainDataset保持一致）
            if not text.strip():
                text = "[EMPTY]"

            text = self.tokenizer.apply_chat_template(
                sample, 
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Tokenize并截断
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tokens = tokens[:self.max_length]
            
            # 填充处理
            padded_ids, loss_mask = self._pad_sequence(tokens)
            
            # 生成SFT专用损失掩码
            sft_mask = self.generate_loss_mask(padded_ids)
            
            # 合并掩码（填充掩码+SFT掩码）
            final_mask = [m1 & m2 for m1, m2 in zip(loss_mask, sft_mask)]
            
            # 转换为numpy数组
            input_ids = np.array(padded_ids, dtype=np.int64)
            final_mask = np.array(final_mask, dtype=np.int64)
            
            # 创建训练对
            X = input_ids[:-1]
            Y = input_ids[1:]
            final_mask = final_mask[1:]
            
            return (
                torch.from_numpy(X),
                torch.from_numpy(Y),
                torch.from_numpy(final_mask)
            )
        
        except Exception as e:
            logging.error(f"Error processing sample {index}: {str(e)}")
            # 返回空样本
            empty = torch.zeros(self.max_length-1, dtype=torch.long)
            return empty, empty, empty

if __name__ == "__main__":
    # 1. 创建虚拟分词器
    class DummyTokenizer:
        bos_token = "[BOS]"
        eos_token = "[EOS]"
        pad_token_id = 0
        
        def encode(self, text, **kwargs):
            return [ord(c) for c in text]  # 简单字符编码
    
    # 2. 创建测试数据文件
    test_data = [
        {"input": "你好", "output": "世界"},
        {"input": "测试", "output": "数据集"}
    ]
    with open("test_data.jsonl", "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    
    # 3. 初始化数据集
    tokenizer = DummyTokenizer()
    dataset = PretrainDataset(
        data_path="test_data.jsonl",
        tokenizer=tokenizer,
        max_length=10,
        fields=["input", "output"],
        template="问:{input} 答:{output}"
    )
    
    # 4. 测试样本获取
    print(f"数据集大小: {len(dataset)}")
    input_tensor, label_tensor, mask_tensor = dataset[0]
    
    print("\n输入张量:", input_tensor)
    print("标签张量:", label_tensor)
    print("掩码张量:", mask_tensor)
    
    # 5. 清理测试文件
    os.remove("test_data.jsonl")
