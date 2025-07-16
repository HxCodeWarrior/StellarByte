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
        # 1) 保存配置
        self.tokenizer   = tokenizer
        self.max_length  = max_length

        # 2) 若 tokenizer 没有 pad_token，则动态注入一个占位符
        if tokenizer.pad_token is None:
            logging.warning("[BaseDataset] Tokenizer has no <pad>, adding '<|pad|>'.")
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            # ⬆️ 上层（model 初始化处）需调用 model.resize_token_embeddings

        # 3) 最终的 pad_token_id（注入后一定存在）
        self.pad_token_id: int = tokenizer.pad_token_id

    def _pad_and_mask(self, input_ids: List[int]) -> tuple[list[int], list[bool]]:
        """
        截断 / 填充，并生成损失掩码。

        Returns
        -------
        padded_ids : list[int]
            长度恰为 `max_length` 的 token 序列。
        loss_mask : list[bool]
            与 `padded_ids` 同长；真实 token = True，PAD = False。
        """
        # ---------- 1. 截断 ----------
        seq_len = len(input_ids)
        if seq_len > self.max_length:
            input_ids = input_ids[: self.max_length]

        # ---------- 2. 计算 pad 长度 ----------
        pad_len = self.max_length - len(input_ids)

        # ---------- 3. 拼接 PAD ----------
        padded_ids: List[int] = input_ids + [self.pad_token_id] * pad_len

        # ---------- 4. 生成掩码 ----------
        # loss_mask: List[int] = [1] * len(input_ids) + [0] * pad_len
        # 真实 token → True，PAD → False；dtype 使用 bool 更节约显存
        loss_mask: list[bool] = [True] * len(input_ids) + [False] * pad_len

        return padded_ids, loss_mask


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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.data[index]
        text = self._format_sample(sample)

        # 处理空文本
        if not text:
            text = "[EMPTY]"

        # 编码 & 截断
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if self.add_bos: # 添加 BOS（例如GPT风格）
            input_ids = [self.tokenizer.bos_token_id] + input_ids

        # 生成输入、标签、loss_mask
        input_ids, loss_mask = self._pad_and_mask(input_ids)

        input_tensor = torch.tensor(input_ids[:-1], dtype=torch.long)
        label_tensor = torch.tensor(input_ids[1:], dtype=torch.long)
        mask_tensor  = torch.tensor(loss_mask[1:], dtype=torch.bool)

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

    @staticmethod
    def _find_token_sequence(pattern: List[int], arr: List[int]) -> List[int]:
        """
        使用 KMP 在线性时间内查找 `pattern` 在 `arr` 中所有出现的起始索引。

        参数
        ----
        pattern : 要匹配的子串（token id 列表）
        arr     : 目标序列

        返回
        ----
        List[int] : 所有匹配起点，升序排列
        """
        if not pattern or len(pattern) > len(arr):
            return []

        # ---------- 1. 预处理前缀函数（最长真前后缀） ----------
        lps = [0] * len(pattern)
        j   = 0
        for i in range(1, len(pattern)):
            while j and pattern[i] != pattern[j]:
                j = lps[j - 1]
            if pattern[i] == pattern[j]:
                j += 1
                lps[i] = j

        # ---------- 2. 主串匹配 ----------
        res, j = [], 0
        for idx, token in enumerate(arr):
            while j and token != pattern[j]:
                j = lps[j - 1]
            if token == pattern[j]:
                j += 1
                if j == len(pattern):
                    res.append(idx - j + 1)  # 记录起点
                    j = lps[j - 1]           # 继续寻找下一个
        return res

    def generate_loss_mask(self, input_ids: List[int]) -> List[bool]:
        """
        针对 SFT 模型，精确标记 **assistant 回复正文** 所在 token 位置：
        仅这些位置参与 loss 计算。
        """
        # 1) 先找到 assistant 起始标记 & 结束标记的所有出现位置
        start_positions = self._find_token_sequence(
            self.assistant_start_tokens, input_ids
        )
        end_positions = self._find_token_sequence(
            self.im_end_token, input_ids
        )

        # 2) 初始化全 False 掩码（不算 loss）
        mask: list[bool] = [False] * len(input_ids)

        # 3) 为每个起始点配对最近的结束点，并设置区间 True
        for start in start_positions:
            content_start = start + len(self.assistant_start_tokens)

            # 找到位于内容后的第一个 `<|im_end|>`
            end = next((e for e in end_positions if e > content_start), len(input_ids))

            # 有效性检查：防止空区间或越界
            if content_start < end:
                for i in range(content_start, end):
                    mask[i] = True

        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        返回统一格式的字典：
        {
            "input_ids": [seq_len-1],
            "labels":    [seq_len-1],
            "loss_mask": [seq_len-1] (bool)
        }
        """
        sample = self.data[idx]

        # ---------- 1. 构造文本（支持 chat template） ----------
        # 若您需要 system/user/assistant 多轮，可在外部预处理
        text = self.tokenizer.apply_chat_template(
            sample,
            tokenize=False,
            add_generation_prompt=False
        ).strip()
        if not text:
            text = "[EMPTY]"

        # ---------- 2. tokenizer.encode（不加额外 special token） ----------
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length
        )

        # ---------- 3. 填充 + 通用掩码 ----------
        padded_ids, pad_mask = self._pad_and_mask(token_ids)

        # ---------- 4. SFT 专属掩码 ----------
        sft_mask = self.generate_loss_mask(padded_ids)

        # ---------- 5. 合并掩码（AND） ----------
        final_mask = [p and s for p, s in zip(pad_mask, sft_mask)]

        # ---------- 6. 构造训练对 (左移 1) ----------
        input_ids = torch.tensor(padded_ids[:-1], dtype=torch.bool)
        labels    = torch.tensor(padded_ids[1:],  dtype=torch.bool)
        loss_mask = torch.tensor(final_mask[1:], dtype=torch.bool)  # 与 labels 对齐

        return {
            "input_ids": input_ids,  # [seq_len-1]
            "labels":    labels,     # [seq_len-1]
            "loss_mask": loss_mask   # [seq_len-1]
        }


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
    sample = dataset[0]
    input_tensor = sample["input_ids"]
    label_tensor = sample["labels"]
    mask_tensor  = sample["loss_mask"]

    print("\n输入张量:",  input_tensor)
    print("标签张量:",  label_tensor)
    print("掩码张量:",  mask_tensor)
    
    # 5. 清理测试文件
    os.remove("test_data.jsonl")
