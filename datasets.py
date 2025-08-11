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
        # 1. 保存配置
        self.tokenizer   = tokenizer
        self.max_length  = max_length

        # 2. 若 tokenizer 没有 pad_token，则动态注入一个占位符
        if tokenizer.pad_token is None:
            logging.warning("[BaseDataset] Tokenizer has no <pad>, adding '<|spad|>'.")
            tokenizer.add_special_tokens({"pad_token": "<|spad|>"})
        if tokenizer.bos_token is None:
            tokenizer.add_special_tokens({"bos_token": "<|sbos|>"})
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "<|seos|>"})

        # 3. 最终的 pad_token_id（注入后一定存在）
        self.pad_token_id: int = tokenizer.pad_token_id

        # 4. 预计算BOS/EOS
        self.has_bos = tokenizer.bos_token is not None
        self.has_eos = tokenizer.eos_token is not None

    def _pad_and_mask(self, input_ids: List[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        截断 / 填充，并生成损失掩码。

        Returns
        -------
        padded_ids     : list[int],长度恰为 `max_length` 的 token 序列。
        attention_mask : list[bool],与 `padded_ids` 同长；真实 token = True，PAD = False。
        loss_mask      : list[bool],与 `padded_ids` 同长；真实 token = True，PAD = False。
        """
        # ---------- 1. 截断 ----------
        input_ids = input_ids[: self.max_length]
        seq_len = len(input_ids)

        # ---------- 2. 计算 pad 长度 ----------
        pad_len = self.max_length - seq_len

        # ---------- 3. 拼接 PAD ----------
        padded_ids: List[int] = input_ids + [self.pad_token_id] * pad_len

        # ---------- 4. 生成掩码 ----------
        # mask: List[int] = [1] * len(input_ids) + [0] * pad_len
        # 真实 token → True，PAD → False；dtype 使用 bool 更节约显存
        attention_mask: list[int] = [1] * seq_len + [0] * pad_len
        loss_mask     : list[int] = [1] * seq_len + [0] * pad_len

        return (
            torch.tensor(padded_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.bool),
            torch.tensor(loss_mask, dtype=torch.bool)
        )


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
        add_bos: bool = True,
        add_eos: bool = True
    ):
        """
        :param data_path: 数据文件路径（支持 .json / .jsonl / .csv）
        :param tokenizer: 分词器
        :param max_length: 最大长度（token级别）
        :param fields: 需要拼接的字段名，例如 ['input', 'thinking', 'output']
        :param template: 拼接模板，例如 "问：{input}\n想：{thinking}\n答：{output}"
        :param add_bos: 是否添加 BOS token
        :param add_eos: 是否添加 EOS token
        """
        super().__init__(tokenizer, max_length)
        self.data_path = data_path
        self.fields = fields or ["input", "output"]
        self.template = template
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.bos_token_id = tokenizer.bos_token_id or None
        self.eos_token_id = tokenizer.eos_token_id or None

        if self.add_bos and self.bos_token_id is None:
            if self.eos_token_id is not None:
                logging.warning("[Dataset] Tokenizer has no <bos>, fallback to <eos> as BOS.")
                self.bos_token_id = self.eos_token_id
            else:
                raise ValueError("Tokenizer must have bos_token or eos_token if add_bos=True")

        if self.add_eos and self.eos_token_id is None:
            if self.bos_token_id is not None:
                logging.warning("[Dataset] Tokenizer has no <eos>, fallback to <eos> as BOS.")
                self.eos_token_id = self.bos_token_id
            else:
                raise ValueError("Tokenizer must have eos_token if add_eos=True")

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

        # BOS 和 EOS 预留长度
        reserve_len = int(self.add_bos) + int(self.add_eos)

        # 编码 & 截断
        input_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=self.max_length - reserve_len,  # 预留 BOS 和 EOS 空间
            truncation=True
        )
        if self.add_bos: # 添加 BOS
            input_ids = [self.bos_token_id] + input_ids
        if self.add_eos: # 添加 EOS
            input_ids = input_ids + [self.eos_token_id]

        # 生成输入、标签、loss_mask
        # 1. 先拆分序列（避免填充污染）
        input_ids_x = input_ids[:-1]  # X: [BOS, T1, ..., Tn-1]
        input_ids_y = input_ids[1:]   # Y: [T1, ..., Tn-1, EOS]
        
        # 2. 分别处理X和Y序列
        # X序列：模型输入上下文
        input_tensor, attention_mask, _ = self._pad_and_mask(input_ids_x)
        
        # Y序列：模型预测目标
        label_tensor, _, loss_mask = self._pad_and_mask(input_ids_y)

        # 3. 设置忽略标签
        # mask 掩码中，-100 代表不计算损失，-100 以外的值代表计算损失，避免影响loss计算
        label_tensor = label_tensor.clone()
        label_tensor[~loss_mask] = -100 # 忽略 PAD 区损失

        return {
            "input_ids"     : input_tensor,       # [seq_len-1]
            "labels"        : label_tensor,       # [seq_len-1]
            "attention_mask": attention_mask,     # [seq_len-1], 用于 padding 掩码
            "loss_mask"     : loss_mask,          # [seq_len-1]
        }



class SFTDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, start_tokens, end_tokens, max_length=512):
        super().__init__(tokenizer, max_length)
        self.data_path    = data_path
        self.start_tokens = start_tokens
        self.end_tokens   = end_tokens
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]
        
        # 动态生成特殊标记序列
        self._init_special_tokens()
        logging.info(f"Loaded {len(self)} SFT samples")
        logging.info(f"Assistant start tokens: {self.start_tokens}")
        logging.info(f"Im_end token: {self.end_tokens}")

    def _init_special_tokens(self):
        """
        动态生成特殊标记序列，增强鲁棒性
        将传入的 start/end token (文本或 ids) 转为 token id 列表，供后续 KMP 匹配使用。

        """
        # 生成起始标记
        if isinstance(self.start_tokens, str):
            self.start_token_ids = self.tokenizer.encode(self.start_tokens, add_special_tokens=False)
        else:
            self.start_token_ids = list(self.start_tokens)
        
        # 生成消息结束标记
        if isinstance(self.end_tokens, str):
            self.end_token_ids = self.tokenizer.encode(self.end_tokens, add_special_tokens=False)
        else:
            self.end_token_ids = list(self.end_tokens)

        # 验证特殊标记
        if not self.start_tokens:
            logging.warning("Assistant start tokens are empty!")
        if not self.end_tokens:
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

    def _generate_loss_mask(self, input_ids: List[int]) -> List[bool]:
        """
        针对 SFT 模型，精确标记 **assistant 回复正文** 所在 token 位置：
        仅这些位置参与 loss 计算。
        """
        # 1) 先找到 assistant 起始标记 & 结束标记的所有出现位置
        start_positions = self._find_token_sequence(
            self.start_token_ids, input_ids
        )
        end_positions = self._find_token_sequence(
            self.end_token_ids, input_ids
        )

        # 2) 初始化全 False 掩码（不算 loss）
        mask: list[bool] = [False] * len(input_ids)

        # 3) 为每个起始点配对最近的结束点，并设置区间 True
        for start in start_positions:
            content_start = start + len(self.start_token_ids)

            # 找到位于内容后的第一个 `<|im_end|>`
            content_end = next((e for e in end_positions if e > content_start), len(input_ids))

            # 有效性检查：防止空区间或越界
            if content_start < content_end:
                mask[content_start:content_end] = [True] * (content_end - content_start)

        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        返回统一格式的字典：
        {
            "input_ids"     :   [seq_len-1],
            "labels"        :   [seq_len-1],
            "attention_mask":   [seq_len-1], (bool)
            "loss_mask"     :   [seq_len-1]  (bool)
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

        # ---------- 3. SFT 专属掩码 ----------
        sft_mask = self._generate_loss_mask(token_ids)

        # ---------- 4. 填充 + 通用掩码 ----------
        padded_ids, attention_mask, pad_mask = self._pad_and_mask(sft_mask)

        # ---------- 5. 扩展 SFT 掩码到填充长度 ----------
        sft_mask = sft_mask + [False] * (self.max_length - len(sft_mask))
        sft_mask = torch.tensor(sft_mask, dtype=torch.bool)

        # ---------- 6. 合并掩码（AND） ----------
        final_mask = pad_mask & sft_mask

        # ---------- 7. 构造训练对 (左移 1) ----------
        input_ids      = padded_ids[:-1]
        labels         = padded_ids[1:]
        attention_mask = attention_mask[:-1]  # 对齐input_ids长度
        loss_mask      = final_mask[1:]  # 与 labels 对齐

        # labels 中非 loss_mask 位置设置为 -100，避免计算损失
        labels = labels.clone()
        labels[~loss_mask] = -100

        return {
            "input_ids"     :   input_ids,       # [seq_len-1]
            "labels"        :   labels,          # [seq_len-1]
            "attention_mask":   attention_mask,  # [seq_len-1]
            "loss_mask"     :   loss_mask        # [seq_len-1]
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from tempfile import NamedTemporaryFile
    
    # -------------------------
    # 测试 PretrainDataset
    # -------------------------
    # 创建一个临时 json 文件作为测试数据
    sample_data = [
        {"input": "今天天气如何？", "output": "今天天气晴朗，适合出行。"},
        {"input": "你是谁？", "output": "我是一个人工智能助手。"}
    ]
    
    with NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False)
        temp_file_path = f.name
    
    # 使用预训练分词器
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer")  # 或换成你模型对应的分词器
    
    # 确保 tokenizer 有 pad_token 和 bos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|PAD|>"})
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({"bos_token": "<|BOS|>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|EOS|>"})
    
    # 初始化数据集
    dataset = PretrainDataset(
        data_path=temp_file_path,
        tokenizer=tokenizer,
        max_length=32,
        fields=["input", "output"],
        template="问：{input}\n答：{output}",
        add_bos=True,
        add_eos=True
    )
    
    print("\n=== PretrainDataset 样本检查 ===")
    # 取出一个样本进行检查
    sample = dataset[0]
    
    # 打印结果
    print("Input IDs:", sample["input_ids"])
    print("Labels:", sample["labels"])
    print("Attention Mask:", sample["attention_mask"])
    print("Loss Mask:", sample["loss_mask"])
    
    # 打印解码后原句，便于人工核对
    print("Decoded Input:", tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
    print("Decoded Labels:", tokenizer.decode(
        [x if x != -100 else tokenizer.pad_token_id for x in sample["labels"]],
        skip_special_tokens=False
    ))
    
    # 清理临时文件
    os.remove(temp_file_path)

    # -------------------------
    # 测试 SFTDataset
    # -------------------------
    sft_sample_data = [
        {
            "role": "user", "content": "你好，今天天气怎么样？"
        },
        {
            "role": "assistant", "content": "今天天气晴朗，适合出门散步。"
        }
    ]
    # SFTDataset 期望是多轮对话格式，这里用一个简单单轮
    # 假设 token 标记为 <|im_start|>assistant / <|im_end|>
    start_token_text = "<|im_start|>assistant"
    end_token_text = "<|im_end|>"

    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write(json.dumps(sft_sample_data, ensure_ascii=False) + "\n")
        temp_sft_path = f.name

    tokenizer.add_special_tokens({
        "additional_special_tokens": [start_token_text, end_token_text]
    })

    sft_dataset = SFTDataset(
        data_path=temp_sft_path,
        tokenizer=tokenizer,
        start_tokens=start_token_text,
        end_tokens=end_token_text,
        max_length=32
    )

    print("\n=== SFTDataset 样本检查 ===")
    sft_item = sft_dataset[0]
    print("Input IDs:", sft_item["input_ids"])
    print("Labels:", sft_item["labels"])
    print("Attention Mask:", sft_item["attention_mask"])
    print("Loss Mask:", sft_item["loss_mask"])
    print("Decoded Input:", tokenizer.decode(sft_item["input_ids"], skip_special_tokens=False))
    print("Decoded Labels:", tokenizer.decode(
        [x if x != -100 else tokenizer.pad_token_id for x in sft_item["labels"]],
        skip_special_tokens=False
    ))

    os.remove(temp_sft_path)
