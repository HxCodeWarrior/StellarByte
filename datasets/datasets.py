import os
import json
import torch
import logging
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from typing import List, Optional, Union

from torch.utils.data import get_worker_info
import torch.distributed as dist


class PretrainDataset(Dataset):
    """
    预训练数据集加载器
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512
    ):
        """
        :param data_path: 数据文件路径
        :param tokenizer: 分词器
        :param max_length: 最大长度（token级别）
        """
        self.data_path  = data_path
        self.tokenizer  = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token is None:
            logging.warning("[Dataset] Tokenizer has no <pad>, adding '<|pad|>'.")
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.pad_token_id = tokenizer.pad_token_id

        self.data = self._load_data()
        logging.info(f"[Dataset] Loaded {len(self)} samples from {data_path}")

    def _load_data(self) -> List[dict]:
        """
        加载数据
        """
        samples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    samples.append(data)
                except json.JSONDecodeError as e:
                    logging.error(f"[Dataset] JSON decode error in line {line_num}: {e}")
        return samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.data[index]
        text = str(sample.get("text", "")).strip()

        # 处理空文本
        if not text:
            text = "[EMPTY]"
    
        # 编码 & 截断
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,     # 是否自动加 special tokens
            truncation=True,             # 是否截断
            max_length=self.max_length,  # 预留 BOS 和 EOS 空间
            padding='max_length',        # 截断策略：截断长序列
            return_tensors="pt"          # 返回 torch.Tensor
        )

        # 生成 input_ids 和 loss_mask
        input_ids = encoding.input_ids.squeeze() # [max_length]
        loss_mask = (input_ids != self.pad_token_id) # [max_length]

        # 生成输入、标签、loss_mask
        X = input_ids[:-1]          # X: [BOS, T1, ..., Tn-1] 输入序列
        Y = input_ids[1:]           # Y: [T1, ..., Tn-1, EOS] 预测目标
        loss_mask = loss_mask[:-1]  # [T1, ..., Tn-1, EOS]    对齐标签

        return {
            "input_ids" : X.to(torch.long),         # [seq_len-1]
            "labels"    : Y.to(torch.long),         # [seq_len-1]
            "loss_mask" : loss_mask.to(torch.long), # [seq_len-1]
        }



class StreamingPretrainDataset(IterableDataset):
    """
    流式加载预训练数据集
    逐条读取数据，避免一次性加载占用大量内存。
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512
    ):
        IterableDataset.__init__(self)
        
        self.data_path  = data_path
        self.tokenizer  = tokenizer
        self.max_length = max_length

        # pad token 处理
        if tokenizer.pad_token is None:
            logging.warning("[Dataset] Tokenizer has no <pad>, adding '<|pad|>'.")
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.pad_token_id = tokenizer.pad_token_id

    def _iter_jsonl(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logging.error(f"[Dataset] JSON decode error in line {line_num}: {e}")

    def __iter__(self):
        ext = os.path.splitext(self.data_path)[-1]
        if ext != ".jsonl":
            raise ValueError(f"StreamingPretrainDataset only supports .jsonl, but got {ext}")

        data_iter = self._iter_jsonl()

        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id   = worker_info.id if worker_info else 0

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank       = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        parallel = num_workers * world_size
        shard_id = rank * num_workers + worker_id

        for i, sample in enumerate(data_iter):
            if i % parallel != shard_id:
                continue
            yield self._encode_one(sample)
    
    def _encode_one(self, sample):
        # 直接取 text 字段，如果没有就转成字符串
        text = str(sample.get("text", "")).strip()
        if not text:
            text = "[EMPTY]"

        # 编码 & 截断
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,     # 是否自动加 special tokens
            truncation=True,             # 是否截断
            max_length=self.max_length,  # 预留 BOS 和 EOS 空间
            padding='max_length',        # 截断策略：截断长序列
            return_tensors="pt"          # 返回 torch.Tensor
        )

        # 生成 input_ids 和 loss_mask
        input_ids = encoding.input_ids.squeeze() # [max_length]
        loss_mask = (input_ids != self.pad_token_id) # [max_length]

        # 生成输入、标签、loss_mask
        X = input_ids[:-1]          # X: [BOS, T1, ..., Tn-1] 输入序列
        Y = input_ids[1:]           # Y: [T1, ..., Tn-1, EOS] 预测目标
        loss_mask = loss_mask[:-1]  # [T1, ..., Tn-1, EOS]    对齐标签

        return {
            "input_ids" : X.to(torch.long),         # [seq_len-1]
            "labels"    : Y.to(torch.long),         # [seq_len-1]
            "loss_mask" : loss_mask.to(torch.long), # [seq_len-1]
        }



class SFTDataset(Dataset):
    def __init__(
        self, 
        data_path, 
        tokenizer,
        max_length=512
    ):
        super().__init__()

        self.data_path    = data_path
        self.tokenizer    = tokenizer
        self.max_length   = max_length
        
        # 加载数据
        self.data = self._load_data()
        
        # 特殊标记
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

        logging.info(f"Loaded {len(self)} SFT samples")

    def _load_data(self) -> List[dict]:
        """
        加载数据
        """
        samples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    samples.append(data)
                except json.JSONDecodeError as e:
                    logging.error(f"[Dataset] JSON decode error in line {line_num}: {e}")
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids: torch.Tensor) -> List[bool]:
        """
        针对 SFT 模型，精确标记 **assistant 回复正文** 所在 token 位置：
        仅这些位置参与 loss 计算。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        返回统一格式的字典：
        {
            "input_ids"     :   [seq_len-1],
            "labels"        :   [seq_len-1],
            "loss_mask"     :   [seq_len-1]
        }
        """
        sample = self.data[idx]
        
        # ---------- 1. 构造文本（支持 chat template） ----------
        # 若您需要 system/user/assistant 多轮，可在外部预处理
        text = self._create_chat_prompt(sample['conversations'])

        # ---------- 2. tokenizer.encode（不加额外 special token） ----------
        input_ids  = self.tokenizer(text).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # ---------- 3. SFT 专属掩码 ----------
        sft_mask = self._generate_loss_mask(input_ids)

        # ---------- 4. 构造训练对 (左移 1) ----------
        X = torch.tensor(input_ids[:-1], dtype=torch.long)          # [max_length-1]
        Y = torch.tensor(input_ids[1:], dtype=torch.long)           # [max_length-1]
        loss_mask = torch.tensor(sft_mask[1:], dtype=torch.long)    # [max_length-1]

        return {
            "input_ids" : X,        # [seq_len-1]
            "labels"    : Y,        # [seq_len-1]
            "loss_mask" : loss_mask # [seq_len-1]
        }


class StreamingSFTDataset(IterableDataset):
    """
    流式加载的SFT数据集，逐行读取jsonl格式，边读边处理，适合大文件。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
    ):
        IterableDataset.__init__(self)

        self.data_path    = data_path
        self.max_length   = max_length
        self.tokenizer    = tokenizer

        # 加载数据
        self.data = self._load_data()

        # 特殊标记
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

        logging.info(f"[StreamingSFTDataset] Streaming from {data_path}")

    def _load_data(self) -> List[dict]:
        """
        加载数据
        """
        samples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    samples.append(data)
                except json.JSONDecodeError as e:
                    logging.error(f"[Dataset] JSON decode error in line {line_num}: {e}")
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids: torch.Tensor) -> List[bool]:
        """
        针对 SFT 模型，精确标记 **assistant 回复正文** 所在 token 位置：
        仅这些位置参与 loss 计算。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
          
    def _encode_one(self, sample):
        text = self._create_chat_prompt(sample['conversations'])

        input_ids  = self.tokenizer(text).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # SFT 专属掩码
        sft_mask = self._generate_loss_mask(input_ids)

        # 构造训练对 (左移 1)
        X = torch.tensor(input_ids[:-1], dtype=torch.long)          # [max_length-1]
        Y = torch.tensor(input_ids[1:], dtype=torch.long)           # [max_length-1]
        loss_mask = torch.tensor(sft_mask[1:], dtype=torch.long)    # [max_length-1]

        return {
            "input_ids" : X,        # [seq_len-1]
            "labels"    : Y,        # [seq_len-1]
            "loss_mask" : loss_mask # [seq_len-1]
        }
    
    def __iter__(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found: {self.data_path}")

        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id   = worker_info.id if worker_info else 0

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        rank       = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        parallel = num_workers * world_size
        shard_id = rank * num_workers + worker_id

        with open(self.data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % parallel != shard_id:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except Exception as e:
                    logging.warning(f"Failed to parse json line: {e}, skipping line")
                    continue
                yield self._encode_one(sample)
    

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from tempfile import NamedTemporaryFile
    
    # -------------------------
    # 测试 PretrainDataset
    # -------------------------
    # 创建一个临时 jsonl 文件作为测试数据
    sample_data = [
        {"text": "<|im_start|>将以下句子总结为一句话： '科技公司都在积极开发新一代的智能手机，这些新型智能手机应该会在所有功能上都有所突破。'\n科技公司正在积极开发新一代的智能手机，预计这些新型智能手机将在所有功能方面都有所突破。<|im_end|> <|im_start|>请用更简洁的句子表达以下内容：由于天气太冷，我穿了一件大衣，在街上走了大约10分钟，逛了三个商店。\n由于寒冷，我穿大衣在街上走了约10分钟并逛了3个商店。因天气寒冷，我穿大衣，逛三家店，步行约10分钟。<|im_end|> <|im_start|>根据提供的文本，重写成更简单的语言，并保留原始文本的主要意思。\n随着医疗技术的不断发展和应用，越来越多的疾病得到了有效的治疗和控制。由于医疗技术的不断进步和使用，现在有更多的疾病可以得到有效治疗和控制。<|im_end|> <|im_start|>请概括以下文章的主要内容。\n最近的一项研究表明，咖啡可能有助于预防某些慢性疾病，如糖尿病和某些类型的癌症。该文章的主要内容是最新一项研究表明，咖啡可能有助于预防某些慢性疾病，例如糖尿病和某些类型的癌症。<|im_end|>"},
        {"text": "<|im_start|>从以下文本中提取出所有的日期。\n请注意，本公司的财年结束于2021年3月31日，所有数据均按此日期结算。本文中唯一提到的日期是 \"2021年3月31日\"，因为该文段指出该公司的财政年度结束时间为该日期。<|im_end|> <|im_start|>重新诠释以下句子，以使其更加规范。即消除其歧义性和复杂性。\n因为我的父亲工作太忙，我们去年没有去旅行。由于我父亲工作繁忙的原因，去年我们没有去旅行。 \n重新诠释后，这个句子更加规范，消除了歧义和复杂性。<|im_end|> <|im_start|>给出以下句子的摘要：在开放科学中，研究人员应该积极地与其他人交流并分享数据和结果，以帮助推动科学进步。\n这个句子的摘要是：在开放科学中，研究人员需要积极地分享数据和结果，与其他人良好交流来推动科学进步。<|im_end|> <|im_start|>将一个长的句子重写成几个简单的句子。\n原始句子：尽管他很困，他仍然坚持看完了整部电影。这个长句子可以重写为以下几个简单句子：\n- 他很困。\n- 他看电影。\n- 他能够坚持看完整部电影。\n- 他没有放弃。<|im_end|>"}
    ]
    
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        temp_file_path = f.name
    
    # 使用预训练分词器
    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)  # 使用一个可用的分词器
    
    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    # 初始化数据集
    dataset = PretrainDataset(
        data_path=temp_file_path,
        tokenizer=tokenizer,
        max_length=450
    )
    
    print("\n=== PretrainDataset 样本检查 ===")
    # 取出一个样本进行检查
    sample = dataset[0]
    
    # 打印结果
    print("Input IDs:", sample["input_ids"])
    print("Labels:", sample["labels"])
    print("Loss Mask:", sample["loss_mask"])
    
    # 打印解码后原句，便于人工核对
    print("Decoded Input:", tokenizer.decode(sample["input_ids"], skip_special_tokens=False))
    print("Decoded Labels:", tokenizer.decode(sample["labels"], skip_special_tokens=False))
    
    # 清理临时文件
    os.remove(temp_file_path)

    # -------------------------
    # 测试 SFTDataset
    # -------------------------
    sft_sample_data = [
        {
            "conversations": [
                {"role": "user", "content": "你好，今天天气怎么样？"},
                {"role": "assistant", "content": "今天天气晴朗，适合出门散步。"}
            ]
        }
    ]
    
    # 创建临时文件
    with NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for item in sft_sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        temp_sft_path = f.name

    sft_dataset = SFTDataset(
        data_path=temp_sft_path,
        tokenizer=tokenizer,
        max_length=50
    )

    print("\n=== SFTDataset 样本检查 ===")
    sft_item = sft_dataset[0]
    print("Input IDs:", sft_item["input_ids"])
    print("Labels:", sft_item["labels"])
    print("Loss Mask:", sft_item["loss_mask"])
    print("Decoded Input:", tokenizer.decode(sft_item["input_ids"], skip_special_tokens=False))
    print("Decoded Labels:", tokenizer.decode(sft_item["labels"], skip_special_tokens=False))

    os.remove(temp_sft_path)
