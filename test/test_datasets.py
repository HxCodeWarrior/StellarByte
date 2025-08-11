import os
import json
import tempfile
import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer

import sys
from pathlib import Path

# 将项目根目录添加到sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from datasets import BaseDataset, PretrainDataset, SFTDataset

# ==== Fixture：加载自训练 tokenizer ====
@pytest.fixture(scope="session")
def tokenizer():
    tokenizer_path = "./tokenizer"  # 这里改成你的实际路径
    tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    # 确保有 pad_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return tok


# ==== 测试 BaseDataset ====
def test_base_dataset_pad_and_mask(tokenizer):
    ds = BaseDataset(tokenizer, max_length=5)
    ids = [1, 2, 3]
    padded_ids, attn_mask, loss_mask = ds._pad_and_mask(ids)
    assert padded_ids.tolist() == [1, 2, 3, ds.pad_token_id, ds.pad_token_id]
    assert attn_mask.tolist() == [1, 1, 1, 0, 0]
    assert loss_mask.tolist() == [1, 1, 1, 0, 0]
    assert padded_ids.dtype == torch.long
    assert attn_mask.dtype == torch.bool
    assert loss_mask.dtype == torch.bool


# ==== 辅助函数：创建临时数据文件 ====
def create_temp_file(ext, data):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, mode="w", encoding="utf-8")
    if ext == ".json":
        json.dump(data, tmp, ensure_ascii=False)
    elif ext == ".jsonl":
        for row in data:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif ext == ".csv":
        pd.DataFrame(data).to_csv(tmp, index=False)
    tmp.close()
    return tmp.name


@pytest.mark.parametrize("ext", [".json", ".jsonl", ".csv"])
def test_pretrain_dataset_loading_and_format(tokenizer, ext):
    data = [{"input": "Hello", "output": "World"}]
    path = create_temp_file(ext, data)

    ds = PretrainDataset(data_path=path, tokenizer=tokenizer, max_length=8, fields=["input", "output"])
    assert len(ds) == 1

    sample = ds[0]
    assert set(sample.keys()) == {"input_ids", "labels", "attention_mask", "loss_mask"}
    assert sample["input_ids"].shape[0] == ds.max_length
    assert torch.all(sample["labels"][sample["labels"] != -100] >= 0)

    os.remove(path)


def test_pretrain_dataset_template(tokenizer):
    data = [{"input": "A", "output": "B"}]
    path = create_temp_file(".json", data)
    template = "Q: {input} A: {output}"
    ds = PretrainDataset(path, tokenizer, max_length=8, fields=["input", "output"], template=template)
    text = ds._format_sample(data[0])
    assert text == "Q: A A: B"
    os.remove(path)


def test_pretrain_dataset_missing_field_in_template(tokenizer):
    data = [{"input": "A"}]  # 缺失 output
    path = create_temp_file(".json", data)
    template = "Q: {input} A: {output}"
    ds = PretrainDataset(path, tokenizer, max_length=8, fields=["input", "output"], template=template)
    text = ds._format_sample(data[0])
    assert text == "Q: A A:"  # 缺失字段应为空字符串
    os.remove(path)


# ==== 测试 SFTDataset ====
def test_sftdataset_find_token_sequence():
    arr = [1, 2, 3, 1, 2, 3]
    pattern = [1, 2]
    idxs = SFTDataset._find_token_sequence(pattern, arr)
    assert idxs == [0, 3]


def test_sftdataset_loss_mask(tokenizer):
    # 构造 SFT 数据
    sft_data = [{
        "role": "user", "content": "Hi"
    }, {
        "role": "assistant", "content": "Hello!"
    }]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w", encoding="utf-8")
    tmp.write(json.dumps(sft_data) + "\n")
    tmp.close()

    ds = SFTDataset(tmp.name, tokenizer, start_tokens="<|im_start|>assistant\n", end_tokens="<|im_end|>", max_length=16)
    assert len(ds) == 1

    sample = ds[0]
    assert sample["input_ids"].shape[0] == ds.max_length - 1  # 因为右移
    assert sample["labels"].shape == sample["input_ids"].shape
    assert torch.all((sample["labels"] == -100) | (sample["labels"] >= 0))

    os.remove(tmp.name)

if __name__ == '__main__':
    pytest.main(['-v', __file__])
