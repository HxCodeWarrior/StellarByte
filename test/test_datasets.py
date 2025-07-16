import os
import sys
import json
import tempfile
import pytest
import torch
from pathlib import Path
from transformers import AutoTokenizer

# 将项目根目录添加到sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from datasets import BaseDataset, PretrainDataset, SFTDataset

tokenizer_path = "./tokenizer"
@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(tokenizer_path)  # 使用本地路径

def test_pad_token_injection(tokenizer):
    tokenizer.pad_token = None
    dataset = BaseDataset(tokenizer, max_length=10)
    assert dataset.pad_token_id is not None
    assert tokenizer.pad_token == "<|PAD|>"

def test_pad_and_mask_correctness(tokenizer):
    dataset = BaseDataset(tokenizer, max_length=5)
    input_ids = [1, 2, 3]
    padded, mask = dataset._pad_and_mask(input_ids)
    assert padded == [1, 2, 3, dataset.pad_token_id, dataset.pad_token_id]
    assert mask == [True, True, True, False, False]

@pytest.fixture
def temp_json_file():
    data = [{"input": "你好", "output": "世界"}]
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump(data, f)
        f.flush()
        yield f.name
    os.remove(f.name)

def test_loading_json(temp_json_file, tokenizer):
    dataset = PretrainDataset(data_path=temp_json_file, tokenizer=tokenizer)
    assert len(dataset) == 1
    sample = dataset[0]
    assert "input_ids" in sample
    assert sample["input_ids"].shape[0] == tokenizer.model_max_length - 1 or dataset.max_length - 1

def test_template_formatting(tokenizer):
    test_dir = Path(__file__).parent
    data = [{"input": "测试", "output": "结果"}]
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", dir=test_dir, delete=False, encoding="utf-8"
    ) as tmpfile:
        import json
        json.dump(data, tmpfile)
        tmpfile.flush()
        tmpfile.close()

        dataset = PretrainDataset(data_path=tmpfile.name, tokenizer=tokenizer, template="问：{input} 答：{output}")
        assert len(dataset) == 1

        sample = dataset[0]
        assert "input_ids" in sample
        assert sample["input_ids"].shape[0] == dataset.max_length - 1

    os.remove(tmpfile.name)

def create_sft_sample(path):
    data = [{"role": "assistant", "content": "你好"}]
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

def test_sft_dataset(tokenizer):
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as f:
        create_sft_sample(f.name)
        f.flush()

        dataset = SFTDataset(data_path=f.name, tokenizer=tokenizer)
        assert len(dataset) == 1
        sample = dataset[0]
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert sample["input_ids"].shape == sample["labels"].shape == sample["loss_mask"].shape

    os.remove(f.name)

def test_kmp_find():
    arr = [1, 2, 3, 1, 2, 3, 4]
    pattern = [1, 2, 3]
    positions = SFTDataset._find_token_sequence(pattern, arr)
    assert positions == [0, 3]

if __name__ == "__main__":
    pytest.main(["-v", __file__])
