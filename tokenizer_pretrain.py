import random
import json
import os
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    Regex,
    normalizers
)
from tokenizers.normalizers import (
    NFKC, 
    NFD, 
    StripAccents,
    Replace,
    Lowercase
)
from tokenizers.pre_tokenizers import (
    ByteLevel,
    Digits,
    Punctuation,
    Metaspace,
    Split
)
from typing import Generator, List, Dict, Union, Optional
from emoji import demojize
import unicodedata

random.seed(42)

def read_data_from_jsonl(
    file_path: str, 
    text_fields: Optional[List[str]] = None,
    fallback_field: str = "text"
) -> Generator[str, None, None]:
    """
    读取JSONL文件，支持多种字段结构和回退机制
    
    参数:
        file_path: JSONL文件路径
        text_fields: 尝试提取的文本字段优先级列表
        fallback_field: 当指定字段不存在时的回退字段
    """
    if text_fields is None:
        text_fields = ["input", "content", "reasoning_content", "text"]
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                
                # 尝试按优先级获取文本字段
                text_parts = []
                for field in text_fields:
                    if field in data:
                        content = data[field]
                        if isinstance(content, list):
                            text_parts.extend(content)
                        elif isinstance(content, dict):
                            text_parts.append(json.dumps(content))
                        else:
                            text_parts.append(str(content))
                
                # 回退机制
                if not text_parts:
                    if fallback_field in data:
                        text_parts.append(str(data[fallback_field]))
                    else:
                        # 尝试提取所有字符串值
                        text_parts = [str(v) for v in data.values() if isinstance(v, str)]
                
                # 合并文本
                text = "\n".join(text_parts)
                if not text.strip():
                    raise ValueError(f"Empty text in line {line_num}")
                
                yield text
                
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            except Exception as e:
                print(f"Error in line {line_num}: {e}")
                continue

def create_tokenizer_config(save_dir: str, model_max_length: int = 8192) -> None:
    """
    创建完整的tokenizer配置文件

    参数:
        save_dir: 保存目录
        model_max_length: 最大模型长度
    """

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    config = {
        # tokenizer类，预训练fast tokenizer
        "tokenizer_class": "PreTrainedTokenizerFast",
        # 是否自动在序列开头添加bos_token，建议开启对话模型时使用
        "add_bos_token": True,  
        # 是否自动在序列结尾添加eos_token，建议开启对话模型时使用
        "add_eos_token": True,
        # 是否在单词前加空格，通常GPT类模型需要
        "add_prefix_space": True,
        # 是否清理tokenization产生的多余空格，建议开启
        "clean_up_tokenization_spaces": True,
        # 最大输入长度
        "model_max_length": model_max_length,
        # padding策略，常用right padding，也可设置为'left'
        "padding_side": "right",
        # 截断策略，常用right截断
        "truncation_side": "right",

        # 预定义的特殊token
        "bos_token": "<|sbos|>",
        "eos_token": "<|seos|>",
        "pad_token": "<|spad|>",  # 专用pad token
        "unk_token": "<unk>",
        "mask_token": "<mask>",  # Mask token，方便mask填空任务

        # chat模板，基于Jinja2模板语法生成对话输入
        "chat_template": (
            "{% for message in messages %}"
            "{% set content = message['content'] | trim %}"
            "{% if message['role'] == 'system' %}"
            "<|sbos|>system\n{{ content }}<|seos|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|sbos|>user\n{{ content }}<|seos|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|sbos|>assistant\n{{ content }}<|seos|>\n"
            "{% elif message['role'] == 'tool' %}"
            "<|sbos|>tool\n{{ content }}<|seos|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|sbos|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置文件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # special_tokens_map配置，映射特殊token
    special_tokens_map = {
        "bos_token": "<|sbos|>",
        "eos_token": "<|seos|>",
        "unk_token": "<unk>",
        "pad_token": "<|spad|>",
        "mask_token": "<mask>",
        # 额外特殊token，用于标记对话角色或特殊分隔符，建议与模板保持一致
        "additional_special_tokens": [
            "<s>",
            "</s>",
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
            "<|tool|>"
        ]
    }

    # 保存特殊token映射
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

def create_advanced_normalizer() -> normalizers.Sequence:
    """创建文本规范化序列"""
    return normalizers.Sequence([
        # Unicode规范化
        NFD(),  # 标准化分解，全角半角统一，兼容中英文
        StripAccents(),  # 去除重音符号
        
        # 处理特殊字符
        Replace(Regex(r"\p{Cc}"), ""),  # 移除控制字符
        Replace(Regex(r"\p{Cf}"), ""),  # 移除格式字符
        Replace(Regex(r"[ \t\n\r\f\v]+"), " "),  # 合并空白字符
        
        # 表情符号处理
        Replace(Regex(r"(\p{Emoji})"), r" \1 "),  # 分隔表情符号
        
        # URL和特殊模式
        Replace(Regex(r"https?://\S+"), " [URL] "),      # URL占位
        Replace(Regex(r"\b\d[\d,.]*\b"), " [NUMBER] "),  # 数字占位
        
        # 最终清理
        Replace(Regex(r"\s+"), " "),  # 合并空格
        Replace(Regex(r"^ | $"), ""),  # 去除首尾空格
    ])

def create_advanced_pre_tokenizer() -> pre_tokenizers.Sequence:
    """创建高级预分词器序列"""
    return pre_tokenizers.Sequence([
        # 按空白分割（保留空白信息）
        Metaspace(replacement="▁", add_prefix_space=True),
        
        # 单独隔离代码常用符号
        Split(Regex(r'([\+\-\*/=%&|^~!@#])'), behavior="isolated"),
        Split(Regex(r"(['\"]|//|#|/\*|\*/)"), behavior="isolated"),

        # 分隔字母与大写字母（CamelCase）
        Split(Regex(r'([a-z])([A-Z])'), behavior="contiguous"),
        Split(Regex(r"(\d+)([a-zA-Z])"), behavior="contiguous"),
        
        # 数字处理
        Digits(individual_digits=True),
        
        # 标点符号处理
        Punctuation(behavior="isolated"),
    ])

def train_tokenizer(
    data_paths: Union[str, List[str]],
    save_dir: str,
    text_fields: List[str] = None,
    vocab_size: int = 32768,
    min_frequency: int = 2,
    model_max_length: int = 8192,
    sampling_rate: float = 0.5,
    num_threads: int = 8
) -> None:
    """
    训练并保存自定义tokenizer
    
    参数:
        data_paths: 单个路径或路径列表
        save_dir: 保存目录
        vocab_size: 词汇表大小
        min_frequency: 最小词频
        model_max_length: 最大模型长度
        sampling_rate: 数据采样率
        num_threads: 并行线程数
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE(
        unk_token="<unk>",
        fuse_unk=True,
        byte_fallback=True  # 启用byte fallback
    ))
    
    # 高级文本规范化
    tokenizer.normalizer = create_advanced_normalizer()
    
    # 高级预分词器
    tokenizer.pre_tokenizer = create_advanced_pre_tokenizer()
    
    # 解码器配置
    tokenizer.decoder = decoders.Metaspace(
        replacement="▁", 
        add_prefix_space=True
    )
    
    # 后处理器配置
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[
            ("<s>", 1),
            ("</s>", 2),
            ("<|sbos|>", 3),
            ("<|seos|>", 4),
            ("<|spad|>", 5),
            ("<mask>", 6),
            ("<|system|>", 7),
            ("<|user|>", 8),
            ("<|assistant|>", 9),
            ("<|tool|>", 10),
        ]
    )

    # 特殊token
    special_tokens = [
        "<unk>", 
        "<s>", 
        "</s>", 
        "<|sbos|>", 
        "<|seos|>",
        "<|spad|>",  # 专用pad token
        "<mask>",    # MLM任务支持
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|tool|>"
    ]

    # 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        continuing_subword_prefix="##",  # 子词前缀
        end_of_word_suffix="</w>",       # 词尾标记
        limit_alphabet=1000,              # 限制基础字符集大小
    )

    # 处理多文件路径
    if isinstance(data_paths, str):
        data_paths = [data_paths]
    
    # 并行数据读取
    def process_path(path):
        return list(read_data_from_jsonl(path, text_fields))
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        all_texts = []
        for texts in executor.map(process_path, data_paths):
            # 应用采样
            if sampling_rate < 1.0:
                sample_size = max(1, int(len(texts) * sampling_rate))
                texts = random.sample(texts, sample_size)
            all_texts.extend(texts)
    
    # 训练tokenizer
    print(f"Training tokenizer with {len(all_texts)} samples from {len(data_paths)} files")
    tokenizer.train_from_iterator(
        all_texts, 
        trainer=trainer, 
        length=len(all_texts)
    )

    # 验证特殊token映射
    special_token_map = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<|sbos|>": 3,
        "<|seos|>": 4,
        "<|spad|>": 5,
        "<mask>": 6,
        "<|system|>": 7,
        "<|user|>": 8,
        "<|assistant|>": 9,
        "<|tool|>": 10
    }
    
    for token, expected_id in special_token_map.items():
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            print(f"Warning: Special token '{token}' has ID {actual_id} (expected {expected_id})")

    # 保存tokenizer文件
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    # 创建配置文件
    create_tokenizer_config(save_dir, model_max_length)
    print(f"Tokenizer saved to {save_dir}")
    
    # 计算基本统计信息
    vocab = tokenizer.get_vocab()
    token_lengths = [len(token) for token in vocab.keys() if isinstance(token, str)]
    
    print("\n=== Tokenizer Statistics ===")
    print(f"Vocab size: {len(vocab)}")
    print(f"Average token length: {sum(token_lengths)/len(token_lengths):.2f} chars")
    print(f"Longest token: {max(token_lengths)} chars")
    print(f"Special tokens: {special_tokens}")

def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            spadding_side="right",
            truncation_side="right"
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 测试基本属性
    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")
    print(f"Model max length: {tokenizer.model_max_length}")

    # 测试聊天模板
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "你好吗?"},
        {"role": "assistant", "content": "我很好，谢谢！你呢?"},
        {"role": "user", "content": "我也很好!"},
        {"role": "tool", "content": "执行结果: 42"},
    ]
    
    print("\n=== 聊天模板测试 ===")
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")

    # 测试编码解码
    print("\n=== 编码解码测试 ===")
    encoded = tokenizer(
        prompt, 
        truncation=True, 
        max_length=256,
        return_offsets_mapping=True
    )
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt)
    
    # 测试边界情况
    print("\n=== 边界情况测试 ===")
    test_cases = [
        "",  # 空字符串
        " ",  # 空格
        "   ",  # 多个空格
        "\n\n",  # 换行符
        "Hello\tworld",  # 制表符
        "1234567890",  # 数字
        "https://example.com/path?query=param",  # URL
        "emoji: 😊👍🔥",  # 表情符号
        "CamelCase and snake_case",  # 命名规范
        "日本語のテキスト",  # 日语
        "العربية النص",  # 阿拉伯语
        "Mixing 语言 and languages",  # 混合语言
        "<|sbos|>special<|seos|>",  # 特殊token
        "Newline\ntest",  # 换行
        "Tab\ttest",  # 制表符
        "   Leading and trailing spaces   ",  # 首尾空格
    ]
    
    for text in test_cases:
        encoded = tokenizer(text)
        decoded = tokenizer.decode(encoded["input_ids"])
        print(f"Original: {repr(text)}")
        print(f"Decoded:  {repr(decoded)}")
        print(f"Match:    {text == decoded}\n")

    # 测试压缩率
    print("\n=== 压缩率测试 ===")
    sample_text = """
    Transformer模型是一种基于自注意力机制的深度学习架构，由Vaswani等人在2017年提出。
    它彻底改变了自然语言处理领域，成为BERT、GPT等现代大型语言模型的基础。
    """
    char_count = len(sample_text)
    encoded = tokenizer(sample_text)
    token_count = len(encoded["input_ids"])
    compression_ratio = char_count / token_count
    print(f"Characters: {char_count}, Tokens: {token_count}, Ratio: {compression_ratio:.2f}")

    # 测试多语言支持
    print("\n=== 多语言支持测试 ===")
    languages = [
        ("English", "The quick brown fox jumps over the lazy dog"),
        ("Chinese", "敏捷的棕色狐狸跳过了懒惰的狗"),
        ("Japanese", "速い茶色のキツネがのろまな犬を飛び越えます"),
        ("Korean", "빠른 갈색 여우가 게으른 개를 뛰어넘습니다"),
        ("Arabic", "الثعلب البني السريع يقفز فوق الكلب الكسول"),
        ("Russian", "Быстрая коричневая лиса перепрыгивает через ленивую собаку")
    ]
    
    for lang, text in languages:
        tokens = tokenizer.tokenize(text)
        print(f"{lang}: {tokens}")

    # 测试特殊token处理
    print("\n=== 特殊token处理 ===")
    test_text = "<|sbos|>user\nHello<|seos|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print(f"Special tokens preserved: {decoded == test_text}")
    
    # 测试未知字符处理
    print("\n=== 未知字符处理 ===")
    test_text = "Character: ⨀⨁⨂ (math symbols)"
    tokens = tokenizer.tokenize(test_text)
    print(f"Tokens: {tokens}")

def main():
    # 配置路径 - 支持多文件路径
    data_paths = [
        "path/to/dataset1.jsonl",
        "path/to/dataset2.jsonl",
        "path/to/plain_text.jsonl"
    ]
    save_dir = "stellarbytetokenizer"

    # 训练tokenizer
    train_tokenizer(
        data_paths=data_paths,
        save_dir=save_dir,
        text_fields=None,
        vocab_size=65536,        # 词汇表大小[2^n] [2^15（32768）或 2^16（65536）]
        min_frequency=2,         # 最低词频
        model_max_length=8192,   # 最大模型长度
        sampling_rate=1.0,       # 数据采样占比
        num_threads=12           # 并行处理线程数
    )

    # 评估tokenizer
    eval_tokenizer(save_dir)

if __name__ == '__main__':
    main()