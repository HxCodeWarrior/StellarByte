import time
import random
import gc
import psutil
import json
import os
import json
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Regex as TokenizersRegex
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
    NFC,
    NFKC, 
    NFD, 
    StripAccents,
    Replace,
    Lowercase,
    Strip
)
from tokenizers.pre_tokenizers import (
    Sequence,
    Split,
    ByteLevel,
    Digits,
    Punctuation,
    UnicodeScripts
)
from typing import (
    Generator, 
    List, 
    Dict, 
    Union, 
    Optional, 
    Iterator,
    Any
)
from trainer.sqlmanager import SQLiteDatabaseManager

random.seed(42)

class MemoryAwareIterator:
    """内存感知迭代器，动态控制数据处理速度"""
    def __init__(self, iterator, max_memory_usage=0.8, check_interval=1000):
        self.iterator = iterator
        self.max_memory_usage = max_memory_usage
        self.check_interval = check_interval
        self.count = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        # 定期检查内存使用情况
        if self.count % self.check_interval == 0:
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > self.max_memory_usage:
                # 内存使用过高，暂停一下让GC工作
                gc.collect()
                # 动态调整处理速度
                time.sleep(0.1 * (memory_percent - self.max_memory_usage) * 10)
        
        self.count += 1
        return next(self.iterator)

def read_data_from_json(
    file_path: str, 
    text_fields: Optional[List[str]] = None,
    sampling_rate: float = 1.0,
    max_samples: Optional[int] = None
) -> Generator[str, None, None]:
    """
    读取JSONL文件，仅支持多字段优先级提取（无回退机制）
    
    参数:
        file_path: JSONL文件路径
        text_fields: 尝试提取的文本字段优先级列表
        sampling_rate: 采样率
        max_samples: 最大样本数限制
    """
    if text_fields is None:
        text_fields = ["input", "content", "reasoning_content", "text"]
    
    sample_count = 0
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line_num, line in enumerate(f, 1):
            if max_samples and sample_count >= max_samples:
                break
                
            if random.random() > sampling_rate:
                continue
                
            try:
                data = json.loads(line)
                
                # 仅提取指定字段
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
                
                # 如果没有提取到任何文本，直接跳过
                if not text_parts:
                    continue
                
                text = "\n".join(text_parts)
                if not text.strip():
                    continue
                
                sample_count += 1
                yield text
                
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            except Exception as e:
                print(f"Error in line {line_num}: {e}")
                continue

def read_data_from_sqlite(
    db_manager: SQLiteDatabaseManager,
    table_name: str = "tokenizer_data",
    text_field: str = "text",
    batch_size: int = 1000,
    sampling_rate: float = 1.0,
    max_samples: Optional[int] = None,
) -> Generator[str, None, None]:
    """
    从SQLite数据库流式读取数据，支持采样和限制
    ⚠️ 注意：必须传入已连接的 db_manager，避免多次重复连接

    参数:
        db_manager: 已连接的数据库管理器实例
        table_name: 表名
        text_field: 文本字段名
        batch_size: 每次读取的批量大小
        sampling_rate: 采样率
        max_samples: 最大样本数限制

    返回:
        文本数据生成器
    """
    sample_count = 0
    for text in db_manager.stream_text_data(table_name, text_field, batch_size, show_progress=True):
        # 检查最大样本限制
        if max_samples and sample_count >= max_samples:
            break

        # 应用采样
        if random.random() > sampling_rate:
            continue

        yield text
        sample_count += 1

def batch_iterator_json(
    file_paths: List[str], 
    text_fields: Optional[List[str]], 
    batch_size: int = 1000,
    sampling_rate: float = 1.0,
    max_samples: Optional[int] = None
) -> Iterator[List[str]]:
    """
    批量迭代器，逐批读取数据以减少内存使用
    
    参数:
        file_paths: 文件路径列表
        text_fields: 文本字段
        batch_size: 批量大小
        sampling_rate: 采样率
        max_samples: 最大样本数限制
    """
    batch = []
    total_samples = 0
    
    for file_path in file_paths:
        for text in read_data_from_json(
            file_path, text_fields, 
            sampling_rate=sampling_rate, 
            max_samples=max_samples
        ):
            batch.append(text)
            total_samples += 1
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
            
            # 检查最大样本限制
            if max_samples and total_samples >= max_samples:
                if batch:
                    yield batch
                return
    
    if batch:
        yield batch

def batch_iterator_sqlite(
    db_manager: SQLiteDatabaseManager,
    table_name: str = "tokenizer_data",
    text_field: str = "text",
    batch_size: int = 1000,
    sampling_rate: float = 1.0,
    max_samples: Optional[int] = None,
) -> Iterator[List[str]]:
    """
    批量迭代器
    ⚠️ 注意：必须传入已连接的 db_manager

    参数:
        db_manager: 已连接的数据库管理器实例
        table_name: 表名
        text_field: 文本字段名
        batch_size: 批量大小
        sampling_rate: 采样率
        max_samples: 最大样本数限制

    返回:
        批量文本数据迭代器
    """
    batch = []
    total_samples = 0

    for text in read_data_from_sqlite(
        db_manager,
        table_name,
        text_field,
        batch_size,
        sampling_rate,
        max_samples
    ):
        batch.append(text)
        total_samples += 1

        if len(batch) >= batch_size:
            yield batch
            batch = []

        # 检查最大样本限制
        if max_samples and total_samples >= max_samples:
            if batch:
                yield batch
            return

    if batch:
        yield batch

def count_samples_json(
    file_paths: List[str], 
    text_fields: Optional[List[str]], 
    sampling_rate: float = 1.0,
    max_samples: Optional[int] = None
) -> int:
    """
    计算满足条件的样本数量
    
    参数:
        file_paths: 文件路径列表
        text_fields: 文本字段
        sampling_rate: 采样率
        max_samples: 最大样本数限制
    
    返回:
        样本数量
    """
    count = 0
    for file_path in file_paths:
        for text in read_data_from_json(
            file_path, text_fields, 
            sampling_rate=sampling_rate, 
            max_samples=max_samples
        ):
            count += 1
            if max_samples and count >= max_samples:
                return count
    return count

def count_samples_sqlite(
    db_manager: SQLiteDatabaseManager,
    table_name: str = "tokenizer_data",
    text_field: str = "text",
    sampling_rate: float = 1.0,
    max_samples: Optional[int] = None,
) -> int:
    """
    计算满足条件的样本数量
    ⚠️ 注意：必须传入已连接的 db_manager

    参数:
        db_manager: 已连接的数据库管理器实例
        table_name: 表名
        text_field: 文本字段名
        sampling_rate: 采样率
        max_samples: 最大样本数限制

    返回:
        样本数量
    """
    count = 0

    for text in read_data_from_sqlite(
        db_manager,
        table_name,
        text_field,
        1000,
        sampling_rate,
        max_samples
    ):
        count += 1
        if max_samples and count >= max_samples:
            return count

    return count

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
        "add_bos_token": False,  
        # 是否自动在序列结尾添加eos_token，建议开启对话模型时使用
        "add_eos_token": False,
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
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|pad|>",  # 专用pad token
        "unk_token": "<unk>",
        "mask_token": "<mask>",  # Mask token，方便mask填空任务

        # 通用chat模板，基于Jinja2模板语法生成对话输入
        "chat_template": (
            "{% for message in messages %}"
            "{% set role = message['role'] %}"
            "{% set content = message['content'] | trim %}"
            "{% if role == 'system' %}"
            "<|im_start|>system\n{{ content }}<|im_end|>"
            "{% elif role == 'user' %}"
            "<|im_start|>user\n{{ content }}<|im_end|>"
            "{% elif role == 'assistant' %}"
            "<|im_start|>assistant\n{{ content }}<|im_end|>"
            "{% elif role == 'tool' %}"
            "<|im_start|>tool\n{{ content }}<|im_end|>"
            "{% elif role == 'function' %}"
            "<|im_start|>function\n{{ content }}<|im_end|>"
            "{% else %}"
            "<|im_start|>unknown\n{{ content }}<|im_end|>"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|im_start|>assistant\n"
            "{% endif %}"
        )
    }

    # 保存主配置文件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # special_tokens_map配置，映射特殊token
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|pad|>",
        "unk_token": "<unk>",
        "mask_token": "<mask>",
        # 额外特殊token，用于标记对话角色或特殊分隔符，建议与模板保持一致
        "additional_special_tokens": [
            "<s>",             # 消息开始
            "</s>",            # 消息结束
            "<|system|>",      # 系统提示
            "<|user|>",        # 用户
            "<|assistant|>",   # 模型
            "<|tool|>",        # 工具
            "<|function|>",    # 函数调用
            "<|observation|>", # 工具/函数输出
            "<|unknown|>",     # 未知token
            "<|safe|>",        # 安全token
            "<|unsafe|>",      # 非安全token
        ]
    }

    # 保存特殊token映射
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

def create_tokenizer_normalizer() -> normalizers.Sequence:
    """创建文本规范化序列"""
    normalizer_list = []

    # 1. Unicode标准化：NFKC 适合自然语言，但会破坏部分数学/代码符号
    # 👉 推荐策略：代码任务用 NFC，自然语言任务用 NFKC
    #    可以在多语料场景用 NFC，保持更保守
    normalizer_list.append(NFC())

    # 2. 规范换行：CRLF/CR -> LF
    normalizer_list.append(Replace(Regex(r"\r\n?"), "\n"))

    # 4. 保留关键 Cf；清理其余“高风险”控制符
    #   - 保留: ZWJ (U+200D), ZWNJ (U+200C), VS-16/VS-15 (U+FE0F/U+FE0E)
    #   - 移除: C0 控制字符(除 \t \n), 大部分 Bidi/隔离控制等
    normalizer_list.append(Replace(Regex(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]"), ""))
    normalizer_list.append(Replace(Regex(r"[\u2066-\u2069]"), ""))     # LRI/RLI/FSI/PDI：去掉以减少噪声
    # 不移除 \u200C \u200D \uFE0E \uFE0F

    # 5. 空格标准化：全角空格、窄不换行空格等 -> 普通空格
    normalizer_list.append(Replace(Regex(r"[\u3000\u00A0\u2007\u202F]"), " "))

    # 6. 合并行内多余空格（自然语言专用）
    #    但不要影响缩进，所以不能替换制表符
    normalizer_list.append(Replace(Regex(r"[ ]{2,}"), " "))

    # 7. 过多空行折叠为最多两个（保留段落语义）
    normalizer_list.append(Replace(Regex(r"\n{3,}"), "\n\n"))

    # 8. 去除首尾空格和制表符，但保留换行符
    normalizer_list.append(Replace(Regex(r"^[ \t]+"), ""))  # 去除开头空格和制表符
    normalizer_list.append(Replace(Regex(r"[ \t]+$"), ""))  # 去除结尾空格和制表符

    return normalizers.Sequence(normalizer_list)

def create_tokenizer_pre_tokenizer() -> pre_tokenizers.Sequence:
    """创建预分词器序列"""
    return Sequence([
        # 1. 保留整段字符串（支持单双引号、三引号）
        Split(
            Regex(r"([rubf]*(['\"]{3}.*?['\"]{3}|['\"].*?['\"]))"),
            behavior="isolated"
        ),

        # 2. 保留注释整体（//, #, /* */）
        Split(
            Regex(r"(#.*|//.*|/\*[\s\S]*?\*/)"),
            behavior="isolated"
        ),

        # 3. URL 检测：保证 URL 整体不被拆开
        Split(
            Regex(r"https?://[^\s]+"),
            behavior="isolated"
        ),

        # 4. 数字：保证连续数字（含小数点、千分位）不被拆散
        Split(
            Regex(r"\d+([._]\d+)*([eE][+-]?\d+)?"),
            behavior="isolated"
        ),

        # 5. 多字符操作符优先切分
        Split(
            Regex(r"(===|!==|==|!=|<=|>=|\+\+|--|&&|\|\||->|=>|::|\?\?|\.?)"),
            behavior="isolated"
        ),

        # 6. 单字符符号/操作符分开
        Punctuation(behavior="isolated"),

        # 7. Emoji / Extended Pictographic 保持完整
        #   - Extended_Pictographic = 大多数 emoji
        #   - 允许和 ZWJ (U+200D) / VS-16 (U+FE0F) 连用形成复合表情
        Split(
            Regex(r"(\p{Extended_Pictographic}(?:\u200D\p{Extended_Pictographic})*)"),
            behavior="isolated"
        ),

        # 8. 标识符（变量名、函数名）整体(使用Unicode属性匹配字母和数字)
        Split(
            Regex(r"[\p{L}_][\p{L}\p{N}_]*"), 
            behavior="isolated"
        ),

        # 9. UnicodeScripts: 按脚本切分（中/英文/阿拉伯文等）
        #   - 中文和日文不加空格也能单独切出
        UnicodeScripts(),

        # 10. ByteLevel：对所有剩余字符做 fallback
        #   - 保证 tokenizer 对任何输入都能编码/解码
        ByteLevel(add_prefix_space=False),
    ])

def train_tokenizer(
    db_path: Union[str, List[str]],
    save_dir: str,
    table_name: str = "tokenizer_data",
    text_fields: str = 'text',
    vocab_size: int = 32768,
    min_frequency: int = 2,
    model_max_length: int = 8192,
    batch_size: int = 10000,
    phase_sampling_rates: List[float] = [0.3, 0.7, 1.0]
) -> None:
    """
    训练并保存自定义tokenizer
    
    参数:
        db_path: 语料数据库路径
        save_dir: 保存目录
        db_path: 数据库路径
        vocab_size: 词汇表大小
        min_frequency: 最小词频
        model_max_length: 最大模型长度
        sampling_rate: 数据采样率
        batch_size: 批量处理大小
        phase_sampling_rates: 三训练阶段的采样率
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE(
        unk_token="<unk>",
        fuse_unk=True,
        byte_fallback=True  # 启用byte fallback
    ))
    
    # 高级文本规范化
    tokenizer.normalizer = create_tokenizer_normalizer()
    
    # 高级预分词器
    tokenizer.pre_tokenizer = create_tokenizer_pre_tokenizer()
    
    # 解码器配置
    tokenizer.decoder = decoders.ByteLevel()
    
    # 后处理器配置
    tokenizer.post_processor = processors.TemplateProcessing(
         # 如果训练的数据已经按照需要的格式进行处理，那么就不需要手动添加 bos_token 和 eos_token, 反之需要
        single="$A",
        pair="$A $B",
        special_tokens=[
            ("<unk>", 0),
            ("<s>", 1),
            ("</s>", 2),
            ("<|im_start|>", 3),
            ("<|im_end|>", 4),
            ("<|pad|>", 5),
            ("<mask>", 6),
            ("<|system|>", 7),
            ("<|user|>", 8),
            ("<|assistant|>", 9),
            ("<|tool|>", 10),
            ("<|function|>", 11),
            ("<|observation|>", 12),
            ("<|unknown|>", 13),
            ("<|safe|>", 14),
            ("<|unsafe|>", 15)
        ]
    )

    # 特殊token
    special_tokens = [
        "<unk>", 
        "<s>", 
        "</s>", 
        "<|im_start|>", 
        "<|im_end|>",
        "<|pad|>",  # 专用pad token
        "<mask>",    # MLM任务支持
        "<|system|>",  # 系统提示
        "<|user|>",  # 用户
        "<|assistant|>",  # 模型
        "<|tool|>",  # 工具
        "<|function|>",  # 函数调用
        "<|observation|>",  # 工具/函数输出
        "<|unknown|>",  # 未知token
        "<|safe|>",  # 安全token
        "<|unsafe|>",  # 非安全token
    ]
    
    # 计算每个阶段的参数，基于最后一阶段的完全抽样模式
    final_phase_sampling = phase_sampling_rates[-1]  # 最后一阶段的采样率

    # 定义训练阶段参数
    training_phases = []
    for i, phase_sampling in enumerate(phase_sampling_rates):
        # 计算当前阶段相对于最后一阶段的比例
        sampling_ratio = phase_sampling / final_phase_sampling
        
        # 根据抽样比例调整参数
        phase_vocab_size = max(
            int(vocab_size * (0.2 + 0.8 * (i / (len(phase_sampling_rates) - 1)))),  # 从20%到100%线性增长
            min(8000, vocab_size) if i == 0 else min(20000, vocab_size) if i == 1 else vocab_size
        )
        
        phase_min_frequency = min_frequency * (len(phase_sampling_rates) - i)
        
        phase_limit_alphabet = min(
            1000,
            max(800, int(800 + 200 * (i / (len(phase_sampling_rates) - 1))))  # 从800到1000线性增长
        )
        
        # phase_batch_size = max(
        #     1000,
        #     min(batch_size, int(batch_size * (0.2 + 0.8 * (i / (len(phase_sampling_rates) - 1)))))  # 从20%到100%线性增长
        # )
        phase_batch_size = batch_size
        
        training_phases.append({
            "name": f"Phase {i+1}",
            "vocab_size": phase_vocab_size,
            "min_frequency": phase_min_frequency,
            "sampling_rate": phase_sampling,
            "batch_size": phase_batch_size,
            "limit_alphabet": phase_limit_alphabet,
            "sampling_ratio": sampling_ratio
        })
    
    # 创建数据库管理器实例，将在所有阶段中重用
    db_manager = SQLiteDatabaseManager(db_path)
    db_manager.connect()

    # 分阶段训练
    try:
        for i, phase in enumerate(training_phases):
            print(f"\n=== {phase['name']}/{len(training_phases)}: Sampling Rate {phase['sampling_rate']} ===")
            print(f"Vocab Size: {phase['vocab_size']}, Min Frequency: {phase['min_frequency']}")
            print(f"Batch Size: {phase['batch_size']}, Limit Alphabet: {phase['limit_alphabet']}")

            # 配置当前阶段的训练器
            trainer = trainers.BpeTrainer(
                vocab_size=phase["vocab_size"],
                special_tokens=special_tokens,
                min_frequency=phase["min_frequency"],
                show_progress=True,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                continuing_subword_prefix="",
                end_of_word_suffix="",
                limit_alphabet=phase["limit_alphabet"]
            )

            # 创建当前阶段的数据迭代器
            def phase_data_iterator():
                for batch in batch_iterator_sqlite(
                    db_manager,
                    table_name=table_name,
                    text_field=text_fields, 
                    batch_size=phase["batch_size"], 
                    sampling_rate=phase["sampling_rate"]
                ):
                    yield batch
                    gc.collect()

            # 使用内存感知迭代器包装
            memory_aware_iterator = MemoryAwareIterator(phase_data_iterator())

            # 自动计算迭代器长度
            length = count_samples_sqlite(
                db_manager,
                table_name=table_name, 
                text_field=text_fields, 
                sampling_rate=phase["sampling_rate"]
            )
            print(f"Found {length} samples for this phase")

            # 训练当前阶段
            tokenizer.train_from_iterator(
                memory_aware_iterator, 
                trainer=trainer, 
                length=length
            )

            # （可选）在每个阶段后保存 tokenizer 状态以备后续使用或检查
            tokenizer.save(f"./stellarbytetokenizer/phase_pretrain/tokenizer_phase_{i+1}.json")

            # 打印当前阶段的统计信息
            current_vocab_size = len(tokenizer.get_vocab())
            print(f"{phase['name']} completed. Current vocab size: {current_vocab_size}")
    finally:
        # 关闭数据库连接
        db_manager.disconnect()

    # 验证特殊token映射
    special_token_map = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<|im_start|>": 3,
        "<|im_end|>": 4,
        "<|pad|>": 5,
        "<mask>": 6,
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

def eval_tokenizer(tokenizer_path: str, test_samples: int = 1000) -> Dict[str, Any]:
    """
    全面评估tokenizer功能，包括性能、质量、多语言支持和特殊场景处理
    
    参数:
        tokenizer_path: tokenizer保存路径
        test_samples: 用于性能测试的样本数量
    
    返回:
        包含所有评估指标的字典
    """
    results = {
        "basic_info": {},
        "performance_metrics": {},
        "quality_metrics": {},
        "special_cases": {},
        "multilingual_support": {},
        "error_handling": {}
    }
    
    # 加载tokenizer
    try:
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            padding_side="right",
            truncation_side="right"
        )
        load_time = time.time() - start_time
        results["basic_info"]["load_time_seconds"] = load_time
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return results

    # 1. 基本信息评估
    print("\n=== Tokenizer基本信息 ===")
    vocab_size = len(tokenizer)
    results["basic_info"]["vocab_size"] = vocab_size
    results["basic_info"]["special_tokens"] = tokenizer.all_special_tokens
    results["basic_info"]["special_token_ids"] = tokenizer.all_special_ids
    results["basic_info"]["model_max_length"] = tokenizer.model_max_length
    
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")
    print(f"Model max length: {tokenizer.model_max_length}")
    print(f"Load time: {load_time:.3f} seconds")

    # 2. 性能指标评估
    print("\n=== 性能指标评估 ===")
    performance_metrics = evaluate_tokenizer_performance(tokenizer, test_samples)
    results["performance_metrics"] = performance_metrics
    
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value}")

    # 3. 质量指标评估
    print("\n=== 质量指标评估 ===")
    quality_metrics = evaluate_tokenizer_quality(tokenizer)
    results["quality_metrics"] = quality_metrics
    
    for metric, value in quality_metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # 4. 特殊场景测试
    print("\n=== 特殊场景测试 ===")
    special_cases = evaluate_special_cases(tokenizer)
    results["special_cases"] = special_cases
    
    for case, result in special_cases.items():
        print(f"{case}: {result}")

    # 5. 多语言支持评估
    print("\n=== 多语言支持评估 ===")
    multilingual_results = evaluate_multilingual_support(tokenizer)
    results["multilingual_support"] = multilingual_results
    
    for lang, metrics in multilingual_results.items():
        print(f"{lang}: {metrics}")

    # 6. 错误处理评估
    print("\n=== 错误处理评估 ===")
    error_handling = evaluate_error_handling(tokenizer)
    results["error_handling"] = error_handling
    
    for test, result in error_handling.items():
        print(f"{test}: {result}")

    # 7. 聊天模板测试
    print("\n=== 聊天模板测试 ===")
    chat_template_results = evaluate_chat_template(tokenizer)
    results["chat_template"] = chat_template_results
    
    for test, result in chat_template_results.items():
        if test == "generated_prompt":
            print(f"Generated prompt:\n{result}")
        else:
            print(f"{test}: {result}")

    # 8. 词汇表分析
    print("\n=== 词汇表分析 ===")
    vocab_analysis = analyze_vocabulary(tokenizer)
    results["vocab_analysis"] = vocab_analysis
    
    for metric, value in vocab_analysis.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    # 9. 与常用tokenizer对比
    print("\n=== 与常用tokenizer对比 ===")
    comparison_results = compare_with_standard_tokenizers(tokenizer)
    results["comparison"] = comparison_results
    
    for tokenizer_name, metrics in comparison_results.items():
        print(f"{tokenizer_name}: {metrics}")

    return results

def evaluate_tokenizer_performance(tokenizer, test_samples: int = 1000) -> Dict[str, Any]:
    """评估tokenizer性能指标"""
    results = {}
    
    # 准备测试数据
    test_texts = [
        "这是一段中文文本，用于测试tokenizer的性能。",
        "This is an English text for testing tokenizer performance.",
        "1234567890 特殊字符 !@#$%^&*()",
        "https://example.com/path?query=param&value=test",
        "Emoji测试: 😊👍🔥🎉",
        "代码示例: def hello_world(): print('Hello, World!')",
        "长文本测试: " + "自然语言处理 " * 50
    ]
    
    # 编码速度测试
    start_time = time.time()
    for i in range(test_samples):
        text = random.choice(test_texts)
        tokenizer(text, truncation=True, max_length=512)
    encode_time = time.time() - start_time
    results["encode_speed_tokens_per_second"] = test_samples * 100 / encode_time  # 估算
    
    # 解码速度测试
    encoded_texts = [tokenizer(text, truncation=True, max_length=512) for text in test_texts]
    start_time = time.time()
    for i in range(test_samples):
        encoded = random.choice(encoded_texts)
        tokenizer.decode(encoded["input_ids"])
    decode_time = time.time() - start_time
    results["decode_speed_tokens_per_second"] = test_samples * 100 / decode_time  # 估算
    
    # 内存使用测试
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # 执行一些操作后再次测量
    for i in range(100):
        text = random.choice(test_texts)
        tokenizer(text, truncation=True, max_length=512)
    
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    results["memory_usage_mb"] = memory_after - memory_before
    
    return results

def evaluate_tokenizer_quality(tokenizer) -> Dict[str, Any]:
    """评估tokenizer质量指标"""
    results = {}
    
    # 测试文本
    test_cases = [
        "The quick brown fox jumps over the lazy dog.",
        "敏捷的棕色狐狸跳过了懒惰的狗。",
        "1234567890",
        "Hello, world! 你好，世界！",
        "https://example.com/path?query=test",
        "Emoji: 😊👍🔥🎉",
        "Code: def function(): return True"
    ]
    
    # 编码-解码一致性测试
    perfect_matches = 0
    total_cases = len(test_cases)
    
    for text in test_cases:
        encoded = tokenizer(text)
        decoded = tokenizer.decode(encoded["input_ids"])
        if text == decoded:
            perfect_matches += 1
    
    results["perfect_decode_ratio"] = perfect_matches / total_cases
    
    # 压缩率测试
    total_chars = 0
    total_tokens = 0
    
    for text in test_cases:
        encoded = tokenizer(text)
        total_chars += len(text)
        total_tokens += len(encoded["input_ids"])
    
    results["compression_ratio"] = total_chars / total_tokens if total_tokens > 0 else 0
    results["average_tokens_per_sample"] = total_tokens / len(test_cases)
    
    # 特殊token保留测试
    special_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(special_text)
    decoded = tokenizer.decode(encoded["input_ids"])
    results["special_tokens_preserved"] = special_text == decoded
    
    return results

def evaluate_special_cases(tokenizer) -> Dict[str, Any]:
    """评估特殊场景处理能力"""
    results = {}
    
    special_cases = [
        ("空字符串", ""),
        ("空格", " "),
        ("多个空格", "   "),
        ("换行符", "\n\n"),
        ("制表符", "Hello\tworld"),
        ("混合空白符", "  Hello  \t  World  \n"),
        ("仅特殊字符", "!@#$%^&*()"),
        ("数字", "1234567890"),
        ("长数字", "1234567890" * 10),
        ("URL", "https://example.com/path?query=param&value=test"),
        ("电子邮件", "user@example.com"),
        ("文件路径", "/home/user/document.txt"),
        ("JSON", '{"key": "value", "array": [1, 2, 3]}'),
        ("HTML", "<div class='container'><p>Hello</p></div>"),
        ("SQL", "SELECT * FROM users WHERE id = 1"),
        ("代码", "def factorial(n): return 1 if n == 0 else n * factorial(n-1)"),
        ("表情符号", "😊👍🔥🎉❤️😂"),
        ("复合表情", "👨‍👩‍👧‍👦"),  # 家庭表情
        ("零宽连接符", "caf\u00e9"),  # 带重音符号
        ("异体字选择器", "︎"),  # VS-15
        ("异体字选择器", "️"),  # VS-16
        ("从左向右标记", "\u200e"),  # LRM
        ("从右向左标记", "\u200f"),  # RLM
        ("ZWNJ", "निर्भर"),  # 印地语中的ZWNJ
        ("ZWJ", "👨‍💻"),  # 男人和技术员
    ]
    
    for name, text in special_cases:
        try:
            encoded = tokenizer(text)
            decoded = tokenizer.decode(encoded["input_ids"])
            results[name] = text == decoded
        except Exception as e:
            results[name] = f"Error: {str(e)}"
    
    return results

def evaluate_multilingual_support(tokenizer) -> Dict[str, Any]:
    """评估多语言支持能力"""
    results = {}
    
    languages = [
        ("英语", "The quick brown fox jumps over the lazy dog."),
        ("中文", "敏捷的棕色狐狸跳过了懒惰的狗。"),
        ("日语", "速い茶色のキツネがのろまな犬を飛び越えます"),
        ("韩语", "빠른 갈색 여우가 게으른 개를 뛰어넘습니다"),
        ("阿拉伯语", "الثعلب البني السريع يقفز فوق الكلب الكسول"),
        ("俄语", "Быстрая коричневая лиса перепрыгивает через ленивую собаку"),
        ("法语", "Le rapide renard brun saute par-dessus le chien paresseux"),
        ("德语", "Der schnelle braune Fuchs springt über den faulen Hund"),
        ("西班牙语", "El rápido zorro marrón salta sobre el perro perezoso"),
        ("印地语", "तेज भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है"),
        ("葡萄牙语", "A rápida raposa marrom salta sobre o cão preguiçoso"),
        ("孟加拉语", "দ্রুত বাদামী শিয়াল অলস কুকুর উপর jumps"),
        ("意大利语", "La volpe marrone veloce salta sopra il cane pigro"),
        ("土耳其语", "Hızlı kahverengi tilki tembel köpeğin üzerinden atlar"),
    ]
    
    for lang, text in languages:
        try:
            encoded = tokenizer(text)
            tokens = tokenizer.tokenize(text)
            decoded = tokenizer.decode(encoded["input_ids"])
            
            results[lang] = {
                "original_length": len(text),
                "token_count": len(encoded["input_ids"]),
                "compression_ratio": len(text) / len(encoded["input_ids"]) if len(encoded["input_ids"]) > 0 else 0,
                "perfect_decode": text == decoded,
                "unique_tokens": len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0  # 令牌多样性
            }
        except Exception as e:
            results[lang] = f"Error: {str(e)}"
    
    return results

def evaluate_error_handling(tokenizer) -> Dict[str, Any]:
    """评估错误处理能力"""
    results = {}
    
    # 测试各种边界和错误情况
    error_cases = [
        ("空输入", ""),
        ("超大输入", "A" * 100000),  # 超长文本
        ("None输入", None),
        ("数字输入", 123),
        ("列表输入", ["text1", "text2"]),
        ("非法Unicode", b'\xFF\xFE'.decode('utf-8', errors='replace')),  # 非法UTF-8
        ("控制字符", "Hello\u0000World"),  # 空字符
        ("无效UTF-8序列", b'\xff\xfe'.decode('utf-8', errors='ignore')),  # BOM标记
    ]
    
    for name, input_data in error_cases:
        try:
            if input_data is None:
                encoded = tokenizer.encode(input_data)
            else:
                encoded = tokenizer.encode(input_data)
            results[name] = "Success"
        except Exception as e:
            results[name] = f"Error: {type(e).__name__}"
    
    return results

def evaluate_chat_template(tokenizer) -> Dict[str, Any]:
    """评估聊天模板功能"""
    results = {}
    
    # 测试消息
    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "你好，请介绍你自己。"},
        {"role": "assistant", "content": "你好！我是一个AI助手，专门设计来回答你的问题和提供帮助。"},
        {"role": "user", "content": "你能做什么？"}
    ]
    
    try:
        # 生成提示
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        results["generated_prompt"] = prompt
        
        # 测试编码解码
        encoded = tokenizer(
            prompt, 
            truncation=True, 
            max_length=512,
            return_offsets_mapping=True
        )
        decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
        
        results["decode_matches_original"] = prompt == decoded
        results["special_tokens_detected"] = any(
            token_id in encoded["input_ids"] 
            for token_id in tokenizer.all_special_ids
        )
        
    except Exception as e:
        results["error"] = f"Chat template error: {str(e)}"
    
    return results

def analyze_vocabulary(tokenizer) -> Dict[str, Any]:
    """分析词汇表特征"""
    results = {}
    
    vocab = tokenizer.get_vocab()
    tokens = list(vocab.keys())
    
    # 基本统计
    token_lengths = [len(token) for token in tokens if isinstance(token, str)]
    results["vocab_size"] = len(vocab)
    results["average_token_length"] = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    results["max_token_length"] = max(token_lengths) if token_lengths else 0
    results["min_token_length"] = min(token_lengths) if token_lengths else 0
    
    # 特殊token比例
    special_tokens = [token for token in tokens if token in tokenizer.all_special_tokens]
    results["special_token_ratio"] = len(special_tokens) / len(tokens) if tokens else 0
    
    # 字符类型分布
    char_categories = {}
    for token in tokens:
        for char in token:
            category = unicodedata.category(char)
            char_categories[category] = char_categories.get(category, 0) + 1
    
    # 标准化比例
    total_chars = sum(char_categories.values())
    for category in char_categories:
        char_categories[category] = char_categories[category] / total_chars
    
    results["char_category_distribution"] = char_categories
    
    # 常见前缀和后缀
    prefixes = Counter([token[:3] for token in tokens if len(token) >= 3])
    suffixes = Counter([token[-3:] for token in tokens if len(token) >= 3])
    
    results["common_prefixes"] = dict(prefixes.most_common(10))
    results["common_suffixes"] = dict(suffixes.most_common(10))
    
    return results

def compare_with_standard_tokenizers(tokenizer) -> Dict[str, Any]:
    """与标准tokenizer进行对比"""
    results = {}
    
    # 测试文本
    test_text = "The quick brown fox jumps over the lazy dog. 敏捷的棕色狐狸跳过了懒惰的狗。"
    
    # 对比的tokenizer列表（如果可用）
    standard_tokenizers = {}
    
    try:
        from transformers import GPT2Tokenizer
        standard_tokenizers["gpt2"] = GPT2Tokenizer.from_pretrained("gpt2")
    except:
        pass
        
    try:
        from transformers import BertTokenizer
        standard_tokenizers["bert"] = BertTokenizer.from_pretrained("bert-base-uncased")
    except:
        pass
    
    # 对比指标
    for name, std_tokenizer in standard_tokenizers.items():
        # 编码测试文本
        test_encoded = tokenizer(test_text)
        std_encoded = std_tokenizer(test_text)
        
        # 计算对比指标
        results[name] = {
            "test_token_count": len(test_encoded["input_ids"]),
            "std_token_count": len(std_encoded["input_ids"]),
            "ratio": len(test_encoded["input_ids"]) / len(std_encoded["input_ids"]) if len(std_encoded["input_ids"]) > 0 else 0,
            "common_tokens": len(set(test_encoded["input_ids"]) & set(std_encoded["input_ids"])) / len(set(test_encoded["input_ids"])) if len(set(test_encoded["input_ids"])) > 0 else 0
        }
    
    return results

def main():
    # 配置路径 - 支持多文件路径
    db_path = "./datasets/tokenizers/pretrain_tokeniser.db"
    data_path = "./datasets/train/model_pretrain.jsonl"

    save_dir = "./stellarbytetokenizer"
    eval_tokenizer_info_dir = './model_info'

    # 检查数据库是否存在，如果不存在则创建并导入数据
    if not os.path.exists(db_path):
        db_manager = SQLiteDatabaseManager(db_path)
        db_manager.connect()
        db_manager.create_table('tokenizer_pretrain_data', 'text')
        db_manager.insert_data(
            data_file_path=data_path,
            table_name='tokenizer_pretrain_data',
            columns='text'
        )
        db_manager.optimize_database()
        db_manager.disconnect()

    # 训练tokenizer
    train_tokenizer(
        db_path=db_path,
        save_dir=save_dir,
        table_name='tokenizer_pretrain_data',
        text_fields='text',
        vocab_size=52000,        # 词汇表大小[2^n] [2^15（32768）或 2^16（65536）]
        min_frequency=2,         # 最低词频
        model_max_length=1000000000000000019884624838656,   # 最大模型长度
        batch_size=500,          # 批处理大小
        phase_sampling_rates=[0.3, 0.7, 1.0],    # 不同阶段数据采样占比
    )

    # 评估tokenizer
    eval_results = eval_tokenizer(save_dir, test_samples=1000)
    
    # 保存评估结果到文件
    with open(os.path.join(eval_tokenizer_info_dir, "tokenizer_evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=4)
    
    print(f"\n评估结果已保存到: {os.path.join(eval_tokenizer_info_dir, 'evaluation_results.json')}")

if __name__ == '__main__':
    main()