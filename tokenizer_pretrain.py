import time
import random
import json
import os
import re
import concurrent.futures
import gc
from tkinter import N
import zstandard as zstd
from collections import Counter
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    Tokenizer,
    processors,
    Regex
)
from tokenizers.normalizers import NFKC, Sequence, Lowercase, StripAccents, Replace, Strip
from tokenizers.pre_tokenizers import ByteLevel, Split, Digits, Punctuation, Metaspace, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import UnigramTrainer
from typing import Generator, List, Iterator, Tuple, Dict, Any
from multiprocessing import cpu_count
from tqdm import tqdm
import psutil
import torch
import numpy as np
from loguru import logger

# 配置日志
logger.add("/log/tokenizer_train.log", rotation="10 MB", level="INFO")
random.seed(42)
np.random.seed(42)

# !!!tokenizers库是用Rust编写的，目前没有GPU加速支持
# 检查GPU可用性并设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger.info(f"Using device: {device}")
# if device.type == "cuda":
#     torch.backends.cudnn.benchmark = True

# 预编译所有正则表达式
XML_PROTECTION_PATTERNS = [
    (re.compile(r'(<\/?[a-zA-Z_]+>)'), r' \1 '),
    (re.compile(r'\s+(<\/?[a-zA-Z_]+>)\s+'), r'\1')
]

MATH_REPLACEMENTS = [
    (re.compile(r'\b(\d+\.\d+)\b'), r'<float> \1'),
    (re.compile(r'\b(0x[0-9a-fA-F]+)\b'), r'<hex> \1'),
    (re.compile(r'([+\-*/=<>!&|^~%]+)'), r' \1 '),
    (re.compile(r'(\d+)\s*([a-zA-Zα-ω])\b'), r'\1\2'),
    (re.compile(r'(\\[a-zA-Z]+)'), r' \1 ')
]

CODE_REPLACEMENTS = [
    (re.compile(r'(function|def|class|import|from)\s+'), r'<\1> '),
    (re.compile(r'(for|while|if|else|switch|case)\s*\('), r'<\1>('),
    (re.compile(r'(return|break|continue|yield)\b'), r'<\1>'),
    (re.compile(r'(\/\/[^\n]*|\#[^\n]*)'), r'<comment> \1')
]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def memory_usage() -> str:
    """获取当前内存使用情况"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)  # GB
    return f"{mem:.2f} GB"

def count_lines(file_path: str) -> int:
    """快速统计文件行数（支持zstd压缩文件）"""
    if file_path.endswith('.zst'):
        dctx = zstd.ZstdDecompressor()
        with open(file_path, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                return sum(1 for _ in reader)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)

def detect_file_format(file_path: str) -> str:
    """检测文件格式"""
    if file_path.endswith('.jsonl') or file_path.endswith('.jsonl.zst'):
        return 'jsonl'
    elif file_path.endswith('.txt') or file_path.endswith('.txt.zst'):
        return 'txt'
    elif any(file_path.endswith(ext) for ext in ['.py', '.java', '.cpp', '.js', '.ts', '.go']):
        return 'code'
    return 'unknown'

def extract_text_from_jsonl(data: dict) -> str:
    """从JSONL对象中提取并组合文本"""
    text_parts = []
    fields = ['input', 'content', 'reasoning_content', 'text', 'prompt', 'response']
    
    for field in fields:
        if field in data:
            content = data[field]
            if isinstance(content, list):
                text_parts.extend(content)
            elif isinstance(content, dict):
                text_parts.append(json.dumps(content, ensure_ascii=False))
            else:
                text_parts.append(str(content))
    
    # 特殊处理对话格式
    if 'conversations' in data:
        for turn in data['conversations']:
            if 'content' in turn:
                text_parts.append(turn['content'])
    
    return "\n".join(filter(None, text_parts))

def read_file_chunk(file_path: str, start: int, end: int) -> List[str]:
    """读取文件块"""
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        f.seek(start)
        while f.tell() < end:
            line = f.readline()
            if not line: break
            texts.append(line)
    return texts

def preprocess(text: str) -> str:
    """
    文本预处理：
    - 保留数学结构
    - 保护代码块
    - 规范化特殊符号
    """
    # XML标签保护
    for pattern, replacement in XML_PROTECTION_PATTERNS:
        text = pattern.sub(replacement, text)
    
    # 数学表达式处理
    for pattern, replacement in MATH_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    
    # 代码元素处理
    for pattern, replacement in CODE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    
    # 清理多余空格（单次全局替换）
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_tokenizer_config(save_dir: str, vocab_size: int, model_max_length: int = 8192) -> None:
    """创建完整的tokenizer配置文件"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "bos_token": "<|SBOS|>",
        "eos_token": "<|SEOS|>",
        "pad_token": "<|PAD|>",
        "unk_token": "<unk>",
        "model_max_length": model_max_length,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|SBOS|>system\n{{ message['content'] }}<|SEOS|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|SBOS|>user\n{{ message['content'] }}<|SEOS|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|SBOS|>assistant\n{{ message['content'] }}<|SEOS|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|SBOS|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置文件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 创建special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|SBOS|>",
        "eos_token": "<|SEOS|>",
        "unk_token": "<unk>",
        "pad_token": "<|PAD|>",
        "additional_special_tokens": [
            # 核心对话标记
            "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
            
            # 数学运算标记
            "<math>", "<formula>", "<equation>", "<matrix>", "<integral>", 
            "<derivative>", "<summation>", "<limit>", "<vector>", "<tensor>",
            "<probability>", "<fraction>", "<sqrt>", "<log>", "<trig>", "<set>",
            
            # 代码生成标记
            "<code>", "<function>", "<class>", "<loop>", "<condition>", "<variable>",
            "<import>", "<comment>", "<api_call>", "<sql>", "<json>", "<xml>", 
            "<yaml>", "<html>", "<css>", "<js>", "<python>", "<java>", "<cpp>",
            
            # 推理结构标记
            "<reasoning>", "<step>", "<proof>", "<theorem>", "<lemma>",
            "<conjecture>", "<corollary>", "<axiom>", "<definition>",
            
            # 数据操作标记
            "<data>", "<table>", "<row>", "<column>", "<cell>", "<dataset>",
            "<stat>", "<mean>", "<median>", "<std>", "<distribution>",
            
            # 领域特定标记
            "<medical>", "<legal>", "<financial>", "<scientific>",

            # 添加XML结构标记
            "<input>", "</input>",
            "<instruction>", "</instruction>",
            "<reasoning_content>", "</reasoning_content>",
            "<output>", "</output>",
            "<source>", "</source>",
            "<score>", "</score>",
        ]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

# ===== 批处理生成器 =====
def batch_generator(
    file_paths: List[str],
    batch_size: int = 10000,
    max_samples: int = None,
    num_workers: int = None
) -> Iterator[List[str]]:
    """
    高效批处理生成器
    """
    # 设置tokenizers并行
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)
    
    # 使用用线程池避免pickle问题
    # 使用进程池：with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor: 
    # 使用线程池：with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        # 分块处理文件
        for file_path in file_paths:
            file_format = detect_file_format(file_path)
            futures.append(executor.submit(
                lambda f: list(process_single_file(f[0], f[1], batch_size)),  # 转换为list
                (file_path, file_format)
            ))
        
        sample_count = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                for batch in future.result():
                    if not batch:
                        continue
                        
                    # 直接处理避免嵌套并行
                    processed_batch = [preprocess(text) for text in batch]
                    
                    yield processed_batch
                    sample_count += len(processed_batch)
                    
                    if max_samples and sample_count >= max_samples:
                        return
            except Exception as e:
                logger.error(f"文件处理错误: {str(e)}")

def process_single_file(
    file_path: str, 
    file_format: str, 
    batch_size: int
) -> Generator[List[str], None, None]:
    """
    处理单个文件，返回批次生成器
    """
    batch = []
    current_batch = []
    
    lines = open(file_path, 'r', encoding='utf-8', errors='replace')
    
    for line in lines:
        try:
            text = ""
            if file_format == 'jsonl':
                data = json.loads(line)
                text = extract_text_from_jsonl(data)
            else:  # txt 或 code
                text = line.strip()
            
            if text:
                batch.append(text)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        except Exception as e:
            logger.warning(f"Error processing line: {str(e)}")
            continue
    
    if batch:
        yield batch
    
    if not file_path.endswith('.zst'):
        lines.close()

def train_tokenizer(
    data_paths: List[str], 
    save_dir: str, 
    vocab_size: int = 16384,
    model_max_length: int = 8192,
    max_piece_length: int = 10,
    min_frequency: int = 2,
    num_phases: int = 3,
    num_workers: int = 2,
    shriking_factor: float = 1.5,
    resume_phase: int = 1
) -> None:
    """
    训练并保存自定义tokenizer（多阶段训练）
    
    参数：
    data_paths: 训练数据路径列表
    save_dir: 保存目录
    vocab_size: 词汇表大小（推荐16384-32768）
    model_max_length: 模型最大长度
    max_piece_length: 最大子词长度
    min_frequency: 最小词频阈值
    num_phases: 训练阶段数（3阶段效果最佳）
    num_workers: 并行线程数（默认为CPU核心数-2）
    shriking_factor: 缩减因子（默认为1.5）
    resume_phase: 恢复训练阶段（默认为1）
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Starting tokenizer training. Initial memory: {memory_usage()}")
    
    # 初始化tokenizer（BPE算法）(包含恢复训练）
    if resume_phase > 1:
        # 加载之前阶段的tokenizer
        prev_phase_dir = os.path.join(save_dir, f"phase_{resume_phase-1}")
        prev_tokenizer_path = os.path.join(prev_phase_dir, "tokenizer.json")
        
        if os.path.exists(prev_tokenizer_path):
            tokenizer = Tokenizer.from_file(prev_tokenizer_path)
            logger.info(f"Resuming training from phase {resume_phase}. Loaded tokenizer from {prev_tokenizer_path}")
        else:
            logger.error(f"Previous phase tokenizer not found at {prev_tokenizer_path}. Starting from scratch.")
            tokenizer = Tokenizer(models.BPE(
                unk_token="<unk>",
                continuing_subword_prefix='##',
                end_of_word_suffix='</w>',
            ))
    else:
        # 全新初始化
        tokenizer = Tokenizer(models.BPE(
            unk_token="<unk>",
            continuing_subword_prefix='##', # 子词前缀
            end_of_word_suffix='</w>',      # 单词后缀
        ))
    
    if resume_phase == 1:
        # 文本规范化
        tokenizer.normalizer = normalizers.Sequence([
            NFKC(),
            Replace(Regex(r'[ \u00A0\ufeff]+'), " "),  # 替换各种空格
            Replace(Regex(r'\s+'), " "),               # 合并连续空格
            StripAccents(),                            # 去除重音符号
            Replace(Regex(r'[“”]'), '"'),              # 统一引号
            Replace(Regex(r'[‘’]'), "'"),
            Replace(Regex(r'[…]+'), "..."),            # 统一省略号
            Replace(Regex(r'(\d{1,3}\.){3}\d{1,3}'), r'<ip> \1'),  # IP地址
            Replace(Regex(r'(CVE-\d{4}-\d{4,7})'), r'<cve> \1'),  # 保护CVE标识符
            Replace(Regex(r'(CWE-\d+)'), r'<cwe> \1'),  # 保护CWE标识符
            Replace(Regex(r'([A-Z]{3,5}-\d{3,5})'), r'<vuln> \1'),  # 通用漏洞标识
            Lowercase()                                # 小写化（可选）
        ])

        # 预分词器
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(r'<[^>]+>'), behavior='isolated'),  # 隔离特殊标记
            pre_tokenizers.Split(Regex(r'CVE-\d{4}-\d{4,7}'), behavior='isolated'),  # CVE
            pre_tokenizers.Split(Regex(r'CWE-\d+'), behavior='isolated'),  # CWE
            pre_tokenizers.Split(Regex(r'([A-Z]{3,5}-\d{3,5})'), behavior='isolated'),  # 漏洞标识
            pre_tokenizers.Split(Regex(r'(\d{1,3}\.){3}\d{1,3}'), behavior='isolated'),  # IP地址
            pre_tokenizers.Split(Regex(r'0x[0-9a-fA-F]+'), behavior='isolated'),  # 隔离16进制
            pre_tokenizers.Split(Regex(r'\b\d+\.\d+\b'), behavior='isolated'),  # 隔离浮点数
            pre_tokenizers.Split(Regex(r'\\[a-zA-Z]+'), behavior='isolated'),  # LaTeX命令
            pre_tokenizers.Split(Regex(r'<code>(.*?)</code>'), behavior='isolated'),  # 代码块
            pre_tokenizers.Split(Regex(r'<equation>(.*?)</equation>'), behavior='isolated'),  # 方程块
            Digits(individual_digits=False),          # 数字处理
            Punctuation(behavior='isolated'),         # 标点符号
            Metaspace(),                              # 空格处理
            ByteLevel(add_prefix_space=True),
            Whitespace()
        ])
    
    tokenizer.decoder = decoders.ByteLevel()
    
    # 配置特殊token
    special_tokens = [
        "<unk>", "<s>", "</s>", "<|SBOS|>", "<|SEOS|>",
        "<PAD>", "<mask>", "<cls>", "<sep>",

        # 数学运算标记
        "<math>", "<formula>", "<equation>", "<matrix>", "<integral>", 
        "<derivative>", "<summation>", "<limit>", "<vector>", "<tensor>",
        "<probability>", "<fraction>", "<sqrt>", "<log>", "<trig>", "<set>",
        
        # 代码生成标记
        "<code>", "<function>", "<class>", "<loop>", "<condition>", "<variable>",
        "<import>", "<comment>", "<api_call>", "<sql>", "<json>", "<xml>", 
        "<yaml>", "<html>", "<css>", "<js>", "<python>", "<java>", "<cpp>",
        
        # 推理结构标记
        "<reasoning>", "<step>", "<proof>", "<theorem>", "<lemma>",
        "<conjecture>", "<corollary>", "<axiom>", "<definition>",
        
        # 数据操作标记
        "<data>", "<table>", "<row>", "<column>", "<cell>", "<dataset>",
        "<stat>", "<mean>", "<median>", "<std>", "<distribution>",
        
        # 领域特定标记
        "<medical>", "<legal>", "<financial>", "<scientific>"

        # 添加XML结构标记
        "<input>", "</input>",
        "<instruction>", "</instruction>",
        "<reasoning_content>", "</reasoning_content>",
        "<output>", "</output>",
        "<source>", "</source>",
        "<score>", "</score>",
    ]

    # 多阶段训练
    phase_vocab_sizes = [
        int(vocab_size * 0.4),  # 第一阶段：基础词汇
        int(vocab_size * 0.8),  # 第二阶段：中等词汇
        vocab_size              # 第三阶段：完整词汇
    ]
    
    # 估计总样本数（用于进度条）
    total_samples_estimate = sum(count_lines(fp) for fp in data_paths)
    logger.info(f"Estimated total samples: {total_samples_estimate}")

    # 配置训练器工作器
    if num_workers is None:
        num_workers = max(1, cpu_count() - 2)

    for phase in range(resume_phase-1, num_phases):
        current_phase = phase + 1
        phase_vocab = phase_vocab_sizes[phase]
        logger.info(f"Starting training phase {current_phase}/{num_phases} with vocab size {phase_vocab}")
        
        # 配置训练器（BpeTrainer）
        trainer = trainers.BpeTrainer(
            vocab_size=phase_vocab,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=special_tokens,
            min_frequency=min_frequency,
            max_piece_length=max_piece_length,  # 限制token最大长度
            show_progress=True,                 # 显示进度条
            continuing_subword_prefix='##',     # 子词前缀
            end_of_word_suffix='</w>',          # 单词后缀
            n_threads=num_workers,              # 多线程
            shriking_factor=shriking_factor     # 缩减因子
        )
        
        # 动态批处理大小（基于内存）
        available_mem = psutil.virtual_memory().available / (1024 ** 3)  # GB
        dynamic_batch_size = max(1000, min(50000, int(available_mem * 1500)))
        logger.info(f"Dynamic batch size: {dynamic_batch_size} | Available memory: {available_mem:.2f} GB")

        # 批处理生成器
        def batched_data_generator() -> Iterator[List[str]]:
            # 批处理生成器
            for batch in batch_generator(
                data_paths, 
                batch_size=dynamic_batch_size,
                num_workers=cpu_count()
            ):
                yield batch
                gc.collect()
        
        # 训练tokenizer
        start_time = time.time()
        
        # 估计总样本数（用于进度条）
        total_samples_estimate = sum(count_lines(fp) for fp in data_paths)
        logger.info(f"Estimated total samples: {total_samples_estimate}")
        
        tokenizer.train_from_iterator(
            batched_data_generator(),
            trainer=trainer,
            length=total_samples_estimate
        )
        
        phase_time = time.time() - start_time
        logger.info(f"Phase {phase+1} completed in {phase_time:.2f} seconds")
        logger.info(f"Phase memory usage: {memory_usage()}")
        
        # 每阶段保存临时tokenizer
        phase_dir = os.path.join(save_dir, f"phase_{phase+1}")
        os.makedirs(phase_dir, exist_ok=True)
        tokenizer.save(os.path.join(phase_dir, "tokenizer.json"))
    
    # 最终阶段配置
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",   # 单文本封装
        pair="<s> $A </s> $B:1 </s>:1", # 双文本封装
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>"))
        ]
    )
    
    # 验证特殊token映射
    missing_tokens = []
    for token in special_tokens:
        if tokenizer.token_to_id(token) is None:
            logger.warning(f"Token missing: {token}")
            missing_tokens.append(token)
    
    if missing_tokens:
        logger.info(f"Adding {len(missing_tokens)} missing tokens")
        tokenizer.add_tokens(missing_tokens)
    
    # 保存最终tokenizer
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    # 创建配置文件
    create_tokenizer_config(save_dir, vocab_size, model_max_length)
    logger.info(f"Tokenizer saved to {save_dir}")

def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 测试基本属性
    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # 计算覆盖率
    print("\n=== 特殊标记覆盖率测试 ===")
    required_tokens = [
        "<formula>", "<equation>", "<integral>", "<derivative>",
        "<code_block>", "<function>", "<sql>", "<json>",
        "<reasoning>", "<step>", "<proof>", "<theorem>",
        "<data>", "<table>", "<stat>", "<distribution>",
        "<user>", "<assistant>", "<system>"
    ]
    coverage = sum(1 for t in required_tokens if t in tokenizer.vocab)
    print(f"Required special tokens coverage: {coverage}/{len(required_tokens)}")

    # XML结构测试
    print("\n=== XML结构保留测试 ===")
    xml_test_text = (
        "<input>网络安全漏洞分析</input>"
        "<reasoning_content>思考过程文本...</reasoning_content>"
        "<output>漏洞修复方案...</output>"
        "<source>CVE-2023-1234</source>"
        "<score>8.5</score>"
    )
    
    tokens = tokenizer.tokenize(xml_test_text)
    decoded = tokenizer.decode(tokenizer.encode(xml_test_text))
    
    print(f"Original: {xml_test_text}")
    print(f"Tokenized: {tokens}")
    print(f"Decoded: {decoded}")
    
    # 验证XML标签是否保留为完整token
    required_xml_tags = [
        "<input>", "</input>",
        "<reasoning_content>", "</reasoning_content>",
        "<output>", "</output>",
        "<source>", "</source>",
        "<score>", "</score>"
    ]
    
    missing_tags = [tag for tag in required_xml_tags if tag not in tokens]
    if missing_tags:
        print(f"Missing XML tags: {missing_tags}")
    else:
        print("All XML tags preserved as single tokens")

    # 测试数学表达式处理
    print("\n=== 高等数学处理测试 ===")
    math_expr = "计算曲线积分: ∫_C (x^2 dx + y^2 dy) <formula>，其中C是圆x^2+y^2=1<equation>"
    math_encoded = tokenizer(math_expr)
    math_decoded = tokenizer.decode(math_encoded['input_ids'])
    print(f"Original: {math_expr}")
    print(f"Decoded:  {math_decoded}")
    print("Math tokens preserved:", all(t in math_decoded for t in ["<formula>", "<equation>"]))

    # 测试代码生成
    print("\n=== 多语言代码生成测试 ===")
    code_snippet = """
    <code_block>
    <function>def calculate_stats(data: list) -> dict:
        <comment># 计算统计指标
        <stat>mean = sum(data) / len(data)
        <stat>variance = sum((x - mean) ** 2 for x in data) / len(data)
        <stat>std = math.sqrt(variance)
        return {"mean": mean, "std": std}
    
    <sql>SELECT * FROM users WHERE age > 30;
    <json>{"name": "John", "age": 30, "city": "New York"}
    """
    code_encoded = tokenizer(code_snippet)
    code_decoded = tokenizer.decode(code_encoded['input_ids'])
    print(f"Original: {code_snippet[:200]}...")
    print(f"Decoded:  {code_decoded[:200]}...")
    print("Code structure preserved:", all(t in code_decoded for t in ["<code_block>", "<function>", "<sql>", "<json>"]))

    # 测试逻辑推理
    print("\n=== 逻辑推理测试 ===")
    reasoning_text = """
    <reasoning>
    <step>1. 已知所有猫都是哺乳动物
    <step>2. 已知Tom是一只猫
    <theorem>根据三段论推理
    <conclusion>因此Tom是哺乳动物
    """
    reasoning_encoded = tokenizer(reasoning_text)
    reasoning_decoded = tokenizer.decode(reasoning_encoded['input_ids'])
    print(f"Original: {reasoning_text.strip()}")
    print(f"Decoded:  {reasoning_decoded.strip()}")
    print("Reasoning tags preserved:", all(t in reasoning_decoded for t in ["<reasoning>", "<step>", "<theorem>"]))

    # 测试罕见词处理
    print("\n=== 罕见词处理测试 ===")
    rare_words = [
        "pneumonoultramicroscopicsilicovolcanoconiosis",
        "floccinaucinihilipilification",
        "antidisestablishmentarianism",
        "supercalifragilisticexpialidocious"
    ]
    for word in rare_words:
        tokens = tokenizer.tokenize(word)
        print(f"Word: {word}")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        print("="*50)

    # 测试多语言支持
    print("\n=== 多语言支持测试 ===")
    multilingual_text = """
    English: The quick brown fox jumps over the lazy dog.
    Spanish: El rápido zorro marrón salta sobre el perro perezoso.
    French: Le rapide renard brun saute par-dessus le chien paresseux.
    German: Der schnelle braune Fuchs springt über den faulen Hund.
    Chinese: 敏捷的棕色狐狸跳过懒惰的狗。
    Japanese: 素早い茶色の狐がのろまな犬を飛び越えます。
    Arabic: الثعلب البني السريع يقفز فوق الكلب الكسول.
    """
    tokens = tokenizer.tokenize(multilingual_text)
    print(f"Multilingual text token count: {len(tokens)}")
    print("Sample tokens:", tokens[:20])

    # 测试对话系统
    print("\n=== 对话系统测试 ===")
    dialog_text = """
    <system>你是一个数学助手
    <user>请计算圆的面积，半径是5cm
    <assistant>圆的面积公式是πr²
    <assistant>计算结果: 3.14 * 5² = 78.5 cm²
    """
    dialog_encoded = tokenizer(dialog_text)
    dialog_decoded = tokenizer.decode(dialog_encoded['input_ids'])
    print(f"Original: {dialog_text.strip()}")
    print(f"Decoded:  {dialog_decoded.strip()}")
    print("Dialog tags preserved:", all(t in dialog_decoded for t in ["<system>", "<user>", "<assistant>"]))

    # 测试文本生成
    print("\n=== 文本生成测试 ===")
    story_text = """
    从前，在一个遥远的王国里，住着一位聪明的公主。她解决了王国里所有的难题，
    从数学谜题到逻辑悖论。有一天，王国遇到了一个特别棘手的问题:<equation>E=mc²</equation>
    <reasoning>公主思考了三天三夜
    <conclusion>最终她找到了解决方案，拯救了王国。
    """
    story_encoded = tokenizer(story_text)
    story_decoded = tokenizer.decode(story_encoded['input_ids'])
    print(f"Original: {story_text[:200]}...")
    print(f"Decoded:  {story_decoded[:200]}...")
    print("Narrative elements preserved:", all(t in story_decoded for t in ["<equation>", "<reasoning>"]))

    # 测试token效率
    print("\n=== Token效率测试 ===")
    test_text = """
    def fibonacci(n: int) -> int:
        \"\"\"计算斐波那契数列的第n项\"\"\"
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n+1):
            a, b = b, a + b
        return b
    
    # 测试函数
    print(fibonacci(10))  # 应输出55
    
    \"\"\"数学公式验证:
    当n趋向无穷大时，fibonacci(n) ≈ φ^n / √5
    其中φ是黄金分割比 (1+√5)/2
    \"\"\"
    """
    encoding = tokenizer(test_text)
    token_count = len(encoding['input_ids'])
    print(f"Text length: {len(test_text)} characters")
    print(f"Token count: {token_count} tokens")
    print(f"Compression ratio: {len(test_text)/token_count:.2f} chars/token")

    # 测试聊天模板
    messages = [
        {"role": "system", "content": "你是一个AI助手，擅长数学和编程。"},
        {"role": "user", "content": "请解释勾股定理"},
        {"role": "assistant", "content": "勾股定理指出：在直角三角形中，直角边的平方和等于斜边的平方。即a² + b² = c²"},
        {"role": "user", "content": "请用Python实现验证"},
    ]
    
    print("\n=== 聊天模板测试 ===")
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")

def main():
    # 设置全局随机数种子
    set_seed(42)

    # 指定从第三阶段恢复训练
    resume_phase = 3

    # 配置路径（支持多个文件）
    data_paths = [
        "data_path1.jsonl",
        "data_path2.jsonl",
    ]
    save_dir = "./tokenizer"

    # 训练tokenizer
    train_tokenizer(
        data_paths=data_paths,
        save_dir=save_dir,
        vocab_size=50000,           # 词汇表大小
        model_max_length=16384,     # 支持长上下文
        max_piece_length=16384,     # 最词条长度
        min_frequency=2,            # 最小词频
        num_phases=3,               # 3阶段训练
        num_workers=cpu_count()-2,  # 使用CPU核心数
        shriking_factor=0.75,       # 缩减因子
        resume_phase=resume_phase   # 恢复训练
    )

    # 评估tokenizer
    eval_results = eval_tokenizer(save_dir)

    # 保存评估结果
    with open(os.path.join(save_dir, "evaluation.json"), "w") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    main()