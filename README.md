<div align="center">

# ✨ StellarByte ✨

<p>把每个字节都点亮成一盏灯，照见古今同望的夜空。</p>

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow?style=flat-square)](https://huggingface.co/)
[![Blog](https://img.shields.io/badge/Blog-ByteWyrm-pink?style=flat-square)](https://blog.devnest.top/)

</div>

## 📚 简介

StellarByte 是一个基于 Transformer 架构的高性能语言模型实现，与 HuggingFace 生态完全兼容。该项目融合了多种现代 Transformer 优化技术，旨在提供高效、灵活且易于使用的深度学习框架，适用于自然语言处理和生成式 AI 任务。

## ✨ 特性

- 🚀 **高性能实现**：集成 FlashAttention、KV 缓存等优化技术
- 🧩 **模块化设计**：各组件可独立使用或组合
- 🔄 **XPos 旋转位置编码**：改进的 RoPE 位置编码，提高长序列建模能力
- 🛠️ **丰富的优化技术**：
  - ⚙️ DeepNorm 归一化策略
  - 🔍 LayerScale 初始化技术
  - 🔀 DropPath 正则化
  - ⚡ 并行残差连接
- 📊 **参数高效微调**：内置 LoRA 低秩适应实现
- 🤗 **HuggingFace 兼容**：无缝集成 Transformers 生态系统
- 📝 **清晰的代码结构**：易于理解和扩展

## 🔥 StellarByte-LLM 1.0模型架构

### 模型整体架构 **ByteTransformer**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ByteTransformer                               │
│                                                                         │
│  ┌───────────────┐                                                      │
│  │ Embed Tokens  │                                                      │
│  └───────┬───────┘                                                      │
│          ▼                                                              │
│  ┌───────────────┐                                                      │
│  │Embed Dropout  │                                                      │
│  └───────┬───────┘                                                      │
│          ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                     Decoder Layers Stack                        │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │                   ByteDecoderLayer × N                  │    │    │
│  │  │  ┌─────────────┐    ┌─────────────┐                     │    │    │
│  │  │  │   RMSNorm   │    │   RMSNorm   │                     │    │    │
│  │  │  └──────┬──────┘    └──────┬──────┘                     │    │    │
│  │  │         ▼                  ▼                            │    │    │
│  │  │  ┌─────────────┐    ┌─────────────┐                     │    │    │
│  │  │  │ Multi-Head  │    │     MLP     │                     │    │    │
│  │  │  │ Attention   │    │ (Gated MLP) │                     │    │    │
│  │  │  └──────┬──────┘    └──────┬──────┘                     │    │    │
│  │  │         │                  │                            │    │    │
│  │  │         ▼                  ▼                            │    │    │
│  │  │  ┌─────────────┐    ┌─────────────┐                     │    │    │
│  │  │  │ LayerScale  │    │ LayerScale  │                     │    │    │
│  │  │  └──────┬──────┘    └──────┬──────┘                     │    │    │
│  │  │         │                  │                            │    │    │
│  │  │         ▼                  ▼                            │    │    │
│  │  │  ┌─────────────┐    ┌─────────────┐                     │    │    │
│  │  │  │  DropPath   │    │  DropPath   │                     │    │    │
│  │  │  └──────┬──────┘    └──────┬──────┘                     │    │    │
│  │  │         │                  │                            │    │    │
│  │  │         └──────────┬───────┘                            │    │    │
│  │  │                    ▼                                    │    │    │
│  │  │             Residual Connection                         │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│          ▼                                                              │
│  ┌───────────────┐                                                      │
│  │    RMSNorm    │                                                      │
│  └───────┬───────┘                                                      │
│          ▼                                                              │
│  ┌───────────────┐                                                      │
│  │    LM Head    │                                                      │
│  └───────────────┘                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 解码层 **ByteDecoderLayer**
```
┌──────────────────────────────────────────────────────────────────────────┐
│                        ByteDecoderLayer                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────┐     ┌───────────────────────────┐                          │
│   │         │     │                           │                          │
│   │  Input  │────►│        RMSNorm_attn       │─┐                        │
│   │   [x]   │     │                           │ │                        │
│   │         │     └───────────────────────────┘ │                        │
│   └─────────┘                                   ▼                        │
│        │         ┌────────────────────────────────────────────┐          │
│        │         │           MultiHeadSelfAttention           │          │
│        │         │                                            │          │
│        │         │  ┌─────────┐  ┌─────────┐  ┌─────────┐     │          │
│        │         │  │   W_q   │  │   W_k   │  │   W_v   │     │          │
│        │         │  └─────────┘  └─────────┘  └─────────┘     │          │
│        │         │       │            │            │          │          │
│        │         │       ▼            ▼            ▼          │          │
│        │         │  ┌─────────────────────────────────────┐   │          │
│        │         │  │        XPosRotaryEmbedding          │   │          │
│        │         │  └─────────────────────────────────────┘   │          │
│        │         │       │            │            │          │          │
│        │         │       ▼            ▼            ▼          │          │
│        │         │  ┌─────────────────────────────────────┐   │          │
│        │         │  │       Scaled Dot-Product Attn       │   │          │
│        │         │  └─────────────────────────────────────┘   │          │
│        │         │                    │                       │          │
│        │         │                    ▼                       │          │
│        │         │  ┌─────────────────────────────────────┐   │          │
│        │         │  │                W_o                  │   │          │
│        │         │  └─────────────────────────────────────┘   │          │
│        │         │                    │                       │          │
│        │         └────────────────────┼───────────────────────┘          │
│        │                              │                                  │
│        │                              ▼                                  │
│        │         ┌─────────────────────────────────────┐                 │
│        │         │           DropPath(p)               │                 │
│        │         └─────────────────────────────────────┘                 │
│        │                              │                                  │
│        │                              ▼                                  │
│        │         ┌─────────────────────────────────────┐                 │
│        │         │           LayerScale_attn           │                 │
│        │         └─────────────────────────────────────┘                 │
│        │                              │                                  │
│        ├──────────────────────────────┼──────────────────────────────────┤
│        │                              ▼                                  │
│        │         ┌─────────────────────────────────────┐                 │
│        └────────►│               Add                   │                 │
│                  └─────────────────────────────────────┘                 │
│                                     │                                    │
│                                     ▼                                    │
│                  ┌─────────────────────────────────────┐                 │
│                  │            RMSNorm_ffn              │                 │
│                  └─────────────────────────────────────┘                 │
│                                     │                                    │
│                                     ▼                                    │
│                  ┌──────────────────────────────────────┐                │
│                  │                MLP                   │                │
│                  │   ┌───────────┐     ┌───────────┐    │                │
│                  │   │    w1     │     │    w3     │    │                │
│                  │   └───────────┘     └───────────┘    │                │
│                  │         │                │           │                │
│                  │         ▼                ▼           │                │
│                  │   ┌───────────┐     ┌───────────┐    │                │
│                  │   │   SiLU    │     │  Linear   │    │                │
│                  │   └───────────┘     └───────────┘    │                │
│                  │         │                │           │                │
│                  │         └────────┬─────────┘         │                │
│                  │                  │                   │                │
│                  │                  ▼                   │                │
│                  │   ┌───────────────────────────┐      │                │
│                  │   │       Element-wise *      │      │                │
│                  │   └───────────────────────────┘      │                │
│                  │                  │                   │                │
│                  │                  ▼                   │                │
│                  │   ┌───────────────────────────┐      │                │
│                  │   │            w2             │      │                │
│                  │   └───────────────────────────┘      │                │
│                  │                  │                   │                │
│                  └──────────────────┼───────────────────┘                │
│                                     │                                    │
│                                     ▼                                    │
│                  ┌─────────────────────────────────────┐                 │
│                  │           DropPath(p)               │                 │
│                  └─────────────────────────────────────┘                 │
│                                     │                                    │
│                                     ▼                                    │
│                  ┌─────────────────────────────────────┐                 │
│                  │           LayerScale_mlp            │                 │
│                  └─────────────────────────────────────┘                 │
│                                     │                                    │
│                                     ▼                                    │
│                  ┌─────────────────────────────────────┐                 │
│                  │               Add                   │───────────────► │
│                  └─────────────────────────────────────┘                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 多头自注意力层 **ByteMultiHeadSelfAttention**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MultiHeadSelfAttention                               │
│                                                                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │      W_q      │  │      W_k      │  │      W_v      │                │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘                │
│          ▼                  ▼                  ▼                        │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                │
│  │  Reshape to   │  │  Reshape to   │  │  Reshape to   │                │
│  │  Multi-head   │  │  Multi-head   │  │  Multi-head   │                │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘                │
│          │                  │                  │                        │
│          ▼                  ▼                  ▼                        │
│  ┌─────────────────────────────────────────────────────────┐            │
│  │                 XPosRotaryEmbedding                     │            │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │            │
│  │  │     Cos     │  │     Sin     │  │    Scale    │      │            │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │            │
│  │         └──────────┬─────┘                │             │            │
│  │                    ▼                      ▼             │            │
│  │             Apply Rotation          Apply Scaling       │            │
│  └─────────────┬─────────────────────┬─────────────────────┘            │
│                │                     │                                  │
│                ▼                     ▼                                  │
│  ┌───────────────────┐      ┌────────────────────┐                      │
│  │ Rotated Q Vectors │      │ Rotated K Vectors  │                      │
│  └─────────┬─────────┘      └──────────┬─────────┘                      │
│            │                           │                                │
│            └───────────┬───────────────┘                                │
│                        ▼                                                │
│  ┌─────────────────────────────────────────────┐                        │
│  │              Attention Scores               │                        │
│  │  (Q·K^T / sqrt(d_k)) + Attention Mask       │                        │
│  └───────────────────┬─────────────────────────┘                        │
│                      ▼                                                  │
│  ┌─────────────────────────────────────────────┐                        │
│  │                  Softmax                    │                        │
│  └───────────────────┬─────────────────────────┘                        │
│                      ▼                                                  │
│  ┌─────────────────────────────────────────────┐                        │
│  │             Attention Dropout               │                        │
│  └───────────────────┬─────────────────────────┘                        │
│                      ▼                                                  │
│  ┌─────────────────────────────────────────────┐                        │
│  │         Weighted Sum (Attention·V)          │                        │
│  └───────────────────┬─────────────────────────┘                        │
│                      ▼                                                  │
│  ┌─────────────────────────────────────────────┐                        │
│  │       Reshape and Concatenate Heads         │                        │
│  └───────────────────┬─────────────────────────┘                        │
│                      ▼                                                  │
│  ┌─────────────────────────────────────────────┐                        │
│  │                    W_o                      │                        │
│  └───────────────────┬─────────────────────────┘                        │
│                      ▼                                                  │
│  ┌─────────────────────────────────────────────┐                        │
│  │              Residual Dropout               │                        │
│  └───────────────────┬─────────────────────────┘                        │
│                      ▼                                                  │
│                   Output                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 位置旋转嵌入层 **XPosRotaryEmbedding**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                      XPosRotaryEmbedding                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────┐     ┌─────────┐                                           │
│   │   xq    │     │   xk    │                                           │
│   └─────────┘     └─────────┘                                           │
│        │              │                                                 │
│        ▼              ▼                                                 │
│   ┌─────────────────────────────────────────────────────────┐           │
│   │                  _get_cos_sin_scale                     │           │
│   │                                                         │           │
│   │   ┌─────────────────────────────────────────────────┐   │           │
│   │   │                 RotaryCache                     │   │           │
│   │   │  (缓存cos/sin/scale张量，避免重复计算)              │   │           │
│   │   └─────────────────────────────────────────────────┘   │           │
│   │                                                         │           │
│   │   ┌─────────┐     ┌─────────┐     ┌─────────┐           │           │
│   │   │   cos   │     │   sin   │     │  scale  │           │           │
│   │   └─────────┘     └─────────┘     └─────────┘           │           │
│   └─────────────────────────────────────────────────────────┘           │
│        │              │              │                                  │
│        ▼              ▼              ▼                                  │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐                           │
│   │ xq*scale│     │ xk/scale│     │_rotate_half│                        │
│   └─────────┘     └─────────┘     └─────────┘                           │
│        │              │              │                                  │
│        │              │              │                                  │
│        ▼              ▼              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐           │
│   │  xq_out = xq_scaled * cos + _rotate_half(xq_scaled) * sin           │
│   │  xk_out = xk_scaled * cos + _rotate_half(xk_scaled) * sin           │
│   └─────────────────────────────────────────────────────────┘           │
│        │              │                                                 │
│        ▼              ▼                                                 │
│   ┌─────────┐     ┌─────────┐                                           │
│   │ xq_out  │     │ xk_out  │                                           │
│   └─────────┘     └─────────┘                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 层归一化层 **RMSNorm**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                             RMSNorm                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────┐                                                           │
│   │  Input  │                                                           │
│   │    x    │                                                           │
│   └─────────┘                                                           │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────────────────────────────────────────────────┐           │
│   │                    _rms_norm                            │           │
│   │                                                         │           │
│   │   ┌─────────────────────────────────────────────────┐   │           │
│   │   │  1. 提升到float32计算平方和均值                     │  │            │
│   │   │  rms = torch.mean(x.to(torch.float32).pow(2),   │   │           │
│   │   │                   dim=-1, keepdim=True)         │   │           │
│   │   └─────────────────────────────────────────────────┘   │           │
│   │                          │                              │           │
│   │                          ▼                              │           │
│   │   ┌─────────────────────────────────────────────────┐   │           │
│   │   │  2. 计算平方根倒数，并确保数值稳定                    │   │           │
│   │   │  inv_rms = torch.rsqrt(rms.clamp_min(eps))      │   │           │
│   │   └─────────────────────────────────────────────────┘   │           │
│   │                          │                              │           │
│   │                          ▼                              │           │
│   │   ┌─────────────────────────────────────────────────┐   │           │
│   │   │  3. 转回原始数据类型                               │   │           │
│   │   │  inv_rms = inv_rms.to(x.dtype)                  │   │           │
│   │   │  weight = weight.to(x.dtype)                    │   │           │
│   │   └─────────────────────────────────────────────────┘   │           │
│   │                          │                              │           │
│   │                          ▼                              │           │
│   │   ┌─────────────────────────────────────────────────┐   │           │
│   │   │  4. 归一化并应用可学习权重                          │   │           │
│   │   │  return (x * inv_rms) * weight                  │   │           │
│   │   └─────────────────────────────────────────────────┘   │           │
│   │                                                         │           │
│   └─────────────────────────────────────────────────────────┘           │
│                          │                                              │
│                          ▼                                              │
│   ┌─────────┐                                                           │
│   │ Output  │                                                           │
│   └─────────┘                                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 多层感知机层 **GatedMLP**
```
┌─────────────────────────────────────────────────────────────────────────┐
│                              MLP                                        │
│                                                                         │
│  ┌───────────────┐                  ┌───────────────┐                   │
│  │      W1       │                  │      W3       │                   │
│  └───────┬───────┘                  └───────┬───────┘                   │
│          ▼                                  ▼                           │
│  ┌───────────────┐                  ┌───────────────┐                   │
│  │  SiLU (Swish) │                  │   (No Activ)  │                   │
│  └───────┬───────┘                  └───────┬───────┘                   │
│          │                                  │                           │
│          └─────────────────┬────────────────┘                           │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────┐                    │
│  │              Element-wise Multiply              │                    │
│  └───────────────────────┬─────────────────────────┘                    │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────┐                    │
│  │                      W2                         │                    │
│  └───────────────────────┬─────────────────────────┘                    │
│                          ▼                                              │
│  ┌─────────────────────────────────────────────────┐                    │
│  │                    Dropout                      │                    │
│  └───────────────────────┬─────────────────────────┘                    │
│                          ▼                                              │
│                       Output                                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🔧 安装

### 环境要求

- Python 3.8+
- PyTorch 2.5.1+
- CUDA 11.8+ (GPU加速，可选)

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/HxCodeWarrior/StellarByte.git
cd StellarByte

# 安装依赖
pip install -r requirements.txt

# 如果需要开发环境
pip install -r requirements.txt[dev]

# 安装开发版本（暂未实现）
# pip install -e .
```

### 依赖说明

项目依赖已按功能模块分类整理：

- **核心依赖**：PyTorch、Transformers、数据处理库
- **模型组件**：位置编码、注意力机制等实现
- **可视化与监控**：实验追踪、指标可视化
- **测试与开发**：单元测试、类型检查
- **分布式训练**：多GPU/多节点训练支持
- **性能优化**：内存优化、计算加速


## 🚀 快速开始

```python
import torch
from stellarbyte import ByteTransformer, ByteConfig

# 创建配置
config = ByteModelConfig(
    vocab_size=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072
)

# 初始化模型
model = ByteTransformer(config)

# 准备输入
inputs = torch.randint(0, 32000, (1, 512))

# 前向传播
outputs = model(inputs)
```

## 📋 使用示例

### 从 HuggingFace 加载预训练模型

```python
from stellarbyte import ByteTransformer
from transformers import AutoTokenizer

# 加载模型和分词器
model = ByteTransformer.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")

# 编码文本
inputs = tokenizer("把每个字节都点亮成一盏灯", return_tensors="pt")

# 生成文本
outputs = model.generate(inputs.input_ids, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### 使用 LoRA 进行参数高效微调

```python
from stellarbyte import ByteTransformer, LoRAConfig
from stellarbyte.lora import apply_lora_to_model

# 加载基础模型
model = ByteTransformer.from_pretrained("path/to/model")

# 配置 LoRA
lora_config = LoRAConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0.05
)

# 应用 LoRA 到模型
model = apply_lora_to_model(model, lora_config)

# 现在只有 LoRA 参数会被更新
```

## 📁 项目结构

```
StellarByte/
|   .gitignore
|   datasets.py
|   LICENSE
|   model_pretrain.py
|   model_stf_train.py
|   README.md
|   requirements.txt
|
+---checkpoints
+---configs
|       pretrain_config.yaml
|
+---datasets
|   |   data_preprocessor.py
|   |   pretrain_hq.jsonl
|   |
|   \---test
|           train.jsonl
|           val.jsonl
|
+---logs
+---model
|   |   Attention.py
|   |   config.py
|   |   DecoderLayer.py
|   |   MLP.py
|   |   Model.py
|   |   MoE.py
|   |   Position_Embedding.py
|   |   RMSNorm.py
|   |   __init__.py
|   |
|   +---utils
|          DropPath.py
|          KVCache.py
|          LoRA.py
|          Memory.py
|          __init__.py
|
+---model_info
+---scripts
+---test
|   |   test_Attention.py
|   |   test_datasets.py
|   |   test_DeocoderLayer.py
|   |   test_KVCache.py
|   |   test_LoRA.py
|   |   test_MLP.py
|   |   test_Position_Embedding.py
|   |   test_RMSNorm.py
|   |
|   +---test_results
|
+---tokenizer
|       special_tokens_map.json
|       tokenizer.json
|       tokenizer_config.json
|
+---utils
        checkpoint.py
        config_params.py
        logger.py
        model_info.py
        progressbar.py
```

## 🔜 开发计划

<details> 
<summary>2025.7.13</summary>

### Done:
1. 实现BaseModelConfig类，后续的超参数将逐渐迭代
2. 实现RMSNorm层归一化类
3. Transformer经典的MultiHeadAttention类

### TODO：
1. Attention应用KV缓存，添加量化机制
2. 构建基础MLP层
3. 构建基础DecoderLayer层

</details>

---

<details> 
<summary>2025.7.14</summary>

### Done:
1. 实现Attention应用缓存机制
2. 实现Attention量化机制
3. 实现基础MLP层
4. 实现基础ByteDecoderLayer层
5. 实现LoRA
6. 实现DropPath
7. 实现KVCache机制
8. 重写了Attention中与KVCache相关的部分
9. 实现模型训练中的工具组件
- 日志记录组件
- 炫酷进度条加载组件
- 模型权重管理组件
- 模型信息分析组件
10. 实现模型训练数据集加载器包括预训练数据集加载器和STF训练数据集加载器
10. 基本构建模型预训练流程

### TODO:
1. 构造Memory类并进行应用
2. 应用LoRA类
3. 构造单步推理接口 def forward_step(self, x_t, past_k, past_v) -> (out, new_k, new_v)
4. Xpos位置编码优化
- 结构性正则
- 动态theta调整
5. Attention
- num_rep 未被使用 
- 实现 线程并行/All‑Reduce
- 进一步融合FlashAttention-2
- 线性层量化quantize() 使用了 torch.quantization.quantize_dynamic()，但这仅限于线性层 + 推理，需要进一步优化以支持GPTQ/AWQ/SmoothQuant
6. KVCache
将 KVCache.append() 改为支持：
- 滑窗（sliding window）截断
- 写入位置并发锁定（if multi-thread）
- Layer-wise token位置自动偏移计算

</details>

---

<details>
<summary>2025.7.15</summary>

### Done:
1. 构建并优化模型训练组件：
- 检查点管理组件
- 意外中断保护组件
2. 实现LLM的记忆管理机制
3. 实现单步推理接口以及相关组件
4. 在模型训练过程中应用检查点管理以及意外中断保护组件
5. 关于LoRA
- 优化LoRA：支持热插拔、正则优化、减少内存占用提升速度
- 测试LoRA
6. 优化位置编码
- 移除旋转位置编码的本地缓存，改用全局缓存类RotaryCache
- 添加可学习的缩放因子参数到XPosRotaryEmbedding
7. 统一了Attention部分的param参数dtype和本层计算数据dtype
- 统一计算精度为float32以提高数值稳定性
8. KVCache添加滑动窗口截断处理

### TODO：
1. LoRA进一步优化
- 非线性LoRA
- 支持Conv2d/Transformer.Conv1d注入
- 适配量化模块
- Tuner冻结层选择策略
2. Attention:
- 实现线程并行/All‑Reduce
- 进一步融合FlashAttention-2
3. 测试
- 测试数据集加载器
- 测试LoRA
- 测试Attention
- 测试Memory

### DEBUG
1. 工具脚本分析模型信息报错

</details>

---

<details>
<summary>2025.7.16</summary>

### Done:
1. 实现分布式多卡训练
2. 实现张量/模型并行
3. 整合模型训练参数，并构造参数读取器
4. 优化显存占用、提升吞吐速度
- 新增可控梯度检查点
- 指定step后清理无用现存
- 新增FlashAttention可控开关
5. Attention 多头注意力机制优化
- 进一步融合FlashAttention-2
- 新增模型/张量并行处理
6. 模型分析脚本 model_info.py 修复如下问题：
- analyze_performance(),每次切换 batch_size 前，把 KVCache 清零并把 batch_size 设回 None，让下一轮 forward 自动重新分配缓存。
7. 测试脚本通过
- Attention测试
- datasets数据集加载器测试
- LoRA测试
8. 优化数据集加载器
- 掩码从01转换为bool类型
- 优化截断处理
9. tokenizer修复：处理tokenizer的pad填充与eos_token一样导致填充混乱，分别使用特殊标识
10. LoRA修复如下问题：
- 权重合并的线程锁作用域过大
- 确保 LoRA 增量计算 时数据类型统一为LoRA统一参数类型self.cfg.dtype or torch.float32
- 解决 LoRA注入风险，注入前判断模块是否已经是 LoRALinear，跳过注入
- 解决 LoRALinear 内部权重布局与 fan_in_fan_out 关联不足 问题，在 forward 阶段增量计算时根据 fan_in_fan_out 转置 LoRA 参数
- 解决 多适配器热切换的 activate() 未解除旧 LoRA 权重占用显存 问题，保存原始层引用，deactivate 时恢复原始层，彻底卸载旧 LoRA层
- 使用 安全 torch.load(), 当前默认 weights_only=False，但官方已宣布未来会改为 True，因此构建自动检测是否使用该参数导入函数
11. KVCache修复如下问题：
- 当 T_new >= self.max_T 条件成立时，overflow 被赋值了，但 current_len 没有被赋值。append函数中，给 current_len 赋一个初始值，且无论哪条分支都保证 current_len 已定义。
13. 使用模型分析脚本对模型进行初步分析


### TODO:
1. 实现DeepSeed
2. 实现动态剪枝
- Attention动态剪枝
- KVCache动态剪枝
3. 数据集加载器针对大数据集进行streaming优化
4. 模型分析脚本
- 绘图中文不显示
- 模型层级结构分析不透彻
- 添加模型层级结构绘图可视化

</details>

---

<details>
<summary>2025.7.17</summary>

### DONE:
1. 增强模型分析报告功能并添加可视化图表
- 添加中文支持，解决图表中文乱码问题
- 新增模型结构图、稀疏度热力图和雷达图等可视化功能
- 改进模型架构描述格式，增加更多细节
- 优化参数分布饼图样式，突出显示最大占比部分
2. 重构模型训练组件——进度条组件
- 将原有的 RichProgressBar 类重构为更灵活的 ProgressBarManager 类
- 支持训练和验证阶段的多任务管理
- 新增验证阶段指标汇总表格显示功能
- 改进进度条样式和交互逻辑
3. 重构模型训练组件,重构训练循环和验证逻辑，改进进度管理
- 将 RichProgressBar 替换为更通用的 ProgressBarManager
- 在训练和验证循环中添加进度条支持
- 改进指标计算和日志记录，添加准确率统计
- 优化设备管理和分布式训练初始化逻辑
- 调整 AMP 上下文管理以支持不同设备类型
4. 优化RMSNorm层，移除不必要的buffer注册并简化eps处理
- 不再将eps注册为buffer，直接作为tensor属性使用，简化代码结构
5. Attention多头子注意力层关于分布式训练解决隐藏bug
- 添加分布式初始化方法并改进并行通信逻辑
- 添加 init_distributed_mode 方法用于更灵活的分布式环境初始化，支持从环境变量或参数读取配置
- 重构模型并行通信组的初始化逻辑，增加对未设置分布式变量的错误检查
6. datasets数据集加载器修复标签张量中掩码未正确应用的问题
- 在PretrainDataset中，当掩码为False时，对应的标签张量值应设为-100以避免影响损失计算。此修改确保了损失函数仅计算有效标记的损失


### TODO:
1. 关键类添加调试函数
2. 统一注释风格并完善注释
3. 修复模型训练脚本中的问题：
- 无法显示一个epoch中的step进度
- 自动构建或者获取虚拟环境信息
4. 调试模型预训练脚本

</details>

---

<details>
<summary>2025.7.18</summary>

### DONE:

1. 训练脚本修复如下问题：
- (1) 性能与训练效率的问题
  - 修复学习率调度器计算方式不合理，get_lr 中将 total_iters = len(train_loader) * epochs * (restarts + 1)，但 step 实际是 per-epoch 内部 step。训练中应使用全局 step（global_step = epoch * steps_per_epoch + step），否则调度曲线不平滑。
  - 修复没有梯度累计下的正确total_step支持，在 get_lr() 和 total_iters 中未考虑 accumulation_steps，导致训练实际更新步数与预期不符，调度失衡。
  - 启用cudnn.allow.tf32，初始训练脚本设定了 torch.backends.cuda.matmul.allow_tf32 = True，但没有设置 torch.backends.cudnn.allow_tf32 = True。这会错失一半以上的 TF32 算子加速机会。
  - torch.cuda.empty_cache() 调用频繁
- (2) 分布式训练的问题，添加分布式训练
  - 修复 DDP日志未加 rank 过滤 的问题，虽然大多数日志都用 is_main_process() 做了判断，但某些异常捕获如 except Exception 或 init_distributed() 内仍会全 rank 打印，建议统一封装日志器。
  - 修复 DDP恢复不完整的问题，在恢复模型 checkpoint 时，start_step 没有继续作为 global_step 传入 train_epoch()，导致断点恢复训练时的 LR调度、日志步数、SwanLab step 不准确。
- (3) 显存管理和稳定性问题
  - 使用 model.zero_grad(set_to_none=True)，显式用 set_to_none=True 替换 zero_grad()，可释放更早的 grad 显存，提升显存效率。
  - 修复 Gradient Checkpoint 未按层粒度配置 的问题，启用了 model.gradient_checkpointing_enable()，但若模型结构较深，应配合逐层显式设置 checkpointing=True 的策略，才有实际效果。
- (4) 鲁棒性与异常恢复问题
  - 解决 异常恢复未记录 global_step 问题，检查点中只有 epoch, step，未记录 global_step，导致调度器与日志重启后不一致。
  - 解决 训练弈场捕获未细化 问题，except Exception as e: 中没有使用 traceback.print_exc()，排查问题困难。
- (5)添加 Tokenizer.embedding_sync()，检查 tokenizer 与 embedding 大小是否同步
- (6)增加标签平滑损失计算函数
- (7)添加CUDA图优化标记
2. RSMNorm修复：
- 将eps从tensor改为float类型避免重复转换，减少内存
- 将eps转换为与rms相同的设备以避免跨设备操作
- 移除冗余的inv_rms类型转换
- 用 torch._dynamo.disable() 装饰器关闭 RMSNorm 的 forward 编译，避免 CUDA Graph 内存复用冲突。
3. Attention修复：
- 添加自动调整additive_mask长度的功能
- 新增_adjust_additive_mask方法用于自动将additive_mask长度与键值序列对齐，解决KV缓存长度不匹配问题
4. Model修复：
- 修复注意力掩码和设备类型不一致的问题并添加梯度检查点
- 修复了注意力掩码与hidden_states设备类型不一致的问题，将掩码转换为相同设备和类型。
- 同时添加了梯度检查点功能以在训练时节省显存，使用非重入方式提高稳定性。
5. Logger日志记录器完善：为日志构建函数添加控制台日志级别参数
- 添加 console_level 参数以允许自定义控制台输出的日志级别，默认保持为 INFO 级别
6. 更新预训练配置参数和注释格式
- 将 eval_max_steps 改为 eval_interval 以更准确描述功能
- 更新特殊标记格式为 <|SBOS|> 和 <|SEOS|>
- 调整 vocab_size 和并行配置参数
- 为日志配置添加注释说明

### TODO:
1. 优化训练脚本：
- 添加 早停检查
- 添加 EMA，平滑收敛过程，提升精度
- 构建 metrics.py，统一管理训练指标
- 精度与训练收敛的问题
  - 使用 Label Smoothing，避免过拟合和提升泛化能力，建议支持参数配置。
  - 引入 Prompt Mask、Position Shift等训练技巧
  - PPL计算可能不准确，当前 evaluate() 中 ppl = exp(avg_loss)，但 avg_loss 是平均 token loss，若 loss mask 未正确处理，可能造成过大偏差。
- 使用FSDP，当前最大支持 DP + DDP，未使用 torch.distributed.fsdp，对于几十亿参数以上的大模型在多节点下不够高效。
2. 完善意外终止处理
- 修复意外终止后爆出大量错误的问题，当训练意外终止时，会触发异常捕获，但异常捕获后未清理环境，导致大量错误日志输出。
```
Traceback (most recent call last):
Traceback (most recent call last):
Traceback (most recent call last):
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 315, in _worker_loop
    r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/queues.py", line 113, in get
    if not self._poll(timeout):
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/connection.py", line 256, in poll
    return self._poll(timeout)
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 315, in _worker_loop
    r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/connection.py", line 423, in _poll
    r = wait([self], timeout)
        ^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/queues.py", line 113, in get
    if not self._poll(timeout):
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/connection.py", line 930, in wait
    ready = selector.select(timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/connection.py", line 256, in poll
    return self._poll(timeout)
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/selectors.py", line 415, in select
    fd_event_list = self._selector.poll(timeout)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/connection.py", line 423, in _poll
    r = wait([self], timeout)
        ^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/utils/checkpoint.py", line 202, in _handler
    self.ckpt_mgr.save_sync(self.model, self.optimizer, self.scaler,
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/connection.py", line 930, in wait
    ready = selector.select(timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/utils/checkpoint.py", line 77, in save_sync
    state = self._collect_state(model, optimizer, scaler,
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/selectors.py", line 415, in select
    fd_event_list = self._selector.poll(timeout)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/utils/checkpoint.py", line 107, in _collect_state
    "scaler_state": scaler.state_dict() if scaler else None,
                    ^^^^^^^^^^^^^^^^^^^
  File "/workspace/utils/checkpoint.py", line 202, in _handler
    self.ckpt_mgr.save_sync(self.model, self.optimizer, self.scaler,
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/amp/grad_scaler.py", line 622, in state_dict
    "scale": self.get_scale(),
             ^^^^^^^^^^^^^^^^
  File "/workspace/utils/checkpoint.py", line 77, in save_sync
    state = self._collect_state(model, optimizer, scaler,
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/amp/grad_scaler.py", line 550, in get_scale
    else cast(float, scale.item())
                     ^^^^^^^^^^^^
  File "/workspace/utils/checkpoint.py", line 107, in _collect_state
    "scaler_state": scaler.state_dict() if scaler else None,
                    ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/amp/grad_scaler.py", line 622, in state_dict
    "scale": self.get_scale(),
             ^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/amp/grad_scaler.py", line 550, in get_scale
    else cast(float, scale.item())
                     ^^^^^^^^^^^^
RuntimeError: CUDA error: initialization error
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

RuntimeError: CUDA error: initialization error
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 315, in _worker_loop
    r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/queues.py", line 113, in get
    if not self._poll(timeout):
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/connection.py", line 256, in poll
    return self._poll(timeout)
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/connection.py", line 423, in _poll
    r = wait([self], timeout)
        ^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/connection.py", line 930, in wait
    ready = selector.select(timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/selectors.py", line 415, in select
    fd_event_list = self._selector.poll(timeout)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/utils/checkpoint.py", line 202, in _handler
    self.ckpt_mgr.save_sync(self.model, self.optimizer, self.scaler,
  File "/workspace/utils/checkpoint.py", line 77, in save_sync
    state = self._collect_state(model, optimizer, scaler,
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/utils/checkpoint.py", line 107, in _collect_state
    "scaler_state": scaler.state_dict() if scaler else None,
                    ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/amp/grad_scaler.py", line 622, in state_dict
    "scale": self.get_scale(),
             ^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/amp/grad_scaler.py", line 550, in get_scale
    else cast(float, scale.item())
                     ^^^^^^^^^^^^
RuntimeError: CUDA error: initialization error
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

Traceback (most recent call last):
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap
    self.run()
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/utils/data/_utils/worker.py", line 315, in _worker_loop
    r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/queues.py", line 113, in get
    if not self._poll(timeout):
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/connection.py", line 256, in poll
    return self._poll(timeout)
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/connection.py", line 423, in _poll
    r = wait([self], timeout)
        ^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/multiprocessing/connection.py", line 930, in wait
    ready = selector.select(timeout)
            ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/selectors.py", line 415, in select
    fd_event_list = self._selector.poll(timeout)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/utils/checkpoint.py", line 202, in _handler
    self.ckpt_mgr.save_sync(self.model, self.optimizer, self.scaler,
  File "/workspace/utils/checkpoint.py", line 77, in save_sync
    state = self._collect_state(model, optimizer, scaler,
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/utils/checkpoint.py", line 107, in _collect_state
    "scaler_state": scaler.state_dict() if scaler else None,
                    ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/amp/grad_scaler.py", line 622, in state_dict
    "scale": self.get_scale(),
             ^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/amp/grad_scaler.py", line 550, in get_scale
    else cast(float, scale.item())
                     ^^^^^^^^^^^^
RuntimeError: CUDA error: initialization error
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

[2025-07-18 13:12:28] [WARNING] [ByteLogger] 💀 收到 SIGINT，写入完整检查点 …
[2025-07-18 13:12:28] [ERROR] [ByteLogger] 训练异常: DataLoader worker (pid 2124) exited unexpectedly with exit code 1. Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.
[2025-07-18 13:12:28] [INFO] [ByteLogger] 💀 异常退出，正在保存检查点…
Traceback (most recent call last):
  File "/workspace/model_pretrain.py", line 685, in <module>
    train(args, logger)
  File "/workspace/model_pretrain.py", line 636, in train
    raise e
  File "/workspace/model_pretrain.py", line 597, in train
    train_epoch(
  File "/workspace/model_pretrain.py", line 397, in train_epoch
    scaler.scale(loss).backward()
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_tensor.py", line 648, in backward
    torch.autograd.backward(
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/autograd/__init__.py", line 353, in backward
    _engine_run_backward(
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/autograd/graph.py", line 824, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/utils/checkpoint.py", line 202, in _handler
    self.ckpt_mgr.save_sync(self.model, self.optimizer, self.scaler,
  File "/workspace/utils/checkpoint.py", line 77, in save_sync
    state = self._collect_state(model, optimizer, scaler,
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/utils/checkpoint.py", line 107, in _collect_state
    "scaler_state": scaler.state_dict() if scaler else None,
                    ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/amp/grad_scaler.py", line 622, in state_dict
    "scale": self.get_scale(),
             ^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/amp/grad_scaler.py", line 550, in get_scale
    else cast(float, scale.item())
                     ^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/utils/data/_utils/signal_handling.py", line 73, in handler
    _error_if_any_worker_fails()
RuntimeError: DataLoader worker (pid 2124) exited unexpectedly with exit code 1. Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.
```
3. 解决 未检测梯度异常（NaN）问题，若 loss = NaN、grad = inf，应立即中止训练保存 checkpoint，避免浪费资源。

### DEBUG
1. 找出torch.utils.checkpoint问题根源并进行修复
```
/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py:838: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  return fn(*args, **kwargs)
```

---

<details>
<summary>2025.7.19</summary>

### DONE
1. 给每个组件添加专属标志
2. 完善requirements.txt
3. 优化训练脚本，添加异常自动处理
4. 测试训练脚本通过
- 解决存在于多头自注意力层中的torch.utils.checkpoint配置use_reentrant=False问题
- 修复意外终止后爆出大量错误的问题，当训练意外终止时，会触发异常捕获，但异常捕获后未清理环境，导致大量错误日志输出。
5. 新增多阶段训练的tokenizer实现
- 支持数学表达式、代码块和XML结构的特殊处理
- 多阶段渐进式词汇表训练
- 内存优化的批处理生成器
- 完整的配置文件和特殊token支持
- 内置评估功能验证tokenizer效果
6. 添加环境设置脚本和安装文档
- 添加 Windows 和 Unix 的环境设置脚本，用于自动化安装和配置开发环境
- 添加 INSTALL.md 和 CONTRIBUTING.md 文档，提供详细的安装和贡献指南
- 添加 setup.py 用于管理项目依赖和安装配置
- 环境设置脚本支持以下功能：
  - 检查系统依赖
  - 创建虚拟环境
  - 安装项目依赖
  - 验证安装
  - 支持开发环境和 CUDA 选项
7. 优化位置编码
- 添加max_seq_len参数，支持预计算并缓存最大长度的位置编码
- 优化_get_cos_sin_scale方法，支持通过offset参数获取指定窗口的编码切片
8. 优化Attention，添加repeat_kv方法实现Grouped-Query Attention用于重复Key/Value张量以匹配Query的头数
9. 重构门控多层感知机模块并添加残差连接
- 合并w1和w3为共享参数的w13线性层
- 使用GEGLU门控结构替代原有实现
- 添加ByteRMSNorm归一化层
- 引入残差连接提升梯度流动
10. 构建基础的MoE层，但是结构逻辑还不完善无法应用，需要优化
11. 构建基础的MeMory记忆机制，支持以下功能：
- update(layer_idx, new_hidden)
  - 更新指定层的 memory。
  - 支持 detach()，避免梯度回传污染；
  - 若原有记忆不为空，则拼接当前 new_hidden 并截断至 mem_len；
  - 自动进行设备匹配（如切 GPU）。
- update_all(new_hiddens)
  - 批量更新所有层的记忆（例如每次 forward 后更新）。
  - 要求传入的列表长度等于 n_layers；
  - 内部调用 update 函数。
- get(layer_idx),返回指定层的记忆，用于当前推理拼接。
- clear(),清除所有层的记忆（可用于每段上下文/任务之间清零）。
- to(device),将当前缓存迁移至指定设备（通常在模型迁移时同步迁移）。
- memory_size(),返回各层当前保留的记忆长度（token 数），有助于调试。
- __repr__(),清晰展示当前内存使用状态，方便日志输出。

### TODO
1. MoE层优化：
- 分布式MoE，接入 DeepSpeed-MoE 或 FSDP 的 MoE 实现模块
- 路由器重参数优化，使用 GShard-style noisy-topk 或 Gumbel Softmax 提高探索能力
- 高效专家共享，多个 MoE 层复用共享专家池（如 M6、GLaM），可用专家池统一调度
- 提高 token 分派和执行效率	,替代逐专家收集方式，转向基于稀疏表示的并行处理
- 减少内存浪费,用稀疏张量或稀疏路由结构替代稠密 dispatch_mask
- 增强鲁棒性,对 token 溢出部分引入“残差路径”或再路由机制
- 改进负载均衡 Loss,引入 softmax entropy、expected load KL loss 等更合理的约束
- 增强模块清晰度和可维护性,拆解功能函数、清晰注释、命名标准化
- 动态专家激活数 k,训练过程中自动调整 k，更智能化调度
- 混合容量调度策略,高优先 token 分配更多容量
- 模块化解耦,分离 Gate, Router, Expert 更便于替换
2. 尝试嵌入MoE层优化模型
3. MeMory机制优化：
- 支持滑动窗口更新，设定窗口滑动的比例，实现更加平滑的记忆更替
- 混合KVCache，将KV缓存与hidden memory统一管理，为Attention服务
- 持久化保存，提供save()、load()，支持中断恢复

</details>

---

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出新功能建议！请遵循以下步骤：

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开一个 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 🌟 致谢

- 感谢所有为 Transformer 架构发展做出贡献的研究者
- 感谢 HuggingFace 团队提供的出色工具和生态系统
- 感谢所有项目贡献者

---

<div align="center">
  <sub>把每个字节都点亮成一盏灯，照见古今同望的夜空。</sub>
</div>