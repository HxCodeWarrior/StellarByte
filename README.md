<div align="center">

# ✨ StellarByte ✨

<p>把每个字节都点亮成一盏灯，照见古今同望的夜空。</p>

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow?style=flat-square)](https://huggingface.co/)
[![Blog](https://img.shields.io/badge/Blog-ByteWyrm=flat-square)](https://blog.devnest.top/)

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

## 🔧 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/StellarByte.git
cd StellarByte

# 安装依赖
pip install -r requirements.txt

# 安装开发版本（暂未实现）
pip install -e .
```

## 🚀 快速开始

```python
import torch
from stellarbyte import ByteModel, ByteConfig

# 创建配置
config = ByteConfig(
    vocab_size=32000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072
)

# 初始化模型
model = ByteModel(config)

# 准备输入
inputs = torch.randint(0, 32000, (1, 512))

# 前向传播
outputs = model(inputs)
```

## 📋 使用示例

### 从 HuggingFace 加载预训练模型

```python
from stellarbyte import ByteModel
from transformers import AutoTokenizer

# 加载模型和分词器
model = ByteModel.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/tokenizer")

# 编码文本
inputs = tokenizer("把每个字节都点亮成一盏灯", return_tensors="pt")

# 生成文本
outputs = model.generate(inputs.input_ids, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### 使用 LoRA 进行参数高效微调

```python
from stellarbyte import ByteModel, LoRAConfig
from stellarbyte.lora import apply_lora_to_model

# 加载基础模型
model = ByteModel.from_pretrained("path/to/model")

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
├── config/             # 配置类
├── datasets/           # 数据集处理
├── model/              # 模型组件
│   ├── Attention.py    # 多头自注意力实现
│   ├── DecoderLayer.py # Transformer 解码器层
│   ├── LoRA.py         # 低秩适应实现
│   ├── MLP.py          # 多层感知机实现
│   ├── MoE.py          # 专家混合实现（计划中）
│   ├── Position_Embedding.py # 位置编码实现
│   └── RMSNorm.py      # RMS 归一化实现
├── tokenizer/          # 分词器
├── utils/              # 工具函数
└── test/               # 测试代码
```

## 🔜 开发计划

### 2025.7.13
#### Done:
1. 实现BaseModelConfig类，后续的超参数将逐渐迭代
2. 实现RMSNorm层归一化类
3. Transformer经典的MultiHeadAttention类

#### TODO：
1. Attention应用KV缓存，添加量化机制
2. 构建基础MLP层
3. 构建基础DecoderLayer层

### 2025.7.14
#### Done:
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

#### TODO:
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
- 接入RetNet
- 线性层量化quantize() 使用了 torch.quantization.quantize_dynamic()，但这仅限于线性层 + 推理，需要进一步优化以支持GPTQ/AWQ/SmoothQuant
6. KVCache
将 KVCache.append() 改为支持：
- 滑窗（sliding window）截断
- 写入位置并发锁定（if multi-thread）
- Layer-wise token位置自动偏移计算

### 2025.7.15
#### Done:
1. 构建并优化模型训练组件：
- 检查点管理组件
- 意外中断保护组件
2. 在模型训练过程中应用检查点管理以及意外中断保护组件

#### TODO：
1. 


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