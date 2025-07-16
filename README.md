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