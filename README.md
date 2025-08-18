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
- 🔄 **Dynamic-RoPE 旋转位置编码**：改进的 RoPE 位置编码，提高长序列建模能力
- 🛠️ **丰富的优化技术**：
  - ⚙️ DeepNorm 归一化策略
  - 🔍 LayerScale 初始化技术
  - 🔀 DropPath 正则化
  - ⚡ 并行残差连接
- 📊 **参数高效微调**：内置 LoRA 低秩适应实现
- 🤗 **HuggingFace 兼容**：无缝集成 Transformers 生态系统
- 📝 **清晰的代码结构**：易于理解和扩展

## 📚 模型结构
> [模型架构](./model_info/model_structure.md)

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
from stellarbyte import ByteModel, ByteConfig

# 创建配置
config = ByteModelConfig(
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

### 从 HuggingFace 加载预训练模型(暂未实现)

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

### 使用 LoRA 进行参数高效微调(暂未时间)

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
|   CONTRIBUTING.md
|   datasets.py
|   INSTALL.md
|   LICENSE
|   model_pretrain.py
|   model_stf_train.py
|   README.md
|   requirements.txt
|   setup.py
|   tokenizer_pretrain.py
|
+---.pytest_cache
|   |   .gitignore
|   |   CACHEDIR.TAG
|   |   README.md
|   |
|   \---v
|       \---cache
|               lastfailed
|               nodeids
|               stepwise
|
+---checkpoints
+---configs
|       model_pretrain.yaml
|
+---datasets
|   |   data_preprocessor.py
|   |   train.jsonl
|   |   eval.jsonl
|   |
|   +---test
|   |       test_eval.jsonl
|   |       test_train.jsonl
|   |
|   \---tokenizers
|           code.jsonl
|           emoji.jsonl
|           en.jsonl
|           multi_lang.jsonl
|           zh.jsonl
|
+---logs
|
+---model
|   |   Attention.py
|   |   config.py
|   |   DecoderLayer.py
|   |   EmbeddingLayer.py
|   |   MLP.py
|   |   Model.py
|   |   MoELayer.py
|   |   MoERouter.py
|   |   Position_Embedding.py
|   |   RMSNorm.py
|   |   __init__.py
|   |
|   +---utils
|           DropPath.py
|           KVCache.py
|           LoRA.py
|           __init__.py
|        
|    
+---model_info
|   |   model_report_xxx.md
|   |   model_structure.md
|   |
|   \---plots
|
+---scripts
|       setup_env.bat
|       setup_env.py
|       setup_env.sh
|
+---sources
|   \---corpora
|           omw-1.4.zip
|           wordnet.zip
|
+---test
|       test_Attention.py
|       test_datasets.py
|       test_DeocoderLayer.py
|       test_KVCache.py
|       test_LoRA.py
|       test_MLP.py
|       test_MoERouter.py
|       test_Position_Embedding.py
|       test_RMSNorm.py
|    
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

</details>

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

<details>
<summary>2025.7.20</summary>

### DONE
1. 构建MoERouter路由系统
- 上下文感知的门控机制（Context-Aware Gating） -> 可支持长序列、多语言、大模型调度场景，对稀疏表示建模能力强。
  - 引入位置编码 + 历史状态 + 当前语义特征形成联合上下文表示（context_feat）。
  - 门控网络 (gate_mlp) 输入为 [x, context_feat]，可适应当前token与上下文的微妙变化。
  - RMSNorm 与 dropout 提升模型稳定性与泛化能力。
- 温度动态调整与专家优先级 -> 实现冷热专家动态激活、收敛更快、路由更稳定、token 分布更均衡。
  - 使用 self.temperature 根据专家负载差异动态调整 softmax 温度，控制专家选择的分布熵。
  - 专家优先级通过 self.expert_priority 和 self.expert_cold_priority 动态更新，对不活跃专家提供冷启动支持。
- 动态容量调控机制 -> 解决 token 分布极不均衡时的容量瓶颈问题，是大规模 MoE 模型部署的关键组件。
  - 使用专家利用率 (expert_utilization) 调整每轮 capacity，防止部分专家长期饱和或闲置。
  - 可设置最大/最小容量上下限，兼顾弹性与稳定性。
- 高效的向量化专家分发调度 -> 完全向量化实现，性能优于 for-loop 调度；适用于 TP/SP 并行环境下的调度计划生成。
  - _vectorized_dispatch 基于 torch_scatter + top-k 分发权重构建专家-样本映射表。
  - token 主导分发（由 gate 输出控制），专家主导筛选（基于分配优先级与容量限制）。
  - 分配逻辑明确：可用专家优先 + 权重越大优先。
- 溢出处理机制（Overflow Handling） -> 避免 token 丢失，保障模型鲁棒性，提高模型训练稳定性。
  - 溢出token 采用 备选专家机制（非 top-k 中分数最高的）、冷启动专家路由、历史专家粘性 fallback 和最终 随机 fallback。
  - fallback 倾向于选择负载轻 + 长期未活跃 + 学习权重高的专家。
- 多目标负载均衡损失（Aux Loss） -> 保证路由稳定与专家均匀负载，提升训练效率和专家泛化能力。
  - 通过 KL、MSE 和专家 load variance 实现路由熵约束，提升分布均匀性。
  - 引入 token entropy 权重机制，高不确定性 token 被更精细处理。
- 专家状态更新与自适应学习 -> 实现专家生命周期管理，支持在线专家剔除/替换、冷热专家切换等机制。
  - 维护专家分配状态（expert_load, utilization, priority）并通过 EMA 更新。
  - 统计信息支持动态调度决策与训练指标监控。
2. 一句MoERouter重构MoE层
- 专家并行（Expert Parallelism）支持：设计中明确区分了num_experts（全局专家数）和num_local_experts（当前设备上的专家数），并且通过dist分布式通信管理专家并行。
- 动态容量管理：通过router_config中max_capacity参数以及缓冲区expert_inputs和expert_outputs的动态分配，对专家输入容量的控制，避免了静态固定容量导致的内存浪费。
- 容错路由机制：使用fallback_expert来处理“溢出token”，防止因专家容量限制导致token丢失，增强鲁棒性。同时提供dropout机制，避免过拟合。
- 零浪费内存管理：预注册缓冲区避免动态内存分配，减少显存碎片和频繁分配开销，利于高效训练。
- 构建丰富性能监控指标：包括专家利用率、负载不均衡度、溢出率等，方便实时监控MoE层运行状况。
- 动态专家负载均衡（split/merge）策略：设计了根据利用率动态分裂过载专家、合并低载专家的机制，有利于训练期间专家资源自适应调整，提升模型效率。
- 设计合理专家模块：专家内部使用带门控的GLU结构（激活+门控乘积）及归一化，符合当前MoE专家的主流设计，计算效率和表达能力兼顾。
3. Memory机制优化：
- 分层记忆控制：底层可保留更长历史，高层可减少计算
- 智能batch处理：支持batch尺寸变化时的自动广播/裁剪
- 记忆融合：新旧记忆加权融合保留关键信息
- 策略配置：提供strict/select/repeat三种尺寸适配策略
4. 构建相关测试代码并修复bug
- 构建Memory测试代码并测试通过，修复了相关BUG
- 构建MoERouter测试代码并测试通过，修复了相关BUG

### TODO
1. MoE层优化：
- 构建MoE层预热机制
2. 训练脚本中加入
- update_cold_priority()，用于更新冷门专家优先级
```python
@torch.no_grad()
def update_cold_priority(self):
    # 利用当前专家利用率，低利用率专家冷启动优先级提升
    utilization = self.expert_utilization.clamp(0, 1)
    cold_priority = 1.0 + (1.0 - utilization)  # 低利用率加成范围[1,2]
    self.expert_cold_priority.copy_(cold_priority)
```
3. 构建test_MoERouter.py测试代码，并修复相关BUG
4. 构建test_MoE.py测试代码，并修复相关BUG
5. 尝试嵌入MoE层优化模型
6. MoERouter还存在如下问题待解决：
- 问题 1：专家冷启动权重的初始化过于统一，self.expert_cold_priority 默认全为 1，缺乏基于历史统计的初始化策略。
  - 建议：可引入冷启动历史时间戳或冷却时间窗口，动态更新 [1.0, 2.0] 分布。
- 问题 2：fallback 过程存在冲突风险，在 _handle_overflow_fallback 中，多个 token 可能争用同一专家，尤其在 batch 大时未完全并发安全。
  - 建议：可引入 dispatch_bitmap 或利用 index_put_ + 原子计数方案更安全更新。
- 问题 4：专家分配策略缺乏分布式感知，当前所有专家调度逻辑基于单节点信息，不考虑跨 GPU / TP experts 的分布。
  - 建议：引入跨设备 expert_rank，构建 local_vs_global_expert_mask，实现跨节点负载均衡。
- 问题 5：调度信息未显式支持多粒度 token 分配，当前所有分配基于 flat token，如果输入有 padding/attention mask，则可能错误分配。
  - 建议：引入 token_mask 机制，精确控制有效 token。
- 问题 6：负载统计状态未持久化/存盘，expert_priority, utilization 等状态参数在训练过程中变化大，但未持久化或复用。
  - 建议：加持久化接口（如save_state_dict / load_state_dict），支持热重启。

</details>

---

<details>
<summary>2025.7.21</summary>

### DONE
1. 将许可证从MIT更改为CC BY-NC 4.0,更新许可证文件以使用CC BY-NC 4.0协议，添加了非商业使用限制和署名要求
2. 优化MoERouter路由系统， 增强路由器的上下文感知能力和容错机制
- 新增max_positions参数控制位置编码范围
- 重构门控网络输入维度，加入位置和历史信息
- 改进溢出处理逻辑，增加动态备选专家选择和粘性回退机制
- 优化专家负载统计精度，使用float32类型
- 修复潜在的空溢出处理边界情况
3. 重构MoELayer，使用ByteMoERouter构造基础ByteMoELayer，暂不支持DDP

### TODO
1. ByteMoELayer升级优化，按照DeepSeed-MoE风格构造分布式ByteMoELayer，从单一的local是先到多卡分布式
2. 使用pytest构建ByteMoELayer，并修复相关BUG

</details>

---

<details>
<summary>2025.7.22</summary>

### DONE
1. 修复XPos位置编码中的bug,将_get_cos_sin_scale函数中的从RotaryCache初始化参数seq_len，从self.max_seq_len修改为seq_len
```
Traceback (most recent call last):
  File "/workspace/model_pretrain.py", line 686, in <module>
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
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/utils/checkpoint.py", line 1124, in unpack_hook
    frame.recompute_fn(*args)
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/utils/checkpoint.py", line 1518, in recompute_fn
    fn(*args, **kwargs)
  File "/workspace/model/Model.py", line 215, in custom_forward
    return layer(*inputs)
           ^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/model/DecoderLayer.py", line 102, in forward
    attn_out = self.self_attn(self.norm_attn(x), additive_mask)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/model/Attention.py", line 375, in forward
    q = q * cos + self.rotary._rotate_half(q) * sin
        ~~^~~~~
RuntimeError: The size of tensor a (2047) must match the size of tensor b (0) at non-singleton dimension 1
```
2. 修复model_pretrain.py中的设备不一致问题
**关键语句**
```python
gpu_mem       = torch.cuda.memory_allocated(args.device) if torch.cuda.is_available() else 0
```
**报错**
```
Traceback (most recent call last):
  File "/workspace/model_pretrain.py", line 686, in <module>
    train(args, logger)
  File "/workspace/model_pretrain.py", line 636, in train
    raise e
  File "/workspace/model_pretrain.py", line 597, in train
    train_epoch(
  File "/workspace/model_pretrain.py", line 429, in train_epoch
    gpu_mem       = torch.cuda.memory_allocated(args.device) if torch.cuda.is_available() else 0
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/cuda/memory.py", line 537, in memory_allocated
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/cuda/memory.py", line 323, in memory_stats
    stats = memory_stats_as_nested_dict(device=device)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/cuda/memory.py", line 334, in memory_stats_as_nested_dict
    device = _get_device_index(device, optional=True)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/cuda/_utils.py", line 34, in _get_device_index
    raise ValueError(f"Expected a cuda device, but got: {device}")
ValueError: Expected a cuda device, but got: cpu
```
**原因**
torch.cuda.memory_allocated() 只能接受 CUDA 设备（如 "cuda:0"），而在训练脚本中可能传入的是 "cpu"，这在 CPU-only 环境下或在逻辑中显式使用 CPU 设备时会出错。
**修复**
```python
gpu_mem       = torch.cuda.memory_allocated(args.device) if args.device=="cuda" and torch.cuda.is_available() else 0
```
3. XPos位置编码，移除XPosRotaryEmbedding中的offset参数，改为直接根据当前序列长度生成cos/sin/scale简化_get_cos_sin_scale和_compute_xpos_scale方法的实现，不再需要offset切片操作。

### TODO
在有限的资源情况下，尝试调试训练脚本，修复报错

### DEBUG
1. 启用grad_checkpoint，优化训练速度报错
```
raceback (most recent call last):
  File "/workspace/model_pretrain.py", line 686, in <module>
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
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/utils/checkpoint.py", line 1128, in unpack_hook
    frame.check_recomputed_tensors_match(gid)
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/utils/checkpoint.py", line 902, in check_recomputed_tensors_match
    raise CheckpointError(
torch.utils.checkpoint.CheckpointError: torch.utils.checkpoint: Recomputed values for the following tensors have different metadata than during the forward pass.
tensor at position 14:
saved metadata: {'shape': torch.Size([2047, 1]), 'dtype': torch.float32, 'device': device(type='cuda', index=0)}
recomputed metadata: {'shape': torch.Size([4094, 1]), 'dtype': torch.float32, 'device': device(type='cuda', index=0)}
tensor at position 15:
saved metadata: {'shape': torch.Size([2047, 24]), 'dtype': torch.float32, 'device': device(type='cuda', index=0)}
recomputed metadata: {'shape': torch.Size([4094, 24]), 'dtype': torch.float32, 'device': device(type='cuda', index=0)}
tensor at position 24:
saved metadata: {'shape': torch.Size([96, 48, 2047]), 'dtype': torch.bfloat16, 'device': device(type='cuda', index=0)}
recomputed metadata: {'shape': torch.Size([96, 48, 2048]), 'dtype': torch.bfloat16, 'device': device(type='cuda', index=0)}
tensor at position 26:
saved metadata: {'shape': torch.Size([6, 16, 2047, 2047]), 'dtype': torch.float32, 'device': device(type='cuda', index=0)}
recomputed metadata: {'shape': torch.Size([6, 16, 2047, 2048]), 'dtype': torch.float32, 'device': device(type='cuda', index=0)}
tensor at position 27:
saved metadata: {'shape': torch.Size([6, 16, 2047, 2047]), 'dtype': torch.bool, 'device': device(type='cuda', index=0)}
recomputed metadata: {'shape': torch.Size([6, 16, 2047, 2048]), 'dtype': torch.bool, 'device': device(type='cuda', index=0)}
tensor at position 28:
saved metadata: {'shape': torch.Size([96, 2047, 48]), 'dtype': torch.bfloat16, 'device': device(type='cuda', index=0)}
recomputed metadata: {'shape': torch.Size([96, 2048, 48]), 'dtype': torch.bfloat16, 'device': device(type='cuda', index=0)}
```
2. 启用torch_compile报错
```
ALLOW_TF32=True, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128, EVEN_K=True, GROUP_M=8, num_stages=3, num_warps=4
  triton_mm_1817 15.9234 ms 69.1% ACC_TYPE='tl.float32', ALLOW_TF32=True, BLOCK_K=16, BLOCK_M=64, BLOCK_N=64, EVEN_K=True, GROUP_M=8, num_stages=2, num_warps=4
SingleProcess AUTOTUNE benchmarking takes 10.1738 seconds and 0.0001 seconds precompiling for 20 choices
skipping cudagraphs due to skipping cudagraphs due to cpu device (primals_9). Found from : 
   File "/workspace/model/Model.py", line 223, in forward
    hidden_states = layer(hidden_states, additive_mask)
  File "/workspace/model/DecoderLayer.py", line 102, in forward
    attn_out = self.self_attn(self.norm_attn(x), additive_mask)
  File "/workspace/model/Attention.py", line 414, in forward
    scores = ( q.to(compute_dtype) @ k_cat.to(compute_dtype).transpose(-1,-2) ) * self.scale.to(compute_dtype)  # [B,H,T,Tk]
```
```
Traceback (most recent call last):
  File "/workspace/model_pretrain.py", line 686, in <module>
    train(args, logger)
  File "/workspace/model_pretrain.py", line 636, in train
    raise e
  File "/workspace/model_pretrain.py", line 597, in train
    train_epoch(
  File "/workspace/model_pretrain.py", line 392, in train_epoch
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py", line 655, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/model/Model.py", line 166, in forward
    def forward(
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py", line 838, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_functorch/aot_autograd.py", line 1201, in forward
    return compiled_fn(full_args)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 315, in runtime_wrapper
    all_outs = call_func_at_runtime_with_args(
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/utils.py", line 126, in call_func_at_runtime_with_args
    out = normalize_as_list(f(args))
                            ^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/utils.py", line 100, in g
    return f(*args)
           ^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/autograd/function.py", line 575, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 1937, in forward
    fw_outs = call_func_at_runtime_with_args(
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/utils.py", line 126, in call_func_at_runtime_with_args
    out = normalize_as_list(f(args))
                            ^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 495, in wrapper
    return compiled_fn(runtime_args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py", line 689, in inner_fn
    outs = compiled_fn(args)
           ^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_inductor/output_code.py", line 460, in __call__
    return self.current_callable(inputs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/_inductor/utils.py", line 2404, in run
    return model(new_inputs)
           ^^^^^^^^^^^^^^^^^
  File "/tmp/torchinductor_root/qo/cqo3uikhoqttxdvnkhrujkln45yocywss2s73k2di42jfnp5a2vc.py", line 4134, in call
    buf379 = empty_strided_cuda((12282, 768), (768, 1), torch.bfloat16)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 18.00 MiB. GPU 0 has a total capacity of 22.07 GiB of which 20.44 MiB is free. Process 2708630 has 22.04 GiB memory in use. Of the allocated memory 21.75 GiB is allocated by PyTorch, and 15.17 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```
3. 不启动torch_compile、不启动grad_checkpoint报错
```
Traceback (most recent call last):
  File "/workspace/model_pretrain.py", line 686, in <module>
    train(args, logger)
  File "/workspace/model_pretrain.py", line 636, in train
    raise e
  File "/workspace/model_pretrain.py", line 597, in train
    train_epoch(
  File "/workspace/model_pretrain.py", line 392, in train_epoch
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/model/Model.py", line 223, in forward
    hidden_states = layer(hidden_states, additive_mask)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/model/DecoderLayer.py", line 102, in forward
    attn_out = self.self_attn(self.norm_attn(x), additive_mask)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/model/Attention.py", line 421, in forward
    probs    = self.attn_dropout(probs).to(param_dtype)
               ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/dropout.py", line 70, in forward
    return F.dropout(input, self.p, self.training, self.inplace)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/functional.py", line 1425, in dropout
    _VF.dropout_(input, p, training) if inplace else _VF.dropout(input, p, training)
                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB. GPU 0 has a total capacity of 22.07 GiB of which 2.23 GiB is free. Process 3811892 has 19.83 GiB memory in use. Of the allocated memory 19.38 GiB is allocated by PyTorch, and 185.87 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. 
```
4. 启动flash_attention报错
```
Traceback (most recent call last):
  File "/workspace/model_pretrain.py", line 686, in <module>
    train(args, logger)
  File "/workspace/model_pretrain.py", line 636, in train
    raise e
  File "/workspace/model_pretrain.py", line 597, in train
    train_epoch(
  File "/workspace/model_pretrain.py", line 392, in train_epoch
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/model/Model.py", line 223, in forward
    hidden_states = layer(hidden_states, additive_mask)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/model/DecoderLayer.py", line 106, in forward
    ffn_out = self.mlp(self.norm_ffn(x))
              ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1751, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.pyenv/versions/3.11.1/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1762, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspace/model/MLP.py", line 78, in forward
    x_gate = torch.sigmoid(x_gate)  # [batch_size, seq_len, hidden_dim]
             ^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB. GPU 0 has a total capacity of 22.07 GiB of which 164.44 MiB is free. Process 3879328 has 21.90 GiB memory in use. Of the allocated memory 21.48 GiB is allocated by PyTorch, and 151.36 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management 
```

</details>

---

<details>
<summary>2025.8.6</summary>

### DONE
1. 位置编码替换：Dynamic-RoPE替换XPosEmbedding
2. 重构Attention 重构多头自注意力模块，优化张量并行和位置编码
- 移除KVCache相关代码，简化注意力模块结构
- 使用ByteDynamicRoPE替代XPosRotaryEmbedding实现动态位置编码
- 重构张量并行逻辑，改进权重初始化和掩码生成
- 优化代码结构，增强可读性和维护
3. Model.py简化模型结构并移除冗余代码
- 移除复杂的初始化逻辑和辅助方法
- 简化前向传播和生成逻辑
- 保留核心Transformer结构
- 优化代码组织结构提高可读性
4. DecoderLayer移除KVCache依赖并重命名mask参数
- 删除未使用的KVCache相关导入和参数
- 将additive_mask重命名为更明确的padding_mask
- 优化并行残差块的代码格式
- 更新测试输出格式以显示输入输出形状
5. 改进 ByteMLP 模块的代码结构和文档
- 重构 ByteMLP 模块的代码结构，使其更清晰易读
- 重新组织类文档字符串，明确模块功能和参数说明
- 优化前向传播过程的注释，分步骤解释门控机制
- 统一代码格式和命名规范
- 更新测试代码的注释说明
6. MLP层调整Dropout和残差连接的顺序以提升模型稳定性
修改了ByteMLP模块中Dropout和残差连接的执行顺序，先应用Dropout再进行残差连接。这种调整可以防止残差连接后的数值范围过大，有助于提升模型训练的稳定性。

### TODO
1. 重新构建tokenizer的训练
2. 主模型应用XPosRotoryEmbedding位置编码
3. 完善模型顶层基础设计forward()、generate()

</details>

---

<details>
<summary>2025.8.7</summary>

### DONE
1. 重构tokenizer训练脚本
- 对话模型配置
- 特殊tokens配置
- 角色标记配置
- BPE算法训练+bytelevel回退
2. 完善模型顶层设计forward函数和generate函数，并配置了自回归函数的辅助函数
- forward向前传播函数，将模型组件进行组装
- generate自回归生成函数
- sample_next_token采样函数，top-k采样，生成时增加top-p、temperature参数，允许在生成时进行top-k、top-p采样操作，避免生成长尾token或低概率的token
- repetition_penalty重复惩罚，在生成文本时，根据重复出现的token的频率对当前生成的token进行惩罚，从而避免生成文本出现低质量内容
3. 构建模型预训练脚本
- 构建了整体的主要函数
  - set_environment() 模型训练环境配置函数
  - cosine_annealing_lr() 学习率调度器函数
  - parse_args() 解析脚本参数函数
  - init_model() 初始化模型函数
  - eval() 模型评估函数
  - train_epoch() 单轮训练函数
  - train() 模型训练函数

### TODO
1. tokenizer解决问题 - 后处理器模板只定义了single和pair格式，缺少对多轮对话或更复杂场景的支持，解码器和后处理器的token替换符需保持同步。
2. 重构模型训练参数文件
3. 完善模型预训练脚本并进行测试

</details>

---

<details>
<summary>2025.8.8</summary>

### DONE
1. 优化数据集加载器
- 为BaseDataset添加BOS/EOS token自动注入逻辑
- 改进_pad_and_mask方法返回类型为torch.Tensor
- 在PretrainDataset中支持EOS token并优化token预留逻辑
- 添加数据集测试用例便于验证功能
2. 完善模型预训练脚本，优化代码结构和功能实现
- 重构参数解析函数，支持YAML配置文件读取和嵌套命名空间转换
- 优化环境设置函数，增强日志记录和设备选择逻辑
- 重写学习率调度器，改进余弦退火算法实现
- 完善模型初始化和评估函数，增加混合精度训练支持
- 重构训练循环，支持梯度累积和周期性验证

### TODO
1. 模型预训练脚本优化与完善
- 优化模型预训练脚本train_epoch中的损失函数与padding_mask设置与计算
- 完善模型预训练脚本train()模型预训练主体函数
2. 检查数据集加载器中attention_mask设计与模型主体中的padding_mask需求是否一致，如果不一致进行相对应的修复
3. 重构配置文件

</details>

---

<details>
<summary>2025.8.9</summary>

### DONE
1. PretrainDataset、SFTDatasset 数据集加载器修复
重构数据集处理逻辑并优化掩码生成
- 修复input_ids、labels、attention_mask、loss_mask生成逻辑
  - 输入序列：[BOS, T1, T2, T3, T4, T5, T6, T7, EOS]
  - 样本拆分：
    - X：[BOS, T1, T2, T3, T4, T5, T6, T7] → 模型输入上下文
    - Y：[T1, T2, T3, T4, T5, T6, T7, EOS] → 模型预测目标
  - 损失掩码：
    - 有效位置：[0, 1, 1, 1, 1, 1, 1, 1, 1] → 仅对T1-EOS计算损失
- 将特殊标记从硬编码改为通过构造函数传入，提高灵活性
- 重命名 `generate_loss_mask` 为 `_generate_loss_mask` 表示内部方法
- 优化掩码生成逻辑，使用切片操作替代循环
- 添加 attention_mask 到返回字典并统一使用 bool 类型
- 在 labels 中非 loss_mask 位置设置为 -100 避免计算损失
- 改进代码格式和注释清晰度
2. MultiHeadSelfAttention 主要修复多头自注意力机制中attention_mask的调整以及attention_mask与causal_mask融合的逻辑
- 优化注意力掩码生成逻辑并重命名参数
- 重构注意力掩码生成逻辑，使用更清晰的变量命名和条件判断。
- 将padding_mask参数重命名为attention_mask以更准确反映其用途，并改进掩码合并逻辑。
- 将最小掩码值从torch.finfo(dtype).min改为固定值-1e9以提高稳定性。
3. DecoderLayer层 调整修复，重命名padding_mask为attention_mask并优化并行残差路径
- 将padding_mask参数重命名为更通用的attention_mask以提升代码可读性
- 在并行残差路径中为FFN添加独立的归一化层，与顺序残差路径保持一致
4. Model模型主体 调整修复，将padding_mask重命名为attention_mask并简化损失计算
- 将参数名从padding_mask改为更通用的attention_mask以保持一致性
- 移除不必要的标签移位操作，直接使用原始logits和labels计算损失
5. model/config.py 移除未使用的tie_word_embeddings参数,清理模型配置中未使用的绑定词嵌入参数，简化配置逻辑
6. configs/pretrain_config.yaml 重构预训练配置文件结构并更新参数
- 重新组织配置文件结构，将相关配置分组更清晰
- 更新模型参数和训练配置以匹配最新需求
- 优化训练配置参数和日志设置
- 添加生成配置用于推理场景
- 移除冗余配置项，简化文件内容
- 调整参数命名以保持一致性
7. 重构日志模块，使用TimedRotatingFileHandler并改进彩色输出
- 移除colorlog依赖，改用colorama实现彩色日志
- 将RotatingFileHandler替换为TimedRotatingFileHandler以支持按时间轮转日志
- 新增日志级别字符串转换功能
- 简化日志配置接口，合并build_logger和get_logger功能
- 改进异常处理逻辑，移除全局异常处理器
8. 重构检查点管理器以支持原子保存和分布式训练
- 使用 Path 替代字符串路径处理
- 实现原子化保存机制防止损坏
- 支持分布式训练的主进程判断
- 改进最佳模型跟踪和旧检查点清理
- 添加信号处理用于紧急保存
- 增加类型注解和文档字符串
9. model_pretrain.py 完善训练脚本，增强模型训练和评估功能
- 构建整体的模型训练train()函数
- 添加NLTK评估指标(BLEU/ROUGE/METEOR)支持
- 重构日志系统使用setup_logger统一管理
- 改进学习率调度器支持动态最小学习率计算
- 优化训练过程增加梯度累积残余处理
- 完善评估函数增加文本生成质量评估
- 添加SwanLab实验跟踪集成

### TODO
1. 更新模型每个单元的测试脚本，对每个组件进行详细测试，并修复对应bug
2. 更新模型预训练相关的组件测试脚本，并修复对应bug
3. 测试模型预训练脚本，调试相关bug
4. 更新代码文档，提升代码可读性和可维护性

</details>

---

<details>
<summary>2025.8.10</summary>

### DONE
1. 完善依赖包相关配置
- 添加NLP相关依赖包,添加sentencepiece、nltk、rouge-score等NLP评估
- 添加tokenizer训练所需的依赖项,添加tokenizers、datasets
2. MultiHeadSelfAttention修复多头注意力计算中的数据类型不一致问题
- 将注意力权重转换为与值张量相同的数据类型，避免计算时出现类型不匹配错误
3. model/__init__ 模型导出文件,统一模块命名规范并添加ByteEmbedding导出
- 将主要类和模块重命名以统一使用"Byte"前缀，提高代码一致性
- 添加ByteEmbedding到导出列表以支持新的嵌入层功能
4. 更新并测试每个单元模块
- 更新Attention多头自注意力机制测试模块
  - 添加测试辅助函数和 fixture 以支持多种数据类型测试
  - 增加对重复KV、因果掩码、窗口注意力等辅助方法的测试
  - 添加前向传播的形状和稳定性测试
  - 移除过时的 KVCache 相关测试
- 重构BaseDatasets、PretrainDatasets、SFTDatasets数据集加载器测试模块，并增加更多测试用例
  - 重构测试文件结构，增加对BaseDataset、PretrainDataset和SFTDataset的测试覆盖
  - 添加对CSV格式数据的支持测试
  - 完善tokenizer处理逻辑的测试
  - 增加对模板格式化和缺失字段的测试
- 更新ByteMLP测试
  - 更新所有测试用例以使用新的ByteMLP类，验证门控机制和归一化层的正确性。
  - 移除对旧MLP类的引用，并添加对新功能的测试覆盖。
- 重构Position_Embedding测试文件并添加新测试用例
  - 将测试类从XPosRotaryEmbedding改为ByteDynamicRoPE
  - 新增测试用例验证初始化、基础频率计算、NTK缩放因子行为等
  - 添加旋转形状和数值正确性测试
  - 包含缓存重建和范数保持的验证
- 更新层归一化测试用例以使用ByteRMSNorm替代RMSNorm
5. 修复多头注意力计算中的数据类型不一致问题
- 将注意力权重转换为与值张量相同的数据类型，避免计算错误
6. 修复SFT数据集加载器
- 优化特殊标记处理并改进掩码计算逻辑
- 将 start_tokens 和 end_tokens 处理逻辑统一为支持字符串或ID列表
- 改进掩码计算方式，使用张量操作替代列表操作提高效率
- 修复标签处理中的类型不一致问题
- 更新测试用例以验证修改后的功能
7. 重构Byte-Transformer模型预训练配置文件
- 添加统一的预训练/继续训练配置文件，包含实验、数据、模型架构、训练、生成和日志等模块的配置参数
- 让配置文件结构更清晰，更具有可读性、可维护性

### TODO
1. 优化tokenizer训练
- 支持中文编码
- 更新对话模板
- 解决token_end和pad使用同样的标记错误问题
- 提升训练速度
- 尝试进行tokenizer进行训练
2. 测试并修复模型预训练脚本，给出最终的模型预训练脚本
- 支持使用命令行 + configs/model_pretrain.yaml 对训练参数进行配置，并直接进行模型训练

</details>

---

<details>
<summary>2025.8.11</summary>

### DONE
1. 在.gitignore中添加sources/目录
2. configs/model_pretrain.yaml 修正配置文件中参数错误并更新数据路径
- 修复use_swanlab拼写错误
- 更新训练和验证数据路径
- 移除tie_word_embeddings参数并添加device配置
3. ByteEmbedding 优化嵌入层参数命名并添加测试用例
重构嵌入层参数命名以提高可读性，并添加测试用例验证功能正确性
4. model_pretrain.py 重构配置解析和设备处理逻辑
- 将嵌套配置展平为一级结构，简化参数访问
- 新增设备解析函数，支持多GPU配置
- 优化SwanLab初始化逻辑，增加配置检查
- 统一训练参数访问方式，移除嵌套结构
- 添加命令行参数支持，提升脚本可用性

### TODO
1. 模型预训练脚本中修复参数读取相关的BUG
2. 测试模型预训练脚本，修复完善相关功能。

### DEBUG
1. 解决启动模型预训练脚本时报错：
```
Traceback (most recent call last):
  File "d:\Objects\StellarByte\model_pretrain.py", line 845, in <module>        
    train(args.config)
  File "d:\Objects\StellarByte\model_pretrain.py", line 700, in train
    model, tokenizer = init_model(config, device)
  File "d:\Objects\StellarByte\model_pretrain.py", line 269, in init_model      
    model = ByteModel(model_config)
  File "d:\Objects\StellarByte\model\Model.py", line 34, in __init__
    self.token_embedding = ByteEmbedding(args)
  File "d:\Objects\StellarByte\model\EmbeddingLayer.py", line 43, in __init__   
    self.embed_tokens  = nn.Embedding(
  File "D:\Develop_Tools\Anconda3\envs\LLM\lib\site-packages\torch\nn\modules\sparse.py", line 167, in __init__
    torch.empty((num_embeddings, embedding_dim), **factory_kwargs),
TypeError: empty() received an invalid combination of arguments - got (tuple, dtype=NoneType, device=NoneType), but expected one of:
 * (tuple of ints size, *, tuple of names names, torch.memory_format memory_format = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)
 * (tuple of ints size, *, torch.memory_format memory_format = None, Tensor out = None, torch.dtype dtype = None, torch.layout layout = None, torch.device device = None, bool pin_memory = False, bool requires_grad = False)
```
ByteEMbedding 中提示信息：
```
vocab_size: namespace(use_swanlab='trua', project_name='ByteLM-Pretrain', run_name='baseline-158M', mode='cloud', api_key='', tokenizer_path='./tokenizer', train_data='./data/test/test_train.jsonl', eval_data='./data/test_eval.jsonl', vocab_size=32768, model_dim=768, num_layers=12, max_seq_len=2048, layer_norm_eps='1e-5', initializer_range=0.02, layerscale_init='1e-5', parallel_residual=True, num_heads=16, num_kv_heads=8, use_flash_attention=False, attention_window_size=0, attention_dropout_prob=0.1, base_theta=10000.0, ntk_alpha=1.0, use_cache=True, key_cache_dtype='float16', value_cache_dtype='float16', hidden_dim=3072, dim_multiplier=4, hidden_dropout_prob=0.1, residual_dropout_prob=0.1, drop_path_prob=0.0, tensor_parallel_size=1, train_epochs=10, batch_size=32, learning_rate='3e-4', min_lr_ratio=0.1, weight_decay=0.1, beta1=0.9, beta2=0.98, warmup_ratio=0.02, plateau_ratio=0.01, gradient_accumulation_steps=4, max_grad_norm=1.0, num_workers=8, use_cuda=True, device='cpu', mixed_precision=True, output_dir='./checkpoints', save_epochs=1, save_interval=1000, log_interval=50, eval_steps=500, eval_batch_size=16, temperature=1.0, top_k=50, top_p=0.9, repetition_penalty=1.2, repetition_context=512, logger_name='StellarByte', log_dir='logs', log_file='StellarByte_pretrain.log', log_level='DEBUG', console_level='INFO', file_level='DEBUG', use_color=True, rotation='midnight', backup_count=7, is_rank_0=True) (<class 'types.SimpleNamespace'>)
model_dim: 768 (<class 'int'>)
tp_size: 1 (<class 'int'>)
embed_dim_per_partition: 768 (<class 'int'>)
```

</details>

---

<details>
<summary>20025.8.12</summary>

### MileStone
**实现基础的Byte-Transformer模型并成功验证了单卡训练**

### DONE
1. model_pretrain.py 优化配置处理和数据集初始化,修复训练损失计算和nltk资源下载问题,添加检查点管理功能以支持模型恢复和最佳模型保存
- 添加环境变量检查避免NLTK资源重复下载
- 改进配置参数转换逻辑，支持自动类型推断
- 重构模型配置初始化，显式提取所需参数，解决启动模型预训练脚本时报错
- 简化数据集初始化参数，统一使用data_path
- 修复训练过程中损失计算不准确的问题，调整梯度累积逻辑以正确统计token级损失
- 添加nltk punkt分词器资源下载，改进分词方式从空格切分到nltk分词
- 修复meteor_score计算时参数传递错误的问题
- 引入CheckpointManager类实现模型检查点管理
- 支持训练中断恢复和最佳模型自动保存
- 添加多进程安全检查和信号处理
- 改进训练日志记录和评估指标跟踪
- 修改logger名称以提高可识别性
- 将SimpleNamespace配置转换为字典以便SwanLab正确记录配置参数
2. configs/model_pretrain.yaml 修正训练和验证数据路径配置错误
- 将训练数据和验证数据的路径从"./data"更正为"./datasets"，以匹配实际项目目录结构
- 完善实验设置部分的参数配置
- 添加实验部分使用说明注释
- 将检查点相关配置从 training 部分提取到独立的 checkpoints 部分，并增加更多控制选项如最大保存数量、监控指标等。同时调整了日志保存的层级结构以提高配置文件的可读性和可维护性。
3. Model.py 统一使用num_layers代替n_layers参数名
- 将模型初始化中的参数名从n_layers统一改为num_layers以保持命名一致性，并更新相关初始化逻辑
4. model/utils/DropPath.py 增强DropPath模块功能并添加衰减计划
- 添加线性/余弦衰减计划支持，实现DDP同步掩码功能
- 支持自动混合精度和任意输入维度
- 添加详细文档说明和测试用例
5. model/utils/KVCache.py 重构工业级KV缓存系统用于Transformer模型，支持高效键值缓存管理，包括：
- 预分配固定形状缓冲区
- 支持批量追加和块写入操作
- 提供束搜索重排序和剪枝功能
- 实现状态保存/加载和分布式分片
- 支持设备管理和内存优化
6. 测试cpu训练、单卡训练成功

### TODO
1. 测试多卡训练
2. 为模型添加KVCache缓存支持，构建脚本并进行测试修复对应BUG
3. 构建检查点管理类，统一管理和加载多个实验的检查点，构建实验脚本进行测试并修复对应bug
4. 优化数据集加载器，不要一次性全部加载，分批次加载，减少内存占用
5. 优化模型预训练，添加早停机制，节省训练时间。

</details>

---

<details>
<summary>2025.8.13 - 2028.8.14</summary>

### MileStone
**实现带KVCache 的Byte-Transformer模型。**

### DONE
1. utils/checkpoint.py 添加checkpoints功能类，提供检查点保存、加载、查找等功能
2. model_pretrain.py 添加早停机制并优化检查点管理
- 实现早停机制函数early_stopping，用于监控验证指标并在性能不再提升时终止训练
- 修复检查点目录配置错误，统一使用config.checkpoints_dir
- 优化检查点保存逻辑，使用配置中的checkpoints_prefix作为前缀
- 调整最佳模型保存策略，使用config.checkpoints_monitor作为监控指标
- 统一缓存参数配置并简化数据类型设置
  - 将key_cache_dtype和value_cache_dtype合并为cache_dtype参数
  - use_cache配置项改为使用config.use_kvcache
3. configs/model_pretrain.yaml 添加早停机制并更新检查点配置
- 添加早停机制相关参数配置，包括监控指标、模式和最小提升幅度
- 更新检查点配置中的监控指标和模式为 loss 和 min
4. datasets.py 数据集加载器新增流式数据集类支持大文件处理
- 添加 StreamingPretrainDataset 和 StreamingSFTDataset 类，支持流式加载 json/jsonl/csv 格式数据，避免内存溢出问题。实现 IterableDataset 接口，逐条读取和处理数据，适用于大规模预训练和微调场景。
5. MultiHeadSelfAttention 实现KV缓存支持及增量推理功能
添加KV缓存功能以支持自回归生成，包括：
- 引入ByteKVCache类管理缓存状态
- 修改注意力掩码生成逻辑以支持缓存偏移
- 实现缓存初始化、更新和状态管理接口
- 支持增量推理时的位置偏移计算
- 添加缓存相关测试用例
6. DecoderLayer 添加对ByteKVCache的支持并更新前向传播
- 为DecoderLayer添加ByteKVCache参数支持，修改前向传播逻辑以处理KV缓存
7. Model 添加KV缓存支持以提高推理性能
- 引入ByteKVCache类来缓存键值对，避免在自回归生成过程中重复计算
- 修改forward方法以支持缓存传递
- 在generate方法中初始化并使用缓存来优化长序列生成性能。
8. KVCache 添加逻辑处理空序列时返回空张量
- 当序列长度为0时返回空张量以避免潜在错误
9. model/config.py 统一KV缓存相关参数命名
- 将`use_cache`重命名为`use_kvcache`以保持命名一致性
- 合并`key_cache_dtype`和`value_cache_dtype`为`cache_dtype`
10. configs/model_pretrain.yaml 移除模型配置中的KV缓存参数并重构为独立模块
- 将KV缓存相关配置从模型主配置中移除，并重构为独立的kv_cache模块，提高配置的可读性和模块化程度
11. MultiHeadSelfAttention 完善多头注意力模块的文档和注释
- 补充类和方法文档字符串，详细说明参数和返回值
- 添加关键步骤的代码注释，提高可读性
- 更新类文档以反映新增功能特性
12. MLP 优化MLP模块结构并改进文档说明
- 将w13拆分为独立的w1和w3线性层，提高代码可读性
- 重新组织前向传播步骤编号，使逻辑更清晰
- 完善模块文档字符串，补充核心特点和参数说明
13. RMSNorm 优化文档注释和代码结构
- 重新组织函数和类的文档字符串，使其更清晰简洁
- 统一参数和返回值的描述格式
- 移除冗余注释，保留核心功能说明
14. EmbeddingLayer 完善分布式词嵌入层的文档和注释
- 张量并行设计原理
- 参数和返回值的详细描述
- 关键实现细节的解释
- 权重共享接口的使用场景
15. DecoderLayer 重构解码器层实现并改进文档
- 重新组织类文档字符串，更清晰地说明实现特性和参数
- 为关键变量添加注释说明其作用
- 优化前向传播逻辑的代码结构，区分并行和顺序模式
- 改进变量命名和代码格式，增强可读性
16. Model 重构模型代码并添加详细文档注释
- 添加类和方法级别的详细文档字符串，说明架构和功能
- 重新组织代码结构，增加模块分隔注释
- 优化权重初始化逻辑，添加缩放因子说明
- 改进前向传播和生成方法的实现细节
- 增强采样方法的可读性和注释
17. DropPath 更新完善DropPath模块的文档字符串，补充功能特性详细说明和参数注释
18. KVCache 完善类和方法文档字符串，增加详细说明和异常描述
- 类功能特性的详细说明
- 方法参数和返回值的完整描述
- 可能抛出的异常类型及触发条件
- 内部实现细节的补充说明
- 分布式操作和状态管理的文档完善
19. tokenizer_pretrain.py 优化tokenizer配置和预处理逻辑
- 重新组织tokenizer配置项顺序，将tokenizer_class移至顶部
- 移除冗余的sep_token，简化特殊token列表
- 改进文本规范化逻辑，统一数字处理为[NUMBER]
- 优化预分词器规则，专注于代码符号和大小写处理
- 调整特殊token的ID映射以保持一致性

### TODO
1. 尝试进行模型训练，并寻找BUG
2. 优化MLP层，尝试加入融合推理
3. 测试KVCache
4. 分析并优化显存占用
5. 寻找并构建tokenizer训练数据集
  - 中文语料
  - 英文语料
  - Emoji语料
  - Code语料
6. 寻找并构建模型训练数据集
  - 文本生成
  - 代码生成/代码理解
  - 逻辑推理/问答/常识
  - 多轮对话

</details>

---

<details>
<summary>2025.8.15</summary>

### DONE
1. 移除MoERouter.py文件及其相关实现
2. MoELayer 实现分布式优化的MoE层并支持all_to_all通信
- 重构MoE层为分布式优化版本，支持多GPU专家并行计算
- 使用向量化top-k路由和容量控制机制
- 实现基于all_to_all的分布式token交换
- 保留负载均衡loss以优化专家利用率

### TODO
1. 测试 MoELayer 并修复对应BUG
2. 尝试使用 MoELayer 替换 MLP
3. 优化 MoELayer
4. 使用流式数据集加载器替换全量数据集加载器进行模型训练，进行测试，并修复对应BUG

</details>

---

<details>
<summary>2025.8.16</summary>

### DONE
1. model_pretrain.py 添加数据集流式加载支持
- 为大规模数据集训练添加流式加载功能，通过配置use_streaming开关控制加载方式。当启用流式加载时，使用StreamingPretrainDataset和islice进行分批处理，避免内存不足问题并支持更大规模数据训练。同时调整了相关训练逻辑和参数计算以适应流式加载模式。
2. configs/model_pretrain.yaml 更新模型预训练配置文件，添加流式训练相关参数
- 添加流式训练相关配置参数，包括use_streaming、steps_per_epoch等
- 调整batch_size参数位置至数据集加载器部分
- 补充梯度累积步数的有效batch_size计算公式
3. datasets 数据集加载器添加分布式训练支持并添加流式数据加载逻辑
- 在BaseDataset中提取_format_sample方法避免代码重复
- 为StreamingPretrainDataset和StreamingSFTDataset添加分布式训练支持
- 将数据编码逻辑重构为_encode_one方法提高可维护性

### TODO
1. 重构MoELayer，添加流式token分发和专家并行支持，使用all_to_all通信模式，支持高效token分发和计算
- 支持多GPU专家并行计算
- 实现分布式top-k路由和专家负载均衡
- 保持显存和gpu占用稳定。
2. 重构MoELayer测试模块，并修复对应BUG。

</details>

---

<details>
<summary>2025.8.18</summary>

### DONE
1. ByteMoELayer 重构MoE层实现，增加专家并行支持并优化通信效率：
- 实现双all_to_all通信模式，支持高效分布式token分发与结果聚合
- 改进路由算法，支持top-1/top-2路由及容量裁剪
- 增强专家模块功能，支持残差连接和LayerNorm
- 优化负载均衡损失计算，提高专家利用率
- 完善单卡/多卡兼容性处理

### TODO
1. 重构ByteMoELayer测试模块，并修复对应BUG
2. 尝试应用ByteMoELayer替代MLP

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

本项目采用 CC BY-NC 4.0 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 📖 引用

如果您在研究或项目中使用了本仓库，请按以下方式引用：

```bibtex
@misc{StellarByte,
  author       = {Yao Xiang Zhang},
  title        = {StellarByte},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/HxCodeWarrior/StellarByte}}
}
```

---

## 🌟 致谢

- 感谢所有为 Transformer 架构发展做出贡献的研究者
- 感谢 HuggingFace 团队提供的出色工具和生态系统
- 感谢所有项目贡献者

---

<div align="center">
  <sub>把每个字节都点亮成一盏灯，照见古今同望的夜空。</sub>
</div>