## 2025.7.13
### Done:
1. 实现BaseModelConfig类，后续的超参数将逐渐迭代
2. 实现RMSNorm层归一化类
3. Transformer经典的MultiHeadAttention类

### TODO：
1. Attention应用KV缓存，添加量化机制
2. 构建基础MLP层
3. 构建基础DecoderLayer层

## 2025.7.14
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
1. 构造基础的模型层ByteTransformer
2. 构造Memory类并进行应用
3. 应用LoRA类
4. 构造单步推理接口 def forward_step(self, x_t, past_k, past_v) -> (out, new_k, new_v)
5. Xpos位置编码优化
- 结构性正则
- 动态theta调整
6. Attention
- num_rep 未被使用 
- 实现 线程并行/All‑Reduce
- 进一步融合FlashAttention-2
- 接入RetNet
- 线性层量化quantize() 使用了 torch.quantization.quantize_dynamic()，但这仅限于线性层 + 推理，需要进一步优化以支持GPTQ/AWQ/SmoothQuant
7. KVCache
将 KVCache.append() 改为支持：
- 滑窗（sliding window）截断
- 写入位置并发锁定（if multi-thread）
- Layer-wise token位置自动偏移计算
