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

### TODO:
1. Attention
- num_rep 未被使用 
-  线程并行/All‑Reduce 未实现
2. 构造基础的模型层ByteTransformer
3. 构造Memory类并进行应用
4. 应用LoRA类

### DeBUG
1. Model测试中报错【KV head_dim 不一致：12 vs 6】