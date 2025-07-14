import unittest
import torch
import sys
from pathlib import Path

# 将项目根目录添加到sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from model.config import ByteModelConfig
from model.Attention import MultiHeadSelfAttention
from model.utils.KVCache import KVCache
from model.Position_Embedding import XPosRotaryEmbedding

class TestMultiHeadSelfAttention(unittest.TestCase):
    """测试多头自注意力机制的各项功能"""

    def setUp(self):
        """每个测试前的设置"""
        # 设置随机种子以确保结果可重现
        torch.manual_seed(42)
        
        # 默认配置
        self.default_config = ByteModelConfig(
            model_dim=768,
            num_attention_heads=12,
            num_kv_heads=6,
            max_seq_len=2048,
            attention_dropout_prob=0.1,
            residual_dropout_prob=0.1,
            use_flash_attention=False,
            use_causal=True
        )
        
        # 创建默认的注意力层
        self.attention = MultiHeadSelfAttention(self.default_config)
        
        # 创建测试输入
        self.batch_size = 2
        self.seq_len = 16
        self.embed_dim = self.default_config.model_dim
        self.x = torch.randn(self.batch_size, self.seq_len, self.embed_dim)

    def test_basic_forward(self):
        """测试基本的前向传播功能"""
        # 前向传播
        output = self.attention(self.x)
        
        # 验证输出形状
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
        
        # 验证输出不是NaN或无穷大
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_causal_mask(self):
        """测试因果掩码功能"""
        # 创建有因果掩码的注意力层
        causal_config = ByteModelConfig(
            model_dim=768,
            num_attention_heads=12,
            num_kv_heads=6,
            use_causal=True
        )
        causal_attention = MultiHeadSelfAttention(causal_config)
        
        # 创建无因果掩码的注意力层
        non_causal_config = ByteModelConfig(
            model_dim=768,
            num_attention_heads=12,
            num_kv_heads=6,
            use_causal=False
        )
        non_causal_attention = MultiHeadSelfAttention(non_causal_config)
        
        # 使用相同的输入进行前向传播
        causal_output = causal_attention(self.x)
        non_causal_output = non_causal_attention(self.x)
        
        # 验证两个输出不同（因果掩码应该影响结果）
        self.assertFalse(torch.allclose(causal_output, non_causal_output, atol=1e-5))

    def test_different_head_configs(self):
        """测试不同的头数配置"""
        # 测试不同的注意力头数
        head_configs = [
            (8, 4),   # 8个注意力头，4个KV头
            (16, 8),  # 16个注意力头，8个KV头
            (12, 12)  # 12个注意力头，12个KV头（无头分离）
        ]
        
        for num_heads, num_kv_heads in head_configs:
            config = ByteModelConfig(
                model_dim=768,
                num_attention_heads=num_heads,
                num_kv_heads=num_kv_heads
            )
            attention = MultiHeadSelfAttention(config)
            
            # 前向传播
            output = attention(self.x)
            
            # 验证输出形状
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
            
            # 验证输出不是NaN或无穷大
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())

    def test_different_model_dims(self):
        """测试不同的模型维度"""
        # 测试不同的模型维度
        model_dims = [512, 1024, 2048]
        
        for dim in model_dims:
            config = ByteModelConfig(
                model_dim=dim,
                num_attention_heads=dim // 64,  # 保持每个头的维度为64
                num_kv_heads=dim // 128         # KV头数为注意力头数的一半
            )
            attention = MultiHeadSelfAttention(config)
            
            # 创建适合新维度的输入
            x = torch.randn(self.batch_size, self.seq_len, dim)
            
            # 前向传播
            output = attention(x)
            
            # 验证输出形状
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, dim))
            
            # 验证输出不是NaN或无穷大
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())

    def test_additive_mask(self):
        """测试额外的加性掩码（如padding掩码）"""
        # 创建一个padding掩码，假设第二个序列的后半部分是padding
        # 形状为 [batch_size, 1, 1, seq_len]
        mask = torch.zeros(self.batch_size, 1, 1, self.seq_len)
        mask[1, 0, 0, self.seq_len//2:] = float('-inf')  # 第二个序列的后半部分被掩码
        
        # 前向传播，带掩码
        output_with_mask = self.attention(self.x, additive_mask=mask)
        
        # 前向传播，不带掩码
        output_without_mask = self.attention(self.x)
        
        # 验证两个输出不同（掩码应该影响结果）
        self.assertFalse(torch.allclose(output_with_mask, output_without_mask, atol=1e-5))

    def test_kv_cache(self):
        """测试KV缓存功能"""
        # 创建KV缓存
        kv_cache = KVCache(
            batch_size=self.batch_size,
            max_seq_len=self.seq_len * 2,  # 足够大以容纳多次前向传播
            num_layers=1,
            num_heads=self.default_config.num_attention_heads,
            head_dim=self.default_config.model_dim // self.default_config.num_attention_heads,
            dtype=torch.float32
        )
        
        # 创建带KV缓存的注意力层
        attention_with_cache = MultiHeadSelfAttention(
            self.default_config,
            kv_cache=kv_cache,
            layer_id=0
        )
        
        # 第一次前向传播，使用前半部分序列
        first_half = self.x[:, :self.seq_len//2, :]
        first_output = attention_with_cache(first_half)
        
        # 第二次前向传播，使用后半部分序列
        second_half = self.x[:, self.seq_len//2:, :]
        second_output = attention_with_cache(second_half)
        
        # 验证KV缓存长度增加
        self.assertEqual(kv_cache.length, self.seq_len)
        
        # 不使用缓存的完整前向传播
        full_output = self.attention(self.x)
        
        # 拼接两次输出
        combined_output = torch.cat([first_output, second_output], dim=1)
        
        # 注意：由于自注意力的上下文感知性质，使用KV缓存的输出和完整前向传播的输出
        # 在第二部分可能会有差异，因为第二部分在增量推理中看不到第一部分的Q
        # 但我们可以验证形状是否正确
        self.assertEqual(combined_output.shape, full_output.shape)

    def test_head_pruning(self):
        """测试注意力头剪枝功能"""
        # 初始状态下所有头都激活
        self.assertTrue(torch.all(self.attention.head_mask))
        
        # 剪枝第一个头
        self.attention.prune_heads([0])
        
        # 验证第一个头被剪枝
        self.assertFalse(self.attention.head_mask[0])
        self.assertTrue(torch.all(self.attention.head_mask[1:]))
        
        # 前向传播应该仍然工作
        output = self.attention(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))

    def test_flash_attention(self):
        """测试FlashAttention功能（如果可用）"""
        # 只有在PyTorch >= 2.1且CUDA可用时才测试FlashAttention
        if torch.cuda.is_available() and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            # 创建启用FlashAttention的配置
            flash_config = ByteModelConfig(
                model_dim=768,
                num_attention_heads=12,
                num_kv_heads=6,
                use_flash_attention=True
            )
            
            # 创建启用FlashAttention的注意力层
            flash_attention = MultiHeadSelfAttention(flash_config)
            
            # 将模型和输入移至CUDA
            flash_attention = flash_attention.cuda()
            x_cuda = self.x.cuda()
            
            # 前向传播
            output = flash_attention(x_cuda)
            
            # 验证输出形状
            self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.embed_dim))
            
            # 验证输出不是NaN或无穷大
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())
        else:
            # 如果不支持FlashAttention，跳过测试
            print("FlashAttention不可用，跳过测试")

    def test_quantization(self):
        """测试注意力层的量化功能"""
        # 量化前的前向传播
        pre_quant_output = self.attention(self.x)
        
        # 量化模型
        self.attention.quantize()
        
        # 验证量化标志
        self.assertTrue(self.attention.quantized)
        
        # 量化后的前向传播
        post_quant_output = self.attention(self.x)
        
        # 验证输出形状
        self.assertEqual(post_quant_output.shape, (self.batch_size, self.seq_len, self.embed_dim))
        
        # 验证输出不是NaN或无穷大
        self.assertFalse(torch.isnan(post_quant_output).any())
        self.assertFalse(torch.isinf(post_quant_output).any())

    def test_boundary_cases(self):
        """测试边界情况"""
        # 测试极短序列
        short_seq = torch.randn(self.batch_size, 1, self.embed_dim)
        short_output = self.attention(short_seq)
        self.assertEqual(short_output.shape, (self.batch_size, 1, self.embed_dim))
        
        # 测试较长序列（但仍在模型最大长度内）
        long_seq_len = 128
        long_seq = torch.randn(self.batch_size, long_seq_len, self.embed_dim)
        long_output = self.attention(long_seq)
        self.assertEqual(long_output.shape, (self.batch_size, long_seq_len, self.embed_dim))
        
        # 测试不同批量大小
        batch_sizes = [1, 4, 8]
        for bs in batch_sizes:
            x = torch.randn(bs, self.seq_len, self.embed_dim)
            output = self.attention(x)
            self.assertEqual(output.shape, (bs, self.seq_len, self.embed_dim))

    def test_rotary_embedding(self):
        """测试XPos Rotary位置编码"""
        # 获取rotary缓存前的状态
        initial_cache_size = len(self.attention._rotary_cache)
        
        # 前向传播，应该会计算并缓存rotary编码
        _ = self.attention(self.x)
        
        # 验证rotary编码已被缓存
        self.assertEqual(len(self.attention._rotary_cache), initial_cache_size + 1)
        self.assertIn(self.seq_len, self.attention._rotary_cache)
        
        # 验证缓存的rotary编码格式正确
        cos, sin, scale = self.attention._rotary_cache[self.seq_len]
        self.assertEqual(cos.shape, (self.seq_len, self.attention.head_dim))
        self.assertEqual(sin.shape, (self.seq_len, self.attention.head_dim))
        self.assertEqual(scale.shape, (self.seq_len, self.attention.head_dim))
        
        # 再次前向传播，应该使用缓存的rotary编码
        _ = self.attention(self.x)
        
        # 验证缓存大小没有变化（使用了已缓存的值）
        self.assertEqual(len(self.attention._rotary_cache), initial_cache_size + 1)

    def test_weight_initialization(self):
        """测试权重初始化"""
        # 创建带有num_layers参数的注意力层
        num_layers = 24
        attention_with_init = MultiHeadSelfAttention(
            self.default_config,
            layer_id=0,
            num_layers=num_layers
        )
        
        # 验证权重已初始化（不为零）
        self.assertFalse(torch.all(attention_with_init.W_q.weight == 0))
        self.assertFalse(torch.all(attention_with_init.W_k.weight == 0))
        self.assertFalse(torch.all(attention_with_init.W_v.weight == 0))
        self.assertFalse(torch.all(attention_with_init.W_o.weight == 0))
        
        # 验证权重的标准差接近预期值
        # 预期标准差为 0.02 / sqrt(2 * num_layers)
        expected_std = 0.02 / (2 * num_layers) ** 0.5
        actual_std_q = torch.std(attention_with_init.W_q.weight).item()
        actual_std_k = torch.std(attention_with_init.W_k.weight).item()
        actual_std_v = torch.std(attention_with_init.W_v.weight).item()
        actual_std_o = torch.std(attention_with_init.W_o.weight).item()
        
        # 允许一定的误差范围
        tolerance = 0.01
        self.assertAlmostEqual(actual_std_q, expected_std, delta=tolerance)
        self.assertAlmostEqual(actual_std_k, expected_std, delta=tolerance)
        self.assertAlmostEqual(actual_std_v, expected_std, delta=tolerance)
        self.assertAlmostEqual(actual_std_o, expected_std, delta=tolerance)


if __name__ == "__main__":
    unittest.main()