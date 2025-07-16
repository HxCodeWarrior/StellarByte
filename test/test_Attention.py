import sys
import pytest
import torch
from pathlib import Path

# 将项目根目录添加到sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from model.config import ByteModelConfig
from model.utils.KVCache import KVCache
from model.Attention import MultiHeadSelfAttention


@pytest.fixture
def base_config():
    return ByteModelConfig(
        model_dim=64,
        num_attention_heads=4,
        num_kv_heads=2,
        xpos_scale_base=512,
        xpos_rope_theta=10000,
        attention_dropout_prob=0.1,  # 关闭 dropout 保持可重复性
        residual_dropout_prob=0.1,
        max_seq_len=32,
        use_flash_attention=False,
        use_causal=True,
        model_parallel_size=1
    )


def test_forward_no_kvcache(base_config):
    model = MultiHeadSelfAttention(base_config)
    x = torch.randn(2, 8, base_config.model_dim)
    out = model(x)
    assert out.shape == x.shape, "输出形状应与输入一致"
    assert not torch.isnan(out).any(), "输出不能含有 NaN"


def test_forward_with_kvcache(base_config):
    cache = KVCache(
        num_layers=1,
        num_heads=base_config.num_attention_heads,
        head_dim=base_config.model_dim // base_config.num_attention_heads,
        max_seq_len=base_config.max_seq_len,
        device="cpu"
    )
    model = MultiHeadSelfAttention(base_config, kv_cache=cache, layer_id=0)

    seq_len = 10
    x = torch.randn(1, seq_len, base_config.model_dim)
    outputs = []

    for i in range(seq_len):
        token = x[:, i:i+1, :]
        out = model(token)
        assert out.shape == token.shape, f"第{i}步输出形状错误"
        outputs.append(out)

    final_output = torch.cat(outputs, dim=1)
    assert final_output.shape == x.shape, "拼接后形状应一致"
    assert not torch.isnan(final_output).any(), "不能含有 NaN"


def test_attention_mask_shape_and_type(base_config):
    model = MultiHeadSelfAttention(base_config)
    x = torch.randn(2, 6, base_config.model_dim)

    pad_mask = torch.zeros(2, 1, 1, 6)
    pad_mask[0, :, :, 3:] = float('-inf')  # 第一条样本后3个是pad

    out = model(x, additive_mask=pad_mask)
    assert out.shape == x.shape, "mask输入后输出形状应一致"
    assert not torch.isnan(out).any(), "mask后不能含NaN"


def test_head_pruning(base_config):
    model = MultiHeadSelfAttention(base_config)
    model.prune_heads([0, 2])
    assert model.head_mask.tolist() == [False, True, False, True], "剪枝头应被标记为 False"


def test_multiple_batch_and_seq(base_config):
    model = MultiHeadSelfAttention(base_config)
    for batch in [1, 4]:
        for seq in [4, 16]:
            x = torch.randn(batch, seq, base_config.model_dim)
            out = model(x)
            assert out.shape == x.shape, f"batch={batch}, seq={seq} 输出错误"


def test_flash_attention_flag_logic(monkeypatch, base_config):
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", lambda *args, **kwargs: args[2])  # 模拟 Flash 输出 = v
    base_config.use_flash_attention = True
    model = MultiHeadSelfAttention(base_config)
    x = torch.randn(2, 6, base_config.model_dim)
    out = model(x)
    assert out.shape == x.shape


def test_rotary_encoding_called(base_config):
    model = MultiHeadSelfAttention(base_config)
    rotary = model.rotary

    # 模拟rotary被调用时输出是否合理（内部测试）
    cos, sin, scale = rotary._get_cos_sin_scale(10, device="cpu", dtype=torch.float32)
    assert cos.shape == (10, base_config.model_dim // base_config.num_attention_heads)
    assert torch.is_tensor(scale), "应返回张量"


def test_kvcache_reset_and_overflow(base_config):
    cache = KVCache(
        num_layers=1,
        num_heads=base_config.num_attention_heads,
        head_dim=base_config.model_dim // base_config.num_attention_heads,
        max_seq_len=8,
        device="cpu"
    )
    model = MultiHeadSelfAttention(base_config, kv_cache=cache, layer_id=0)
    x = torch.randn(1, 16, base_config.model_dim)

    for i in range(16):
        token = x[:, i:i+1, :]
        _ = model(token)

    # KVCache 最多保留 max_seq_len 个 token
    k, v = cache.get(0)
    assert k.shape[1] == 8, "KVCache 应自动滑窗裁剪"
    cache.reset()
    assert cache.layer_length(0) == 0, "reset 后长度应为 0"

if __name__ == "__main__":
    pytest.main(["-v",__file__])