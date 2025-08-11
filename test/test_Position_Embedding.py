# test_byterope_full.py
import sys
import pytest
import torch
import math
from pathlib import Path

# 将项目根目录添加到sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from model.Position_Embedding import ByteDynamicRoPE


def test_init_odd_dim_raises():
    """dim 必须是偶数，否则应触发断言"""
    with pytest.raises(AssertionError):
        ByteDynamicRoPE(dim=7)


def test_base_freq_values():
    """base_freq 计算结果与公式一致"""
    dim = 8
    rope = ByteDynamicRoPE(dim=dim, base_theta=10000.0, ntk_alpha=1.0, max_seq_len=16, device='cpu')
    dim_half = dim // 2
    expected = 10000.0 ** (-2 * torch.arange(0, dim_half, 1) / dim)
    torch.testing.assert_close(rope.base_freq.cpu(), expected, rtol=1e-6, atol=1e-6)


def test_ntk_scale_factor_behaviour():
    dim = 8
    rope = ByteDynamicRoPE(dim=dim, ntk_alpha=2.0, max_seq_len=10, device='cpu')
    # 短序列，返回 1.0
    assert math.isclose(rope._ntk_scale_factor(5), 1.0)
    # 长序列，按公式计算
    seq_len = 20
    ratio = seq_len / rope.max_seq_len
    exponent = dim / (dim - 2)
    expected = 2.0 * (ratio ** exponent)
    assert math.isclose(rope._ntk_scale_factor(seq_len), expected)


def test_build_cos_sin_table_shapes_and_values():
    dim = 8
    seq_len = 4
    rope = ByteDynamicRoPE(dim=dim, base_theta=10000.0, device='cpu')
    rope._build_cos_sin_table(seq_len)
    # 检查形状
    assert rope.cos_cached.shape == (1, seq_len, 1, dim // 2)
    assert rope.sin_cached.shape == (1, seq_len, 1, dim // 2)
    # 手动构建预期值并比较
    inv_freq = rope.base_freq / rope._ntk_scale_factor(seq_len)
    t = torch.arange(seq_len, device='cpu').float()
    freqs = torch.outer(t, inv_freq)
    torch.testing.assert_close(rope.cos_cached[0, :, 0, :], freqs.cos())
    torch.testing.assert_close(rope.sin_cached[0, :, 0, :], freqs.sin())


def test_apply_rotary_shapes_and_numerical():
    dim = 8
    batch = 1
    seq_len = 3
    heads = 2
    rope = ByteDynamicRoPE(dim=dim, device='cpu')
    q = torch.randn(batch, seq_len, heads, dim)
    k = torch.randn(batch, seq_len, heads, dim)
    q_rot, k_rot = rope.apply_rotary(q, k)

    # 检查形状
    assert q_rot.shape == (batch, seq_len, heads, dim)
    assert k_rot.shape == (batch, seq_len, heads, dim)

    # 检查数值正确性（和手动旋转比较）
    cos = rope.cos_cached[:, :seq_len, :, :]
    sin = rope.sin_cached[:, :seq_len, :, :]
    def manual_rot(x):
        x_complex = x.float().view(*x.shape[:-1], -1, 2)
        x_real = x_complex[..., 0]
        x_imag = x_complex[..., 1]
        xr = x_real * cos - x_imag * sin
        xi = x_real * sin + x_imag * cos
        return torch.stack([xr, xi], dim=-1).flatten(-2, -1).type_as(x)
    torch.testing.assert_close(q_rot, manual_rot(q), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_rot, manual_rot(k), rtol=1e-5, atol=1e-5)


def test_cache_rebuild_on_offset_exceed():
    dim = 8
    rope = ByteDynamicRoPE(dim=dim, max_seq_len=4, device='cpu')
    old_len = rope.cos_cached.shape[1]
    q = torch.randn(1, 3, 1, dim)
    k = torch.randn(1, 3, 1, dim)
    # seq_len + seq_offset > max_seq_len -> 重建缓存
    _ = rope.apply_rotary(q, k, seq_offset=5)
    new_len = rope.cos_cached.shape[1]
    assert new_len >= 3 + 5
    assert new_len != old_len


def test_rotation_preserves_norm():
    dim = 8
    rope = ByteDynamicRoPE(dim=dim, device='cpu')
    q = torch.randn(2, 5, 3, dim)
    k = torch.randn(2, 5, 3, dim)
    q_rot, k_rot = rope.apply_rotary(q, k)
    # 最后一维范数应保持
    norm_before_q = q.norm(dim=-1)
    norm_after_q = q_rot.norm(dim=-1)
    norm_before_k = k.norm(dim=-1)
    norm_after_k = k_rot.norm(dim=-1)
    torch.testing.assert_close(norm_before_q, norm_after_q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(norm_before_k, norm_after_k, rtol=1e-6, atol=1e-6)

if __name__ == '__main__':
    pytest.main(['-v', __file__])