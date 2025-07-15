# test_kv_cache.py
import pytest
import torch
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from model.utils.KVCache import KVCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_kv(bsz, t, h_local, d, dtype, device):
    """生成随机 K/V，用于 append 测试"""
    return torch.randn(bsz, t, h_local, d, dtype=dtype, device=device)


# ---------- 基础初始化 ----------
def test_init_basic():
    cache = KVCache(
        num_layers=4,
        num_heads=8,
        head_dim=64,
        max_seq_len=32,
        tensor_parallel_size=2,   # H_local = 4
        tensor_parallel_rank=0,
        device=DEVICE,
    )
    assert cache.L == 4
    assert cache.H_local == 4
    assert cache.global_length() == 0
    assert cache.batch_size is None


# ---------- Append + Get ----------
@pytest.mark.parametrize("tp_size", [1, 2])
def test_append_and_get(tp_size):
    B, T = 2, 8
    cache = KVCache(
        num_layers=2,
        num_heads=8,
        head_dim=32,
        max_seq_len=32,
        tensor_parallel_size=tp_size,
        tensor_parallel_rank=0,
        device=DEVICE,
    )
    H_local = 8 // tp_size
    kv = make_kv(B, T, H_local, 32, torch.float16, DEVICE)
    cache.append(0, kv, kv)          # layer‑0
    k, v = cache.get(0)
    assert k.shape == (B, T, H_local, 32)
    assert torch.allclose(k, kv)
    assert cache.layer_length(0) == T
    assert cache.global_length() == T


# ---------- Sliding‑Window ----------
def test_sliding_window():
    B, max_T, head_dim = 1, 16, 64
    cache = KVCache(
        num_layers=1,
        num_heads=8,
        head_dim=head_dim,
        max_seq_len=max_T,
        device=DEVICE,
    )

    H_local = cache.H_local
    # 先写满
    kv1 = make_kv(B, max_T, H_local, head_dim, torch.float16, DEVICE)
    cache.append(0, kv1, kv1)

    # 再追加 4 token，会触发窗口左移 4
    kv2 = make_kv(B, 4, H_local, head_dim, torch.float16, DEVICE)
    cache.append(0, kv2, kv2)

    k, _ = cache.get(0)
    # 期望顺序 = kv1[:,4:] + kv2
    expected = torch.cat([kv1[:, 4:], kv2], dim=1)
    assert torch.allclose(k, expected, atol=1e-3)
    assert cache.layer_length(0) == max_T  # 仍然是窗口长度


# ---------- Reset ----------
def test_reset():
    cache = KVCache(1, 8, 32, 8, device=DEVICE)
    H = cache.H_local
    kv = make_kv(1, 8, H, 32, torch.float16, DEVICE)
    cache.append(0, kv, kv)
    cache.reset()
    k, v = cache.get(0)
    assert cache.layer_length(0) == 0
    assert torch.all(k == 0) and torch.all(v == 0)


# ---------- Device 迁移 ----------
def test_to_device():
    tgt = "cpu"
    cache = KVCache(1, 4, 16, 4, device=DEVICE)
    H = cache.H_local
    kv = make_kv(1, 4, H, 16, torch.float16, DEVICE)
    cache.append(0, kv, kv)
    cache.to(tgt)
    k, _ = cache.get(0)
    assert k.device.type == "cpu"


# ---------- 张量并行非法配置 ----------
def test_wrong_tp_size():
    with pytest.raises(AssertionError):
        KVCache(1, num_heads=10, head_dim=32, max_seq_len=16, tensor_parallel_size=3)


# ---------- Shape / Batch Size 错误 ----------
def test_shape_mismatch():
    cache = KVCache(1, 4, 16, 8, device=DEVICE)
    H = cache.H_local
    good_kv = make_kv(2, 4, H, 16, torch.float16, DEVICE)
    cache.append(0, good_kv, good_kv)
    # batch 大小改成 3 -> 触发断言
    bad_kv = make_kv(3, 4, H, 16, torch.float16, DEVICE)
    with pytest.raises(AssertionError):
        cache.append(0, bad_kv, bad_kv)


# ---------- 混合精度 ----------
def test_dtypes():
    cache = KVCache(1, 4, 16, 8, key_dtype=torch.bfloat16,
                    value_dtype=torch.float16, device=DEVICE)
    H = cache.H_local
    k = make_kv(1, 2, H, 16, torch.bfloat16, DEVICE)
    v = make_kv(1, 2, H, 16, torch.float16, DEVICE)
    cache.append(0, k, v)
    got_k, got_v = cache.get(0)
    assert got_k.dtype == torch.bfloat16
    assert got_v.dtype == torch.float16


# ---- 直接运行脚本时也执行全部测试 ----
if __name__ == "__main__":
    pytest.main([__file__])
