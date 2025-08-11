import sys
import pytest
import torch
import torch.nn.functional as F
from pathlib import Path

# 将项目根目录添加到sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from model.config import ByteModelConfig
from model.Attention import ByteMultiHeadSelfAttention


# --------------------
# Helper fixtures
# --------------------
@pytest.fixture(params=[torch.float32, torch.float16])
def dtype(request):
    return request.param

@pytest.fixture
def base_config():
    # 使用较小尺寸以加快测试
    cfg = ByteModelConfig(
        vocab_size=100,
        model_dim=64,               # embed_dim
        num_layers=2,
        num_attention_heads=8,      # 必须能整除 model_dim
        num_kv_heads=4,
        hidden_dim=256,
        max_seq_len=128,
        attention_window_size=0,    # 默认不使用窗口（可在个别测试覆盖）
        use_flash_attention=False,
        tensor_parallel_size=1,     # 禁用张量并行（测试中使用单设备）
    )
    return cfg

@pytest.fixture
def mha(base_config, dtype):
    torch.manual_seed(42)
    # 实例化
    m = ByteMultiHeadSelfAttention(base_config, layer_id=0, num_layers=base_config.num_layers)
    # 确保 rotary_emb 存在并将其 apply_rotary 设置为恒等映射，避免外部依赖
    if not hasattr(m, "rotary_emb") or not hasattr(m.rotary_emb, "apply_rotary"):
        class DummyRot:
            def apply_rotary(self, q, k):
                return q, k
        m.rotary_emb = DummyRot()
    else:
        # 覆盖为恒等
        m.rotary_emb.apply_rotary = lambda q, k: (q, k)

    # 将投影层初始化为确定性小值(可预测)
    torch.manual_seed(0)
    for p in [m.W_q.weight, m.W_k.weight, m.W_v.weight, m.W_o.weight]:
        with torch.no_grad():
            nn_std = 0.02
            p.normal_(mean=0.0, std=nn_std)

    m = m.to(dtype)
    return m

# --------------------
# Tests for helper methods
# --------------------
def test_repeat_kv_identity():
    # kv shape: [B, T, num_kv_heads, head_dim]
    B, T, H_k, D = 2, 5, 2, 4
    kv = torch.arange(B * T * H_k * D, dtype=torch.float32).view(B, T, H_k, D)
    cfg = ByteModelConfig(model_dim=H_k * D, num_attention_heads=H_k, num_kv_heads=H_k, tensor_parallel_size=1)
    m = ByteMultiHeadSelfAttention(cfg)

    # n_rep = 1 -> 返回值应和输入相同
    out = m._repeat_kv(kv, n_rep=1)
    assert out.shape == kv.shape
    assert torch.equal(out, kv)

    # n_rep = 3 -> GQA 顺序
    n_rep = 3
    out3 = m._repeat_kv(kv, n_rep=n_rep)
    assert out3.shape == (B, T, H_k * n_rep, D)

    # 检查每个单独的 head 是否重复了 n_rep 次
    for h in range(H_k):
        for rep in range(n_rep):
            idx = h * n_rep + rep
            chunk = out3[:, :, idx, :]  # 单个 head
            expected = kv[:, :, h, :]
            assert torch.equal(chunk, expected), f"Head {h} repetition {rep} mismatch"


def test_build_causal_mask_basic():
    cfg = ByteModelConfig(model_dim=16, num_attention_heads=4, num_kv_heads=4, tensor_parallel_size=1)
    m = ByteMultiHeadSelfAttention(cfg)
    seq_len = 6
    mask = m._build_causal_mask(seq_len, device=torch.device("cpu"), dtype=torch.float32)
    # shape: [1,1,seq_len,seq_len]
    assert mask.shape == (1,1,seq_len,seq_len)
    # 上三角 (j > i) 应为极小值 (约等于 -1e9)
    min_val = -1e9
    # 检查部分位置
    assert mask[0,0,0,1] <= min_val/2  # 未来位置
    assert torch.isclose(mask[0,0,3,1], torch.tensor(0.0), atol=1e-6)  # 过去或自身位置

def test_build_causal_mask_window():
    cfg = ByteModelConfig(model_dim=32, num_attention_heads=4, num_kv_heads=4, tensor_parallel_size=1, attention_window_size=3)
    m = ByteMultiHeadSelfAttention(cfg)
    # 强制设置 window_size（构造时可能没带入）
    m.window_size = 3
    seq_len = 7
    mask = m._build_causal_mask(seq_len, device=torch.device("cpu"), dtype=torch.float32)
    assert mask.shape == (1,1,seq_len,seq_len)
    min_val = -1e9
    # 对于位置 i=4, 允许关注 [2,3,4] (window_size=3 -> i-j < 3 即 j> i-3)
    i = 4
    allowed_js = [2,3,4]
    for j in allowed_js:
        assert torch.isclose(mask[0,0,i,j], torch.tensor(0.0), atol=1e-6)
    # j = 0 (超出窗口) 应为 min_val
    assert mask[0,0,i,0] <= min_val/2

def test_adjust_attention_mask_bool_and_float():
    cfg = ByteModelConfig(model_dim=32, num_attention_heads=4, num_kv_heads=4, tensor_parallel_size=1)
    m = ByteMultiHeadSelfAttention(cfg)
    seq_len = 5
    device = torch.device("cpu")
    dtype = torch.float32

    # bool mask shape [B, T]
    bool_mask = torch.tensor([[1,1,0,1,1]], dtype=torch.bool)
    out = m._adjust_attention_mask(bool_mask, seq_len, device, dtype)
    assert out.shape == (1,1,1,seq_len)
    # positions with 0 -> 应该是 min_val
    min_val = -1e9
    assert out[0,0,0,2] <= min_val/2
    # positions with 1 -> 0
    assert torch.isclose(out[0,0,0,1], torch.tensor(0.0), atol=1e-6)

    # already additive float mask [B,1,1,T]
    float_mask = torch.tensor([[[[0.0, -1e9, 0.0, 0.0, 0.0]]]])
    out2 = m._adjust_attention_mask(float_mask, seq_len, device, dtype)
    assert out2.shape == (1,1,1,seq_len)
    assert torch.isclose(out2, float_mask, atol=1e-6).all()

def test_merge_masks_combines_shapes():
    cfg = ByteModelConfig(model_dim=32, num_attention_heads=4, num_kv_heads=4, tensor_parallel_size=1)
    m = ByteMultiHeadSelfAttention(cfg)
    seq_len = 4
    causal = m._build_causal_mask(seq_len, device=torch.device("cpu"), dtype=torch.float32)  # [1,1,T,T]
    # attention_mask [B,1,1,T]
    attn = torch.tensor([[[[0.0, -1e9, 0.0, 0.0]]]], dtype=torch.float32)
    merged = m._merge_masks(causal, attn)
    # merged shape [B,1,T,T]
    assert merged.shape == (1,1,seq_len,seq_len)
    # 比较：对于每个位置，merged == causal + expanded(attn)
    expanded = attn.expand(-1, -1, seq_len, -1)
    assert torch.allclose(merged, causal + expanded)

# --------------------
# Tests for forward execution (shape & stability)
# --------------------
def test_forward_returns_correct_shape(mha, dtype):
    B, T = 2, 10
    embed_dim = mha.embed_dim
    x = torch.randn(B, T, embed_dim, dtype=dtype)
    # attention_mask as bool [B, T]
    attn_mask = torch.ones(B, T, dtype=torch.bool)
    out = mha(x, attention_mask=attn_mask)
    assert out.shape == (B, T, embed_dim)
    # dtype is preserved on output projection (可能会因内部计算差异，但输出张量 dtype 应与参数dtype一致)
    assert out.dtype == dtype

def test_forward_runs_in_eval_and_train(mha):
    B, T = 1, 12
    embed_dim = mha.embed_dim
    x = torch.randn(B, T, embed_dim, dtype=next(mha.parameters()).dtype)
    mha.train()
    out_train = mha(x)
    mha.eval()
    out_eval = mha(x)
    # 形状一致
    assert out_train.shape == out_eval.shape == (B, T, embed_dim)
    # 当 dropout 在训练时，结果在数值上可能不同
    if mha.attn_dropout.p > 0 or mha.resid_dropout.p > 0:
        assert not torch.allclose(out_train, out_eval)

# 一个简单的数值稳定性/可复现性测试（固定随机种子）
def test_forward_reproducible_with_same_seed(mha):
    mha.eval()  # 禁用dropout
    B, T = 2, 8
    embed_dim = mha.embed_dim
    torch.manual_seed(123)
    x1 = torch.randn(B, T, embed_dim, dtype=next(mha.parameters()).dtype)
    torch.manual_seed(123)
    x2 = torch.randn(B, T, embed_dim, dtype=next(mha.parameters()).dtype)
    assert torch.allclose(x1, x2)
    out1 = mha(x1)
    out2 = mha(x2)
    assert torch.allclose(out1, out2)

# --------------------
# Optional: test that invalid head / tp settings assert early
# --------------------
def test_asserts_on_invalid_parallel_divisibility():
    # num_heads=7, tp_size=2 -> 7 % 2 != 0 应触发断言
    cfg = ByteModelConfig(model_dim=14, num_attention_heads=7, num_kv_heads=7, tensor_parallel_size=2)
    with pytest.raises(AssertionError):
        ByteMultiHeadSelfAttention(cfg)


if __name__ == "__main__":
    pytest.main(["-v",__file__])