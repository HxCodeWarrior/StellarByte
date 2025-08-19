import os
import sys
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing import assert_close
from pathlib import Path

# 将项目根目录添加到sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from model.MoE import ExpertFFN, ByteMoELayer


# =========================================================
# ExpertFFN 测试
# =========================================================
@pytest.mark.parametrize("activation", ["gelu", "relu", "swish", "silu"])
@pytest.mark.parametrize("use_layernorm", [True, False])
@pytest.mark.parametrize("residual", [True, False])
def test_expertffn_forward(activation, use_layernorm, residual):
    torch.manual_seed(42)
    d_model, d_ff = 16, 32
    x = torch.randn(4, d_model)

    model = ExpertFFN(d_model, d_ff, activation=activation, use_layernorm=use_layernorm, residual=residual)
    out = model(x)

    assert out.shape == x.shape
    if not residual and not use_layernorm:
        assert not torch.allclose(out, x)


def test_expertffn_invalid_activation():
    with pytest.raises(ValueError):
        ExpertFFN(16, 32, activation="xxx")


def test_expertffn_dropout_effect():
    torch.manual_seed(0)
    model = ExpertFFN(16, 32, activation="relu", dropout=0.5)
    model.train()  # Dropout 仅在训练模式生效
    x = torch.randn(8, 16)

    out1 = model(x)
    out2 = model(x)
    # Dropout 生效时，两次前向应当不同
    assert not torch.allclose(out1, out2)

    model.eval()  # 评估模式下 Dropout 关闭
    out3 = model(x)
    out4 = model(x)
    # 评估模式下，两次前向应当完全一致
    assert torch.allclose(out3, out4)


# =========================================================
# ByteMoELayer 单卡功能测试
# =========================================================
@pytest.mark.parametrize("k", [1, 2])
def test_moe_forward_singlecard(k):
    torch.manual_seed(0)
    moe = ByteMoELayer(d_model=16, d_ff=32, n_experts=4, world_size=1, k=k)

    x = torch.randn(12, 16)
    probs, vals, idx = moe._compute_routing(x)
    assert probs.shape == (12, 4)
    assert idx.shape == (12, k)

    aux_loss = moe._compute_aux_loss(probs, idx)
    assert aux_loss.ndim == 0

    s_all = torch.tensor([x.shape[0]])
    offsets = torch.tensor([0])

    recv_tokens, recv_gate, recv_expert_local, recv_origin_local_idx, send_splits_back = moe._dispatch_tokens(
        x, idx, vals, s_all, offsets
    )
    assert recv_tokens.shape[1] == 16

    cap = int(1.25 * x.shape[0] / 4)
    out, gate, origin_idx = moe._local_expert_computation(
        recv_tokens, recv_expert_local, recv_gate, recv_origin_local_idx, capacity=cap
    )
    if out is not None:
        combined, g, o = moe._combine_results(out, gate, origin_idx, send_splits_back, s_all, offsets)
        assert combined.shape[1] == 16
        assert combined.shape[0] == g.shape[0]


def test_moe_empty_input():
    moe = ByteMoELayer(16, 32, 2, world_size=1, k=1)
    empty = torch.empty(0, 16)
    out = moe._local_expert_computation(empty, torch.empty(0, dtype=torch.long),
                                        torch.empty(0), torch.empty(0, dtype=torch.long), capacity=4)
    assert all(v is None for v in out)


def test_aux_loss_balanced():
    probs = torch.full((8, 4), 0.25)
    idx = torch.randint(0, 4, (8, 1))
    moe = ByteMoELayer(16, 32, 4)
    loss = moe._compute_aux_loss(probs, idx)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-2)


def test_backward_gradcheck():
    torch.manual_seed(0)
    moe = ByteMoELayer(16, 32, 4, world_size=1, k=2)
    x = torch.randn(2, 3, 16, requires_grad=True)  # B=2, T=3, D=16
    out, aux = moe(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# =========================================================
# 分布式测试 (world_size=2)
# =========================================================
def run_dist(rank, world_size):
    # 找一个随机可用端口
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))
    port = s.getsockname()[1]
    s.close()

    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    torch.manual_seed(0)

    moe = ByteMoELayer(16, 32, 4, world_size=world_size, k=1).to(rank)
    x = torch.randn(2, 4, 16)  # B=2, T=4, D=16

    out, aux_loss = moe(x)
    assert out.shape == (2, 4, 16)
    assert aux_loss.ndim == 0

    dist.destroy_process_group()


@pytest.mark.dist
def test_distributed_world2():
    world_size = 2
    if __name__ == "__main__":
        mp.spawn(run_dist, args=(world_size,), nprocs=world_size, join=True)


# =========================================================
# 边界与异常
# =========================================================
def test_invalid_k_bigger_than_experts():
    with pytest.raises(AssertionError):
        ByteMoELayer(16, 32, n_experts=2, k=3)

if __name__ == "__main__":
    pytest.main(["-v", __file__])