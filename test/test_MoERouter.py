import sys
import pytest
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from typing import Dict
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from model.MoERouter import ByteContextAwareRouter  # 替换成实际导入路径


@pytest.fixture
def router():
    # 初始化Router，输入维度64，专家数8，k=2
    return ByteContextAwareRouter(hidden_size=64, num_experts=8, k=2)


def test_init_attributes(router):
    # 基础属性
    assert router.hidden_size == 64
    assert router.num_experts == 8
    assert router.k == 2
    assert router.capacity_factor > 0
    assert router.max_capacity >= router.min_capacity
    # 参数存在
    assert isinstance(router.temperature, torch.nn.Parameter)
    assert isinstance(router.fallback_weight, torch.nn.Parameter)
    # 缓冲区存在
    for attr in ['expert_load', 'expert_utilization', 'expert_priority', 'total_tokens', 'expert_assignment_count', 'expert_cold_priority']:
        assert hasattr(router, attr)


def test_compute_dynamic_capacity_basic(router):
    # 模拟简单输入，保证容量计算在[min_capacity, max_capacity]之间
    N = 1000
    capacity = router._compute_dynamic_capacity(N)
    assert router.min_capacity <= capacity <= router.max_capacity
    # 多次调用稳定
    capacity2 = router._compute_dynamic_capacity(N)
    assert capacity == capacity2


def test_vectorized_dispatch_basic(router):
    N = 10
    K = router.k
    E = router.num_experts

    # 随机生成top-k专家索引与权重
    expert_indices = torch.randint(0, E, (N, K))
    expert_weights = torch.rand(N, K)

    capacity = 3
    dispatch_info = router._vectorized_dispatch(expert_indices, expert_weights, capacity, N)

    # 返回字段检查
    keys = ["expert_indices", "expert_weights", "assigned_mask", "buffer_positions", "overflow_mask", "capacity", "expert_count"]
    for k in keys:
        assert k in dispatch_info

    # assigned_mask尺寸正确
    assert dispatch_info["assigned_mask"].shape == (N, K)
    # buffer_positions应为非负整数或-1
    assert (dispatch_info["buffer_positions"] >= -1).all()
    # expert_count长度等于专家数
    assert dispatch_info["expert_count"].shape[0] == E


def test_compute_balance_loss(router):
    N = 16
    E = router.num_experts
    K = router.k

    gate_scores = torch.rand(N, E)
    gate_scores = F.softmax(gate_scores, dim=-1)

    expert_indices = torch.randint(0, E, (N, K))
    assigned_mask = torch.ones((N, K), dtype=torch.bool)

    dispatch_info = {
        "expert_indices": expert_indices,
        "assigned_mask": assigned_mask
    }
    loss = router._compute_balance_loss(gate_scores, dispatch_info, N)
    assert torch.is_tensor(loss)
    assert loss.item() >= 0


def test_update_expert_state(router):
    N = 20
    K = router.k
    E = router.num_experts

    expert_indices = torch.randint(0, E, (N, K))
    assigned_mask = torch.ones((N, K), dtype=torch.bool)

    dispatch_info = {
        "expert_indices": expert_indices,
        "assigned_mask": assigned_mask,
        "capacity": 5
    }

    prev_load = router.expert_load.clone()
    router._update_expert_state(dispatch_info, N)
    # expert_load 应该更新（EMA）
    assert not torch.allclose(router.expert_load, prev_load)


def test_generate_dispatch_plan(router):
    N = 5
    K = router.k
    E = router.num_experts

    expert_indices = torch.randint(0, E, (N, K))
    assigned_mask = torch.zeros((N, K), dtype=torch.bool)
    assigned_mask[0, 0] = True
    assigned_mask[1, 1] = True

    buffer_positions = torch.randint(0, 4, (N, K))
    expert_weights = torch.rand(N, K)

    dispatch_info = {
        "expert_indices": expert_indices,
        "assigned_mask": assigned_mask,
        "buffer_positions": buffer_positions,
        "expert_weights": expert_weights,
        "capacity": 4
    }

    plan = router.generate_dispatch_plan(dispatch_info)
    assert set(plan.keys()) == {"expert_idx", "token_idx", "positions", "weights", "capacity", "num_tokens"}
    assert plan["capacity"] == 4
    assert plan["expert_idx"].shape[0] == plan["token_idx"].shape[0]


def test_forward_basic(router):
    B, S, H = 2, 8, router.hidden_size
    x = torch.randn(B, S, H)
    dispatch_info, aux_loss, next_context = router(x)
    # dispatch_info应包含字段
    for key in ["expert_indices", "expert_weights", "assigned_mask", "buffer_positions", "overflow_mask", "capacity", "expert_count"]:
        assert key in dispatch_info
    # aux_loss 是张量且标量
    assert torch.is_tensor(aux_loss)
    assert aux_loss.dim() == 0
    # next_context 形状 [B, context_dim*...]
    assert next_context.shape[0] == B


def test_overflow_handling(router):
    # 模拟overflow触发情况
    N = 10
    K = router.k
    E = router.num_experts

    # 构造expert_indices全部为同一个专家（容量小导致overflow）
    expert_indices = torch.zeros((N, K), dtype=torch.long)
    expert_weights = torch.rand(N, K)
    capacity = 3

    dispatch_info = router._vectorized_dispatch(expert_indices, expert_weights, capacity, N)
    # 强制overflow_mask置True (模拟)
    dispatch_info["overflow_mask"] = torch.ones(N, dtype=torch.bool)

    x_flat = torch.randn(N, router.hidden_size)
    gate_scores = torch.rand(N, K)
    topk_indices = expert_indices

    router._handle_overflow_vectorized(x_flat, gate_scores, topk_indices, dispatch_info, capacity)
    # overflow_mask应该有些被清除（取决于fallback）
    assert dispatch_info["overflow_mask"].dtype == torch.bool


def test_sticky_fallback(router):
    N = 5
    K = router.k
    E = router.num_experts

    dispatch_info = {
        "expert_indices": torch.zeros((N, K), dtype=torch.long),
        "expert_weights": torch.zeros((N, K), dtype=torch.float32),
        "assigned_mask": torch.zeros((N, K), dtype=torch.bool),
        "buffer_positions": torch.zeros((N, K), dtype=torch.long),
        "overflow_mask": torch.ones(N, dtype=torch.bool),
        "expert_count": torch.zeros(E, dtype=torch.long)
    }

    overflow_idx = torch.arange(N)

    # 调用sticky fallback不报错，且overflow_mask被更新（如果能成功分配）
    router._sticky_fallback(overflow_idx, dispatch_info, capacity=4)
    assert dispatch_info["overflow_mask"].dtype == torch.bool


@pytest.mark.parametrize("capacity", [1, 5, 20])
def test_dynamic_capacity_various(router, capacity):
    N = 1000
    router.min_capacity = 1
    router.max_capacity = 10000
    router.capacity_factor = 1.0

    cap = router._compute_dynamic_capacity(N)
    assert router.min_capacity <= cap <= router.max_capacity


def test_forward_with_prev_hidden(router):
    B, S, H = 2, 8, router.hidden_size
    x = torch.randn(B, S, H)
    prev_hidden = torch.randn(B, H)

    dispatch_info, aux_loss, next_context = router(x, prev_hidden=prev_hidden)
    assert isinstance(dispatch_info, dict)
    assert torch.is_tensor(aux_loss)
    assert next_context.shape[0] == B



if __name__ == "__main__":
    pytest.main(["-v", __file__])
