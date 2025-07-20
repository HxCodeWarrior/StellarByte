import pytest
import torch
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from model.utils.Memory import ByteMemoryManager  # 假设类保存在该模块中


@pytest.fixture
def sample_hidden_state():
    return torch.randn(10, 2, 64)  # [seq_len=10, batch=2, hidden=64]


def test_initialization():
    mgr = ByteMemoryManager(n_layers=3, mem_lens=20)
    assert len(mgr.memory) == 3
    assert mgr.memory_size() == [0, 0, 0]
    assert mgr.batch_policy == "strict"
    assert mgr.fusion_weights == [0.7, 0.3]


def test_update_and_get(sample_hidden_state):
    mgr = ByteMemoryManager(n_layers=1, mem_lens=15)
    mgr.update(0, sample_hidden_state)
    memory = mgr.get(0)
    assert memory.shape == (10, 2, 64)
    assert memory.requires_grad is False  # should be detached


def test_residual_fusion_behavior():
    mgr = ByteMemoryManager(n_layers=1, mem_lens=5)
    h1 = torch.ones(3, 1, 4) * 2  # prev
    h2 = torch.ones(4, 1, 4) * 6  # new
    mgr.update(0, h1)
    mgr.update(0, h2)
    mem = mgr.get(0)
    assert mem.shape == (5, 1, 4)
    # 中间部分应该融合: (0.7 * 2 + 0.3 * 6 = 3.2)
    fused_value = mgr.fusion_weights[0] * 2 + mgr.fusion_weights[1] * 6
    assert torch.allclose(mem[1], torch.tensor([[[fused_value] * 4]]), atol=1e-5)


def test_batch_mismatch_strict():
    mgr = ByteMemoryManager(n_layers=1, mem_lens=10, batch_mismatch_policy="strict")
    h1 = torch.randn(5, 2, 16)
    h2 = torch.randn(5, 3, 16)
    mgr.update(0, h1)
    with pytest.raises(RuntimeError):
        mgr.update(0, h2)


def test_batch_mismatch_select():
    mgr = ByteMemoryManager(n_layers=1, mem_lens=10, batch_mismatch_policy="select")
    h1 = torch.randn(5, 4, 16)
    h2 = torch.randn(5, 2, 16)
    mgr.update(0, h1)
    mgr.update(0, h2)
    mem = mgr.get(0)
    assert mem.shape[1] == 2


def test_batch_mismatch_repeat():
    mgr = ByteMemoryManager(n_layers=1, mem_lens=10, batch_mismatch_policy="repeat")
    h1 = torch.randn(5, 1, 32)
    h2 = torch.randn(5, 3, 32)
    mgr.update(0, h1)
    mgr.update(0, h2)
    mem = mgr.get(0)
    assert mem.shape[1] == 3


def test_update_all():
    mgr = ByteMemoryManager(n_layers=2, mem_lens=5)
    hs = [torch.randn(5, 2, 16), torch.randn(5, 2, 16)]
    mgr.update_all(hs)
    assert mgr.memory_size() == [5, 5]


def test_active_indices():
    mgr = ByteMemoryManager(n_layers=1, mem_lens=10)
    h1 = torch.randn(5, 4, 8)
    mgr.update(0, h1)
    indices = torch.tensor([0, 2])
    h2 = torch.randn(5, 2, 8)
    mgr.update(0, h2, active_indices=indices)
    mem = mgr.get(0)
    assert mem.shape[1] == 2


def test_clear_and_to():
    mgr = ByteMemoryManager(n_layers=2, mem_lens=5)
    mgr.update(0, torch.randn(4, 2, 16))
    mgr.to("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert mgr.get(0).device == device
    mgr.clear()
    assert mgr.memory[0] is None and mgr.memory[1] is None


def test_configure_fusion_weights_valid():
    mgr = ByteMemoryManager(n_layers=1, mem_lens=5)
    mgr.configure_fusion_weights([0.6, 0.4])
    assert mgr.fusion_weights == [0.6, 0.4]


def test_configure_fusion_weights_invalid():
    mgr = ByteMemoryManager(n_layers=1, mem_lens=5)
    with pytest.raises(AssertionError):
        mgr.configure_fusion_weights([0.6, 0.3])  # sum ≠ 1


def test_repr():
    mgr = ByteMemoryManager(n_layers=2, mem_lens=[5, 10])
    mgr.update(0, torch.randn(3, 2, 8))
    out = repr(mgr)
    assert "层0" in out and "层1" in out
    assert "步" in out
    assert "设备" in out

if __name__ == "__main__":
    pytest.main(["-v", __file__])
