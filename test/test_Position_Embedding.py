import sys
from pathlib import Path

# 将项目根目录添加到sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from model.config import ByteModelConfig
from model.Position_Embedding import XPosRotaryEmbedding
import torch
import torch.nn as nn
import pytest

# 测试配置
@pytest.fixture
def config():
    return ByteModelConfig(
        model_dim=64,
        max_seq_len=2048,
        xpos_rope_theta=10000.0,
        xpos_scale_base=512
    )

# 测试初始化
def test_initialization(config):
    """测试位置编码模块初始化"""
    head_dim = config.model_dim
    rotary = XPosRotaryEmbedding(head_dim, config.max_seq_len, config.xpos_scale_base, config.xpos_rope_theta)
    
    # 验证缓冲区注册
    assert hasattr(rotary, "inv_freq")
    assert rotary.inv_freq.shape == (head_dim // 2,)
    
    # 验证参数设置
    assert rotary.dim == head_dim
    assert rotary.scale_base == config.xpos_scale_base
    assert rotary.theta == config.xpos_rope_theta

# 测试维度异常
def test_odd_dimension_exception():
    """测试奇数维度时的异常抛出"""
    with pytest.raises(AssertionError, match="Dimension must be even for rotary embedding."):
        XPosRotaryEmbedding(head_dim=63, max_seq_len=1024, scale_base=512, theta=10000.0)

# 测试前向传播形状
def test_forward_shape(config):
    """测试输出形状一致性"""
    head_dim = config.model_dim
    rotary = XPosRotaryEmbedding(head_dim, config.max_seq_len, config.xpos_scale_base, config.xpos_rope_theta)
    
    # 创建测试输入
    batch_size, seq_len, num_heads = 2, 128, 4
    xq = torch.randn(batch_size, seq_len, num_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    # 执行前向传播
    xq_rot, xk_rot = rotary(xq, xk)
    
    # 验证输出形状
    assert xq_rot.shape == (batch_size, seq_len, num_heads, head_dim)
    assert xk_rot.shape == (batch_size, seq_len, num_heads, head_dim)

# 测试旋转不变性
def test_xpos_numerical_stability(config):
    """XPos 应该能够在较长序列上保持数值稳定而不爆炸"""
    head_dim = config.model_dim // config.num_attention_heads
    rotary = XPosRotaryEmbedding(head_dim, config.max_seq_len, config.xpos_scale_base, config.xpos_rope_theta)

    # 构造大序列
    xq = torch.randn(1, 2048, config.num_attention_heads, head_dim)
    xk = torch.randn(1, 2048, config.num_attention_heads, head_dim)

    xq_rot, xk_rot = rotary(xq, xk)

    # 不应出现NaN或inf
    assert torch.isfinite(xq_rot).all(), "xq_rot contains NaNs or Infs"
    assert torch.isfinite(xk_rot).all(), "xk_rot contains NaNs or Infs"

# 测试缩放因子影响
def test_scale_effect(config):
    """测试缩放因子对输出的影响"""
    head_dim = config.model_dim
    base_rotary = XPosRotaryEmbedding(head_dim, config.max_seq_len, config.xpos_scale_base, config.xpos_rope_theta)
    
    # 创建不同缩放因子的模块
    scaled_rotary = XPosRotaryEmbedding(head_dim, config.max_seq_len, config.xpos_scale_base * 2, config.xpos_rope_theta)
    
    # 相同输入
    xq = torch.randn(1, 10, 1, head_dim)
    xk = torch.randn(1, 10, 1, head_dim)
    
    # 获取输出
    base_xq, base_xk = base_rotary(xq, xk)
    scaled_xq, scaled_xk = scaled_rotary(xq, xk)
    
    # 验证缩放因子改变导致输出不同
    assert not torch.allclose(base_xq, scaled_xq, atol=1e-6)
    assert not torch.allclose(base_xk, scaled_xk, atol=1e-6)

# 测试设备兼容性
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_compatibility(config, device):
    """测试不同设备上的兼容性"""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    head_dim = config.model_dim
    rotary = XPosRotaryEmbedding(head_dim, config.max_seq_len, config.xpos_scale_base, config.xpos_rope_theta).to(device)
    
    # 创建测试输入
    xq = torch.randn(2, 64, 2, head_dim, device=device)
    xk = torch.randn(2, 64, 2, head_dim, device=device)
    
    # 执行前向传播
    xq_rot, xk_rot = rotary(xq, xk)
    
    # 验证输出设备
    assert xq_rot.device.type == device
    assert xk_rot.device.type == device

# 测试不同序列长度
@pytest.mark.parametrize("seq_len", [1, 16, 128, 1024])
def test_variable_sequence_length(config, seq_len):
    """测试不同序列长度的处理能力"""
    head_dim = config.model_dim
    rotary = XPosRotaryEmbedding(head_dim, config.max_seq_len, config.xpos_scale_base, config.xpos_rope_theta)
    
    # 创建测试输入
    xq = torch.randn(1, seq_len, 1, head_dim)
    xk = torch.randn(1, seq_len, 1, head_dim)
    
    # 执行前向传播
    xq_rot, xk_rot = rotary(xq, xk)
    
    # 验证输出形状
    assert xq_rot.shape == (1, seq_len, 1, head_dim)
    assert xk_rot.shape == (1, seq_len, 1, head_dim)

# 测试数值稳定性
def test_numerical_stability(config):
    """测试极端条件下的数值稳定性"""
    head_dim = config.model_dim
    rotary = XPosRotaryEmbedding(head_dim, config.max_seq_len, config.xpos_scale_base, config.xpos_rope_theta)
    
    # 创建极端输入（大值）
    xq = torch.randn(1, 2048, 8, head_dim) * 100
    xk = torch.randn(1, 2048, 8, head_dim) * 100
    
    # 执行前向传播
    xq_rot, xk_rot = rotary(xq, xk)
    
    # 验证无NaN/Inf
    assert not torch.isnan(xq_rot).any()
    assert not torch.isinf(xq_rot).any()
    assert not torch.isnan(xk_rot).any()
    assert not torch.isinf(xk_rot).any()

if __name__ == "__main__":
    pytest.main(["-v", __file__])