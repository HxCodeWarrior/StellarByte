import sys
import pytest
import torch
from pathlib import Path

# 将项目根目录添加到sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from model.DecoderLayer import ByteDecoderLayer
from model.config import ByteModelConfig

# 基础配置
@pytest.fixture
def base_config():
    return ByteModelConfig(
        model_dim=128,
        num_layers=12,
        num_attention_heads=8,
        layerscale_init=1e-4,
        drop_path_prob=0.2,
        parallel_residual=True,
        use_flash_attention=False
    )

# 测试设备兼容性
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_compatibility(base_config, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    layer = ByteDecoderLayer(base_config, layer_id=0).to(device)
    x = torch.randn(2, 16, base_config.model_dim).to(device)
    y = layer(x)
    assert y.device.type == device

# 测试前向传播形状
def test_forward_shape(base_config):
    layer = ByteDecoderLayer(base_config, layer_id=0)
    x = torch.randn(2, 16, base_config.model_dim)
    y = layer(x)
    assert y.shape == x.shape

# 测试残差连接
def test_residual_connection(base_config):
    layer = ByteDecoderLayer(base_config, layer_id=0)
    x = torch.randn(2, 16, base_config.model_dim)
    y = layer(x)
    # 确保输出与输入在相同数量级
    assert torch.allclose(x.mean(), y.mean(), atol=1e-1)

# 测试并行/顺序模式
@pytest.mark.parametrize("parallel", [True, False])
def test_residual_mode(base_config, parallel):
    config = ByteModelConfig(**{**base_config.to_dict(), "parallel_residual": parallel})
    layer = ByteDecoderLayer(config, layer_id=0)
    x = torch.randn(2, 16, config.model_dim)
    y = layer(x)
    assert not torch.isnan(y).any()

# 测试DropPath行为
def test_drop_path_training(base_config):
    layer = ByteDecoderLayer(base_config, layer_id=0)
    layer.train()
    x = torch.randn(2, 16, base_config.model_dim)
    y = layer(x)
    assert not torch.equal(x, y)  # 应有变化

def test_drop_path_inference(base_config):
    layer = ByteDecoderLayer(base_config, layer_id=0)
    layer.eval()
    x = torch.randn(2, 16, base_config.model_dim)
    y = layer(x)
    # 推理时DropPath应禁用
    assert not torch.isnan(y).any()

# 测试层深影响
def test_layer_depth_effect(base_config):
    outputs = []
    for layer_id in [0, 6, 11]:  # 首层、中间层、末层
        layer = ByteDecoderLayer(base_config, layer_id=layer_id)
        x = torch.randn(2, 16, base_config.model_dim)
        y = layer(x)
        outputs.append(y)
    
    # 确保不同层输出不同
    assert not torch.allclose(outputs[0], outputs[1], atol=1e-3)
    assert not torch.allclose(outputs[1], outputs[2], atol=1e-3)

# 测试数值稳定性
def test_numerical_stability(base_config):
    layer = ByteDecoderLayer(base_config, layer_id=0)
    
    # 极端值输入
    x = torch.full((2, 16, base_config.model_dim), 1e6)
    y = layer(x)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    
    # 极小值输入
    x = torch.full((2, 16, base_config.model_dim), 1e-6)
    y = layer(x)
    assert not torch.allclose(y, torch.zeros_like(y))

# 测试不同序列长度
@pytest.mark.parametrize("seq_len", [1, 16, 128, 1024])
def test_variable_sequence_length(base_config, seq_len):
    layer = ByteDecoderLayer(base_config, layer_id=0)
    x = torch.randn(2, seq_len, base_config.model_dim)
    y = layer(x)
    assert y.shape == (2, seq_len, base_config.model_dim)

# 测试梯度传播
def test_gradient_flow(base_config):
    layer = ByteDecoderLayer(base_config, layer_id=0)
    x = torch.randn(2, 16, base_config.model_dim, requires_grad=True)
    y = layer(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

# 测试配置参数影响
@pytest.mark.parametrize("param,value", [
    ("drop_path_prob", 0.5),
    ("layerscale_init", 1e-6),
    ("parallel_residual", False),
    ("use_flash_attention", True)
])
def test_config_parameters(base_config, param, value):
    config = ByteModelConfig(**{**base_config.to_dict(), param: value})
    layer = ByteDecoderLayer(config, layer_id=0)
    x = torch.randn(2, 16, config.model_dim)
    y = layer(x)
    assert y.requires_grad
