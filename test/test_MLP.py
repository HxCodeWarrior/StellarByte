import sys
import torch
import pytest
from pathlib import Path

# 将项目根目录添加到sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from model.MLP import MLP
from model.config import ByteModelConfig

# 测试配置
@pytest.fixture
def config():
    return ByteModelConfig(
        model_dim=128,
        num_attention_heads=8,
        dim_multiplier=4,
        residual_dropout_prob=0.1
    )

# 测试设备兼容性
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_compatibility(device, config):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    mlp = MLP(config.model_dim).to(device)
    x = torch.randn(2, 16, config.model_dim).to(device)
    
    output = mlp(x)
    assert output.device.type == device
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

# 测试输出形状
def test_forward_shape(config):
    mlp = MLP(config.model_dim)
    x = torch.randn(2, 16, config.model_dim)
    
    output = mlp(x)
    assert output.shape == x.shape

# 测试数值稳定性
def test_numerical_stability(config):
    mlp = MLP(config.model_dim)
    
    # 测试大数值输入
    x_large = torch.randn(2, 16, config.model_dim) * 1e6
    output_large = mlp(x_large)
    assert not torch.isnan(output_large).any()
    assert not torch.isinf(output_large).any()
    
    # 测试小数值输入
    x_small = torch.randn(2, 16, config.model_dim) * 1e-6
    output_small = mlp(x_small)
    assert not torch.isnan(output_small).any()
    assert not torch.isinf(output_small).any()

# 测试dropout行为
def test_dropout_behavior(config):
    # 训练模式
    mlp_train = MLP(config.model_dim)
    mlp_train.train()
    x = torch.randn(2, 16, config.model_dim)
    output_train = mlp_train(x)
    
    # 评估模式
    mlp_eval = MLP(config.model_dim)
    mlp_eval.eval()
    output_eval = mlp_eval(x)
    
    # 训练模式输出应包含随机性
    assert not torch.allclose(output_train, output_eval, atol=1e-6)

# 测试梯度传播
def test_gradient_flow(config):
    mlp = MLP(config.model_dim)
    x = torch.randn(2, 16, config.model_dim, requires_grad=True)
    
    output = mlp(x)
    loss = output.sum()
    loss.backward()
    
    # 检查梯度是否存在
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()

# 测试参数更新
def test_parameter_update(config):
    mlp = MLP(config.model_dim)
    optimizer = torch.optim.SGD(mlp.parameters(), lr=0.01)
    
    # 获取初始参数
    initial_params = [p.clone() for p in mlp.parameters()]
    
    # 前向传播和反向传播
    x = torch.randn(2, 16, config.model_dim)
    output = mlp(x)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    
    # 检查参数是否更新
    for initial, updated in zip(initial_params, mlp.parameters()):
        assert not torch.allclose(initial, updated, atol=1e-6)

# 测试不同维度配置
@pytest.mark.parametrize("dim,hidden_dim", [
    (64, None),
    (128, 256),
    (256, 512),
    (512, 1024)
])
def test_dimension_config(dim, hidden_dim):
    mlp = MLP(dim, hidden_dim=hidden_dim)
    x = torch.randn(2, 16, dim)
    
    output = mlp(x)
    assert output.shape == x.shape

# 测试门控机制
def test_gate_mechanism(config):
    # 创建无偏置的MLP
    mlp = MLP(config.model_dim, bias=False)
    mlp.eval()  # 禁用dropout
    
    # 获取权重并固定（只设置权重，忽略偏置）
    with torch.no_grad():
        mlp.w1.weight.fill_(1.0)
        mlp.w2.weight.fill_(1.0)
        mlp.w3.weight.fill_(1.0)
    
    # 测试输入
    x = torch.ones(1, 1, config.model_dim)
    output = mlp(x)
    
    # 验证门控机制：a = SiLU(w1(x)), b = w3(x), output = w2(a * b)
    a = torch.nn.functional.silu(mlp.w1(x))
    b = mlp.w3(x)
    expected = mlp.w2(a * b)
    
    assert torch.allclose(output, expected, atol=1e-6)


# 测试不同序列长度
@pytest.mark.parametrize("seq_len", [1, 16, 64, 256])
def test_variable_sequence_length(seq_len, config):
    mlp = MLP(config.model_dim)
    x = torch.randn(2, seq_len, config.model_dim)
    
    output = mlp(x)
    assert output.shape == (2, seq_len, config.model_dim)

if __name__ == "__main__":
    pytest.main(["-v", __file__])