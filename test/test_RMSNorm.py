import sys
from pathlib import Path

# 将项目根目录添加到sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from model.config import ByteModelConfig
from model.RMSNorm import RMSNorm
import torch
import torch.nn as nn
import pytest

class TestRMSNorm:
    @pytest.fixture
    def config(self):
        """创建模型配置fixture"""
        return ByteModelConfig(
            model_dim=768,
            layer_norm_eps=1e-5
        )
    
    @pytest.fixture
    def rms_norm(self, config):
        """创建RMSNorm实例fixture"""
        return RMSNorm(config.model_dim, config.layer_norm_eps)
    
    def test_forward_shape(self, rms_norm):
        """测试输出形状是否正确"""
        # 创建测试输入 (batch_size, seq_len, model_dim)
        x = torch.randn(2, 128, 768)
        output = rms_norm(x)
        assert output.shape == x.shape
    
    def test_forward_values(self, rms_norm):
        """测试归一化值是否正确"""
        # 创建全1输入
        x = torch.ones(1, 1, 768)
        output = rms_norm(x)
        
        # 验证RMS值计算正确
        expected_rms = torch.sqrt(torch.tensor(1.0))
        actual_rms = torch.sqrt(torch.mean(output.pow(2)))
        assert torch.allclose(actual_rms, expected_rms, atol=1e-6)
    
    def test_parameter_update(self, rms_norm):
        """测试参数是否在训练中更新"""
        optimizer = torch.optim.SGD(rms_norm.parameters(), lr=0.1)
        
        # 保存初始权重
        initial_weight = rms_norm.weight.data.clone()
        
        # 模拟训练步骤
        x = torch.randn(1, 10, 768)
        output = rms_norm(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # 验证权重已更新
        assert not torch.equal(rms_norm.weight.data, initial_weight)
    
    def test_eps_handling(self, config):
        """测试极小值处理"""
        # 使用极小的eps值
        rms_norm = RMSNorm(config.model_dim, eps=1e-10)
        
        # 创建接近零的输入
        x = torch.full((1, 1, 768), 1e-8)
        output = rms_norm(x)
        
        # 验证没有出现NaN或inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_different_dtypes(self, rms_norm):
        """测试不同数据类型支持"""
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(2, 64, 768, dtype=dtype)
            output = rms_norm(x)
            assert output.dtype == dtype
    
    def test_zero_input(self, rms_norm):
        """测试全零输入处理"""
        x = torch.zeros(1, 10, 768)
        output = rms_norm(x)
        
        # 验证输出全为零
        assert torch.all(output == 0)
    
    def test_gradient_flow(self, rms_norm):
        """测试梯度是否正确传播"""
        x = torch.randn(1, 5, 768, requires_grad=True)
        output = rms_norm(x)
        
        # 创建虚拟损失
        loss = output.sum()
        loss.backward()
        
        # 验证输入梯度存在
        assert x.grad is not None
        assert not torch.all(x.grad == 0)
        
        # 验证权重梯度存在
        assert rms_norm.weight.grad is not None
        assert not torch.all(rms_norm.weight.grad == 0)

if __name__ == "__main__":
    pytest.main(["-v", "-s", "./test/test_RMSNorm.py"])