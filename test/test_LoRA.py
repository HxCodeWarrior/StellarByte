import sys
import re
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from model.utils.LoRA import (  # 替换成你的模块名
    LoRAConfig, 
    LoRALinear,
    LoRAManager, 
    inject_lora,
    merge_lora, 
    unmerge_lora, 
    lora_state_dict, 
    load_lora_state_dict,
    save_lora_state_dict
)

# ================================
# 1. LoRAConfig 测试
# ================================
def test_lora_config_default_and_validation():
    cfg = LoRAConfig()
    assert cfg.r == 8
    assert cfg.alpha == 32
    assert cfg.alpha % max(cfg.r, 1) == 0

    # r=0 禁用LoRA仍可用
    cfg2 = LoRAConfig(r=0, alpha=32)
    assert cfg2.r == 0

    # alpha必须能被r整除，否则断言异常
    with pytest.raises(AssertionError):
        LoRAConfig(r=3, alpha=10)

    # r不能为负
    with pytest.raises(AssertionError):
        LoRAConfig(r=-1)

# ================================
# 2. LoRALinear 初始化与参数形状
# ================================
@pytest.mark.parametrize("r,enable_lora", [(8, True), (0, True), (8, False)])
def test_loralinear_init_shapes(r, enable_lora):
    cfg = LoRAConfig(r=r, enable_lora=enable_lora)
    layer = LoRALinear(16, 32, bias=True, lora_cfg=cfg)

    # 原始权重形状
    assert layer.weight.shape == (32, 16)
    assert (layer.bias is not None) == True

    if r > 0 and enable_lora:
        assert layer.lora_A.shape == (r, 16)
        assert layer.lora_B.shape == (32, r)
        # dropout模块存在且是 nn.Identity 或 nn.Dropout
        assert isinstance(layer.lora_dropout, (nn.Identity, nn.Dropout))
        assert layer.scaling == cfg.alpha / r
    else:
        assert layer.lora_A is None
        assert layer.lora_B is None
        assert layer.scaling == 0.0

# ================================
# 3. LoRALinear forward 行为测试（合并与非合并）
# ================================
@pytest.mark.parametrize("training,merged", [(True, False), (False, False), (False, True)])
def test_loralinear_forward_shapes(training, merged):
    torch.manual_seed(42)
    cfg = LoRAConfig(r=4, alpha=8, dropout=0.1, enable_lora=True, merge_weights=False)
    layer = LoRALinear(10, 20, lora_cfg=cfg)
    layer.train(training)
    layer.merged = merged
    layer.active = True

    x = torch.randn(3, 10)  # [batch, in_features]

    y = layer(x)
    assert y.shape == (3, 20)
    # 前向结果dtype一致
    assert y.dtype == x.dtype or y.dtype == layer.weight.dtype

# ================================
# 4. 测试 merge / unmerge 正确性
# ================================
def test_merge_unmerge_effect():
    torch.manual_seed(1)
    cfg = LoRAConfig(r=2, alpha=4, enable_lora=True)
    layer = LoRALinear(5, 5, lora_cfg=cfg)

    # 复制合并前权重
    w_before = layer.weight.detach().clone()
    layer.merge()
    # merged标记应为True
    assert layer.merged is True
    # 权重应已更新（理论上有改动）
    assert not torch.allclose(layer.weight, w_before)
    # 再撤回
    layer.unmerge()
    assert layer.merged is False
    # 权重应回到原始
    assert torch.allclose(layer.weight, w_before)

# ================================
# 5. 测试 inject_lora 及替换行为
# ================================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(10, 10)
        self.v_proj = nn.Linear(10, 10)
        self.other = nn.Linear(10, 10)

def test_inject_lora_replaces_linear():
    model = SimpleModel()
    orig_q_weight = model.q_proj.weight.clone()
    inject_lora(model, target_modules=("q_proj",), verbose=False)
    # q_proj 被替换成 LoRALinear
    assert isinstance(model.q_proj, LoRALinear)
    # 权重复制正确
    assert torch.allclose(model.q_proj.weight, orig_q_weight)
    # v_proj 未被替换
    assert isinstance(model.v_proj, nn.Linear)
    # other 未被替换
    assert isinstance(model.other, nn.Linear)

# ================================
# 6. 测试 merge_lora 和 unmerge_lora 对整个模型生效
# ================================
def test_merge_unmerge_lora_global():
    model = SimpleModel()
    inject_lora(model, target_modules=("q_proj", "v_proj"), verbose=False)
    merge_lora(model)
    # 检查所有 LoRALinear 的 merged == True
    for m in model.modules():
        if isinstance(m, LoRALinear):
            assert m.merged is True

    unmerge_lora(model)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            assert m.merged is False

# ================================
# 7. 测试 lora_state_dict 仅导出 LoRA 参数
# ================================
def test_lora_state_dict_contains_only_lora_params():
    model = SimpleModel()
    inject_lora(model, target_modules=("q_proj",), verbose=False)
    sd = lora_state_dict(model)
    for k in sd.keys():
        assert ".lora_" in k

# ================================
# 8. 测试 load_lora_state_dict 加载行为
# ================================
def test_load_lora_state_dict_roundtrip(tmp_path):
    model = SimpleModel()
    inject_lora(model, target_modules=("q_proj",), verbose=False)

    # 保存并加载
    path = tmp_path / "lora.pth"
    save_lora_state_dict(model, path)

    model2 = SimpleModel()
    inject_lora(model2, target_modules=("q_proj",), verbose=False)
    load_lora_state_dict(model2, path, strict=False)

    # 两模型 lora_A 参数应相等
    lora_A_1 = model.q_proj.lora_A.detach()
    lora_A_2 = model2.q_proj.lora_A.detach()
    assert torch.allclose(lora_A_1, lora_A_2)

    # 加载严格失败示例（修改key名）
    sd = lora_state_dict(model)
    sd["wrong_key"] = torch.randn_like(lora_A_1)
    with pytest.raises(RuntimeError):
        load_lora_state_dict(model2, sd, strict=True)

# ================================
# 9. LoRAManager 功能测试
# ================================
def test_lora_manager_register_activate_deactivate():
    model = SimpleModel()
    mgr = LoRAManager()
    cfg = LoRAConfig(r=2, alpha=4)

    mgr.register("test", cfg, target_modules=("q_proj",))
    mgr.activate("test", model)
    assert isinstance(model.q_proj, LoRALinear)
    assert mgr.active_tag == "test"

    # 激活相同tag不重复注入
    mgr.activate("test", model)
    # 再注册并切换不同tag
    cfg2 = LoRAConfig(r=4, alpha=8)
    mgr.register("test2", cfg2, target_modules=("q_proj",))
    mgr.activate("test2", model)
    assert mgr.active_tag == "test2"

    # 撤销激活，恢复原始层
    mgr.deactivate(model)
    assert mgr.active_tag is None
    # q_proj 应回到 nn.Linear
    assert isinstance(model.q_proj, nn.Linear)

# ================================
# 10. 边界测试：r=0 禁用 LoRA 时行为
# ================================
def test_disabled_lora_forward_behavior():
    cfg = LoRAConfig(r=0, alpha=32, enable_lora=True)
    layer = LoRALinear(10, 10, lora_cfg=cfg)
    x = torch.randn(4, 10)
    y = layer(x)
    # 输出应等同普通线性层
    assert y.shape == (4, 10)

# ================================
# 11. 输入异常测试
# ================================
def test_forward_input_shape_error():
    cfg = LoRAConfig()
    layer = LoRALinear(10, 10, lora_cfg=cfg)
    x_wrong = torch.randn(4)  # 一维输入

    with pytest.raises(ValueError):
        layer.forward(x_wrong)

# ================================
# 12. fan_in_fan_out=True 行为测试
# ================================
def test_fan_in_fan_out_behavior():
    cfg = LoRAConfig(fan_in_fan_out=True)
    layer = LoRALinear(10, 10, lora_cfg=cfg)
    x = torch.randn(2, 10)
    y = layer(x)
    assert y.shape == (2, 10)

# ================================
# 13. dropout效果测试（不直接测试随机性，测试模块类型）
# ================================
def test_dropout_module_type():
    cfg0 = LoRAConfig(dropout=0.0)
    layer0 = LoRALinear(10, 10, lora_cfg=cfg0)
    assert isinstance(layer0.lora_dropout, nn.Identity)

    cfg1 = LoRAConfig(dropout=0.5)
    layer1 = LoRALinear(10, 10, lora_cfg=cfg1)
    assert isinstance(layer1.lora_dropout, nn.Dropout)

if __name__ == "__main__":
    pytest.main(["-v", __file__])