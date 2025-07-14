import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing      import Dict, Optional, Tuple


# ------------------------------------------------------- #
# 1. 配置结构体
# ------------------------------------------------------- #
@dataclass
class LoRAConfig:
    r: int = 8                     # 秩 (rank)
    alpha: int = 32                # 缩放因子
    dropout: float = 0.0           # LoRA dropout
    enable_lora: bool = True       # 便于总开关
    merge_weights: bool = False    # 初始化时是否立即合并（常用于推理）
    fan_in_fan_out: bool = False   # 为 True 时，W.T 作为权重（Conv用）
    dtype: torch.dtype = torch.float32

# ------------------------------------------------------- #
# 2. LoRALinear 模块（自带 merge / unmerge）
# ------------------------------------------------------- #
class LoRALinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_cfg: Optional[LoRAConfig] = None,
    ):
        # 原始 Linear
        super().__init__(in_features, out_features, bias)
        self.lora_cfg = lora_cfg or LoRAConfig()

        # LoRA 增量参数
        self.r = self.lora_cfg.r
        if self.r > 0 and self.lora_cfg.enable_lora:
            # 两个低秩矩阵
            self.lora_A = nn.Parameter(torch.zeros(self.r, in_features, dtype=self.lora_cfg.dtype))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.r, dtype=self.lora_cfg.dtype))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.scaling = self.lora_cfg.alpha / self.r
            # 可选 dropout
            self.lora_dropout = nn.Dropout(self.lora_cfg.dropout) if self.lora_cfg.dropout > 0 else nn.Identity()
        else:
            # 关闭 LoRA 时用占位 Identity，保持接口对齐
            self.lora_A = self.lora_B = None
            self.lora_dropout = nn.Identity()
            self.scaling = 0

        self.merged = self.lora_cfg.merge_weights

    # ---------------- 合并 / 分离 ---------------- #
    def _apply_lora_to_weight(self):
        """把 ΔW 合并到原权重，计算上更省时；推理可调用"""
        if self.r > 0 and not self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data += delta_w.to(self.weight.dtype)
            self.merged = True

    def _remove_lora_from_weight(self):
        """撤回 ΔW；方便继续训练或切换任务"""
        if self.r > 0 and self.merged:
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.weight.data -= delta_w.to(self.weight.dtype)
            self.merged = False

    # ---------------- 前向传播 ------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.r > 0 and not self.merged:
            # 原始输出
            result = nn.functional.linear(x, self.weight, self.bias)
            # 低秩增量
            lora_out = self.lora_dropout(x) @ self.lora_A.T       # [B,*,r]
            lora_out = lora_out @ self.lora_B.T * self.scaling    # [B,*,out]
            result = result + lora_out.to(result.dtype)
            return result
        else:
            # 已合并或未启用 LoRA
            return super().forward(x)

# ------------------------------------------------------- #
# 3. 工具函数：按名字模式注入 / 合并 / 分离
# ------------------------------------------------------- #
def inject_lora(model: nn.Module,
                target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),  # 例：Llama 结构
                lora_cfg: Optional[LoRAConfig] = None,
                verbose: bool = True):
    """
    遍历模型，把目标 Linear 替换为 LoRALinear
    target_modules: 层名字关键词；遇到则注入
    """
    lora_cfg = lora_cfg or LoRAConfig()
    for name, module in model.named_modules():
        if any(tm in name for tm in target_modules) and isinstance(module, nn.Linear):
            parent = _get_parent(model, name)
            child_name = name.split(".")[-1]
            lora_linear = LoRALinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                lora_cfg=lora_cfg,
            )
            # 复制原始权重 & bias
            lora_linear.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_linear.bias.data = module.bias.data.clone()
            # 用新模块替换
            setattr(parent, child_name, lora_linear)
            if verbose:
                print(f"[LoRA] Injected into {name}")

def merge_lora(model: nn.Module):
    """把所有 LoRA 合并到权重中"""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m._apply_lora_to_weight()

def unmerge_lora(model: nn.Module):
    """撤回合并，恢复可训练 LoRA"""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m._remove_lora_from_weight()

# —— 工具：找到父模块 —— #
def _get_parent(root: nn.Module, full_name: str) -> nn.Module:
    names = full_name.split(".")
    for n in names[:-1]:
        root = getattr(root, n)
    return root

# ------------------------------------------------------- #
# 4. LoRA 管理器（可选）
# ------------------------------------------------------- #
class LoRAManager:
    """
    用于服务端 / notebook 热插拔：
        mgr.register("zh-chat", cfg, target=("q_proj","v_proj"))
        mgr.activate("zh-chat", model)
        mgr.deactivate(model)
    """
    def __init__(self):
        self.registry: Dict[str, Tuple[Tuple[str, ...], LoRAConfig]] = {}
        self.active_tag: Optional[str] = None

    def register(self, tag: str,
                 lora_cfg: LoRAConfig,
                 target_modules: Tuple[str, ...]):
        self.registry[tag] = (target_modules, lora_cfg)

    def activate(self, tag: str, model: nn.Module):
        if self.active_tag:
            self.deactivate(model)
        target, cfg = self.registry[tag]
        inject_lora(model, target, cfg, verbose=False)
        self.active_tag = tag
        print(f"[LoRA] Activated → {tag}")

    def deactivate(self, model: nn.Module):
        if self.active_tag:
            unmerge_lora(model)          # 若已 merge 则撤回
            # remove modules? 这里简单不移除，防止 state_dict 丢失
            self.active_tag = None
            print("[LoRA] Deactivated current LoRA")
