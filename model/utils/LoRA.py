import math
import re
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. LoRA 配置结构体
# =============================================================================
@dataclass
class LoRAConfig:
    """存放 LoRA 相关超参数与开关的配置类"""

    # ---------------- 超参数 ---------------- #
    r: int         = 8               # 低秩分解的秩 rank；r=0 表示禁用 LoRA
    alpha: int     = 32              # 缩放因子，等效于论文中的 \alpha
    dropout: float = 0.0             # LoRA 路径上的 Dropout，比率越高越正则化

    # ---------------- 行为开关 ---------------- #
    enable_lora: bool    = True      # 全局开关；False 时等价于普通 Linear
    merge_weights: bool  = False     # 推理时是否自动将 ΔW 合并到 W
    fan_in_fan_out: bool = False     # 卷积或转置线性层权重布局差异
    freeze_base: bool    = True      # 训练时是否冻结原始权重，仅训练 LoRA 参数

    # ---------------- 其它 ---------------- #
    dtype: torch.dtype = torch.float32  # LoRA 参数的数据类型

    # 自动检查：在实例化完成后执行
    def __post_init__(self):
        assert self.r >= 0, "rank r 必须非负"
        # 要让 alpha 可被 r 整除，这样 scaling=alpha/r 为整数更稳定
        assert self.alpha % max(self.r, 1) == 0, "alpha 应能被 r 整除"

# =============================================================================
# 2. LoRALinear 模块
# =============================================================================
class LoRALinear(nn.Module):
    """带 LoRA 适配器的线性层，可选择在推理阶段合并权重"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lora_cfg: Optional[LoRAConfig] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # 保存超参数与维度信息
        self.cfg          = lora_cfg or LoRAConfig()       # 若未显式传入则用默认配置
        self.in_features  = in_features                    # 输入维度
        self.out_features = out_features                   # 输出维度
        self.device       = device or torch.device("cpu")  # 默认在 CPU 上初始化
        self.dtype        = self.cfg.dtype or torch.float32

        # ---------------- 原始权重 W, b ---------------- #
        # 注意：此处不直接调用 nn.Linear，而是手动创建 weight/bias，
        # 便于灵活控制 requires_grad 与权重布局。
        self.weight  = nn.Parameter(torch.empty(out_features, in_features, device=self.device, dtype=self.dtype))
        self.bias    = nn.Parameter(torch.empty(out_features, device=self.device, dtype=self.dtype)) if bias else None
        self.reset_parameters()  # 初始化权重

        # ---------------- LoRA 增量参数 ---------------- #
        self.r = self.cfg.r
        if self.r > 0 and self.cfg.enable_lora:
            # A: [r, in_features]   B: [out_features, r]
            self.lora_A = nn.Parameter(
                torch.zeros(self.r, in_features, dtype=self.dtype, device=self.device, requires_grad=True)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, self.r, dtype=self.dtype, device=self.device, requires_grad=True)
            )
            # 初始化：A/B 用 Kaiming，B 用零(论文推荐)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

            # LoRA 缩放系数；scaling = alpha / r
            self.scaling: float = self.cfg.alpha / self.r

            # Dropout 模块：若 rate=0 则用 Identity 节省开销
            self.lora_dropout = (
                nn.Dropout(self.cfg.dropout) if self.cfg.dropout > 0 else nn.Identity()
            )

            # 可选：冻结原始权重以节省显存与梯度计算
            if self.cfg.freeze_base:
                self.weight.requires_grad_(False)
                if self.bias is not None:
                    self.bias.requires_grad_(False)
        else:
            # LoRA 被禁用：占位符，保持接口一致
            self.lora_A       = None
            self.lora_B       = None
            self.lora_dropout = nn.Identity()
            self.scaling      = 0.0

        # merged=True 表示 ΔW 已经加到 W 上，无需再计算 LoRA 路径
        self.merged: bool = False                         # ΔW 是否已并入 W
        self.auto_merge_enabled = self.cfg.merge_weights  # 是否允许自动合并
        self.active: bool  = True                         # 是否启用 LoRA 分支

        # 线程安全锁
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 参数初始化：仿 nn.Linear 默认实现
    # ------------------------------------------------------------------
    def reset_parameters(self):
        # 使用 Kaiming 均匀分布初始化 weight
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # bias 按 fan_in 反比初始化
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # ------------------------------------------------------------------
    # 语法糖：打印模型时能显示 LoRA 配置
    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        base = f"{self.in_features}->{self.out_features}, bias={self.bias is not None}"
        if self.r > 0 and self.cfg.enable_lora:
            return base + f", LoRA(r={self.r}, α={self.cfg.alpha}, dropout={self.cfg.dropout})"
        return base

    # ------------------------------------------------------------------
    # 权重合并：将 ΔW = B @ A * scaling 累加到 W
    # ------------------------------------------------------------------
    @torch.no_grad()
    def merge(self):
        """在推理场景下调用：合并一次后可省去增量路径计算(线程安全)"""
        if self.r == 0 or self.merged:
            return  # 无 LoRA 或已合并直接返回

        delta_w = (self.lora_B @ self.lora_A) * self.scaling # 计算 ΔW
        # 若 fan_in_fan_out=True（例如 conv 转置权重），需转置
        if self.cfg.fan_in_fan_out:
            delta_w = delta_w.T
        with self._lock:  # 锁住合并操作，避免推理中并发写入
            # 将增量累加到原始权重
            self.weight.data += delta_w.to(self.weight.dtype)
            self.merged = True
            self.auto_merge_enabled = True

    # ------------------------------------------------------------------
    # 权重撤回：将之前合并的 ΔW 再减回去
    # ------------------------------------------------------------------
    @torch.no_grad()
    def unmerge(self):
        if self.r == 0 or not self.merged:
            return

        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        if self.cfg.fan_in_fan_out:
            delta_w = delta_w.T

        with self._lock:  # 锁住撤销合并操作，避免推理中并发写入
            self.weight.data -= delta_w.to(self.weight.dtype)
            self.merged = False
            self.auto_merge_enabled = False # 用户显式 unmerge ⇒ 禁掉后续自动合并

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入尺寸与 dtype 校验
        if x.dim() < 2 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"LoRALinear expected input shape [..., {self.in_features}], "
                f"got {tuple(x.shape)}"
            )
        # ★ 性能优化：若处于 eval 模式且 cfg.merge_weights=True，
        #   则自动先合并一次权重，后续前向都走“快捷路径”。
        if (not self.training
            and self.auto_merge_enabled
            and not self.merged):
            # 若处于推理阶段且未合并，则自动合并
            if not self.merged:
                self.merge()
        if self.r > 0 and self.active and not self.merged:
            # 主路径输出
            result = F.linear(x, self.weight, self.bias)
            # 情况 1：存在 LoRA 且未合并，需要计算增量
            if self.cfg.fan_in_fan_out:
                # 1) # 如果是 fan_in_fan_out 布局，LoRA 参数需要转置后使用
                lora_A = self.lora_A.T  # [in_features, r]
                lora_B = self.lora_B.T  # [r, out_features]
                # 2) LoRA 路径输出 = x * A^T -> [B, *, r]
                lora_out = self.lora_dropout(x) @ lora_A
                # 3) 再乘 B^T 得到 [B, *, out_features] 并缩放
                lora_out = lora_out @ lora_B
                # 4) 缩放
                lora_out = lora_out * self.scaling
            else:
                # 情况 2：已合并或 LoRA 被禁用，走普通线性层
                lora_out = F.linear(self.lora_dropout(x), self.lora_A)  # [batch, ..., r]
                lora_out = F.linear(lora_out, self.lora_B) * self.scaling  # [batch, ..., out_features]
            
            # 将 LoRA 输出与主路径输出相加
            return result + lora_out.to(result.dtype)

        return F.linear(x, self.weight, self.bias)


# =============================================================================
# 3. LoRA 注入与辅助函数
# =============================================================================
# 类型别名：目标模块匹配可用字符串或 regex Pattern
RegexPattern = Union[str, re.Pattern]

def _should_replace(name: str, patterns: Iterable[RegexPattern]) -> bool:
    """判断模块名是否匹配目标模式"""
    for p in patterns:
        if isinstance(p, re.Pattern):  # regex
            if p.search(name):
                return True
        elif p in name:               # 普通子串
            return True
    return False

def inject_lora(
    model: nn.Module,
    target_modules: Tuple[RegexPattern, ...] = (r"q_proj", r"v_proj"),
    lora_cfg: Optional[LoRAConfig] = None,
    verbose: bool = True,
):
    """递归遍历模型，将匹配到的 nn.Linear 替换为 LoRALinear"""
    lora_cfg = lora_cfg or LoRAConfig()

    # 先把 named_modules 抓成 list，避免遍历时结构变化
    for name, module in list(model.named_modules()):
        if not _should_replace(name, target_modules):
            continue

        # --------- 跳过已注入层（防止重复注入） ---------
        if isinstance(module, LoRALinear):
            if verbose:
                print(f"[LoRA] Skipped (already LoRALinear) → {name}")
            continue

        if not isinstance(module, nn.Linear):
            continue

        # --------- 创建 LoRALinear 并复制权重 ---------
        parent = _get_parent(model, name)
        child_name = name.split(".")[-1]

        new_layer = LoRALinear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            lora_cfg=lora_cfg,
        ).to(module.weight.device, dtype=module.weight.dtype)

        new_layer.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            new_layer.bias.data.copy_(module.bias.data)

        # 保存原层指针，方便 LoRAManager 撤销
        new_layer._orig_module = module

        # --------- 真正替换 ---------
        setattr(parent, child_name, new_layer)
        if verbose:
            print(f"[LoRA] Injected → {name}")



# ---------------- 工具：获取父模块 ---------------- #
def _get_parent(root: nn.Module, full_name: str) -> nn.Module:
    """根据 full_name 逐层向下获取父模块，对于顶层直接返回 root"""
    for n in full_name.split(".")[:-1]:
        root = getattr(root, n)
    return root


# ---------------- 合并 / 撤回（全局一次性） ---------------- #
def merge_lora(model: nn.Module):
    """遍历模型，调用每层的 merge()"""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_lora(model: nn.Module):
    """遍历模型，调用每层的 unmerge()"""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()


# ---------------- 轻量级保存 / 加载 LoRA 参数 ---------------- #
def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Function: 只加载 *.lora_* 参数/权重。
    """
    return {k: v for k, v in model.state_dict().items() if ".lora_" in k}


def save_lora_state_dict(model: nn.Module, path: str):
    torch.save(lora_state_dict(model), path)

def _safe_torch_load(path: str):
    """
    尝试使用 weights_only=True 加载；若旧版本 PyTorch 不支持该参数，则退回。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:                         # < 2.3 版本没有该参数
        return torch.load(path, map_location="cpu")

def load_lora_state_dict(
    model: nn.Module, 
    obj: Union[str, Dict[str, torch.Tensor]], 
    strict: bool = True
):
    """加载轻量级 LoRA 参数，可用于推理或继续训练
     支持两种调用：
         • load_lora_state_dict(model, "lora.bin")
         • load_lora_state_dict(model, state_dict)
         Function: 只加载 *.lora_* 参数/权重。
    Params: obj, str 或 dict,
    Params: strict, 是否严格匹配，若为 False 则忽略未匹配的键,只加载参数；若为True则抛出异常。
    Return: dict, 包含所有 *.lora_* 参数/权重
     """
    sd = obj if isinstance(obj, dict) else _safe_torch_load(obj)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Load mismatch: missing={missing}, unexpected={unexpected}"
        )



# =============================================================================
# 4. LoRA Manager：多适配器热切换
# =============================================================================
class LoRAManager:
    """\n    用法示例::

        mgr = LoRAManager()
        mgr.register("zh", cfg_zh, target_modules=("q_proj", "v_proj"))
        mgr.register("en", cfg_en, target_modules=("q_proj", "v_proj"))

        # 切换到中文 LoRA
        mgr.activate("zh", model)

        # 取消当前 LoRA（恢复原始权重）
        mgr.deactivate(model)
    """

    def __init__(self):
        self.registry: Dict[str, Tuple[Tuple[RegexPattern, ...], LoRAConfig]] = {}
        self.active_tag: Optional[str] = None

    # ---------------- 注册 ---------------- #
    def register(
        self,
        tag: str,
        lora_cfg: LoRAConfig,
        target_modules: Tuple[RegexPattern, ...],
    ):
        """将一组 LoRA 配置注册到管理器，方便后续 activate"""
        self.registry[tag] = (target_modules, lora_cfg)

    # ---------------- 激活 ---------------- #
    def activate(self, tag: str, model: nn.Module):
        """激活指定 tag 的 LoRA；若已有激活则先撤销"""
        if self.active_tag == tag:
            return  # 已经是该 LoRA，无需重复切换
        self.deactivate(model)       # 确保彻底清理旧 LoRA
        tgt, cfg = self.registry[tag]
        inject_lora(model, tgt, cfg, verbose=False)
        # 打开所有 LoRA 分支
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.active = True
        self.active_tag = tag
        print(f"[LoRA] ► Activated '{tag}'")

    # ---------------- 撤销 ---------------- #
    def deactivate(self, model: nn.Module):
        """撤销当前激活的 LoRA 并恢复原始权重"""
        if not self.active_tag:
             return
        # 1) 撤回已合并的权重
        unmerge_lora(model)
        # 2) 彻底关闭 LoRA 分支
        for name, module in list(model.named_modules()):
            if isinstance(module, LoRALinear) and hasattr(module, "_orig_module"):
                parent = _get_parent(model, name)
                child_name = name.split(".")[-1]
                setattr(parent, child_name, module._orig_module)
                del module._orig_module
        self.active_tag = None
        print("[LoRA] ◄ Deactivated")
