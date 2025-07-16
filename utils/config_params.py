# ======================================================================
# 文件: config_utils.py
# 功能: 读取 YAML → argparse.Namespace，并支持 CLI 覆盖，
#      同时生成「层次化访问」和「旧版扁平字段」两套接口
# 用法:
#   from config_utils import load_config
#   cfg = load_config("pretrain_config.yaml")  # cfg 即原先的 args
# ======================================================================
import sys
import yaml
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
from typing import Any

class DeepNamespace(SimpleNamespace):
    """
    扩展 SimpleNamespace：
      • 读取不存在的属性时，递归到所有子 DeepNamespace 中查找。
      • 若仍找不到，正常抛出 AttributeError。
    写入（__setattr__）逻辑保持默认 —— 只写当前节点。
    """
    def __getattr__(self, name):
        # 先按正常流程取
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass

        # 递归向下查找
        for v in self.__dict__.values():
            if isinstance(v, DeepNamespace):
                try:
                    return getattr(v, name)
                except AttributeError:
                    continue
        # 仍找不到 → 抛错
        raise AttributeError(f"{name} not found in DeepNamespace hierarchy")

# ---------- 基础工具 ----------
def _dict_to_ns(d: dict[str, Any]) -> DeepNamespace:
    ns = DeepNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_ns(v) if isinstance(v, dict) else v)
    return ns

def _register_cli(flat: dict[str, Any], cli_args: list[str] | None):
    """把 YAML 默认值注册给 argparse，并让 CLI 可以覆盖"""
    parser = ArgumentParser(add_help=False)
    for key, default in flat.items():
        if isinstance(default, bool):
            parser.add_argument(f"--{key}", action="store_true" if not default else "store_false")
        else:
            parser.add_argument(f"--{key}", type=type(default), default=None)
    return parser.parse_known_args(cli_args)[0]

# ---------- 关键：把所有叶子写到根 ----------
def _bubble_up(node: Namespace, root: Namespace):
    for k, v in vars(node).items():
        if isinstance(v, Namespace):
            _bubble_up(v, root)
        else:                               # 叶子节点
            setattr(root, k, v)             # 写到顶层

def load_config(yaml_path: str, cli_args: list[str] | None = None) -> Namespace:
    """
    1. 读取 YAML → Namespace（分层）
    2. 把所有叶子同步到顶层 → 兼容旧代码
    3. 解析 CLI，覆盖同名字段
    4. 注入常用别名（ddp / device / local_rank）
    """
    # ------- 1. YAML -------
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    args = _dict_to_ns(cfg_dict)

    # ------- 2. CLI 覆盖 -------
    #   先构造 “默认值扁平表” 供 parser 用
    flat_defaults = {}
    def _collect(ns: Namespace, prefix=""):
        for k, v in vars(ns).items():
            if isinstance(v, Namespace):
                _collect(v, prefix)
            else:
                flat_defaults[f"{prefix}{k}"] = v
    _collect(args)

    cli_ns = _register_cli(flat_defaults, cli_args)
    for k, v in vars(cli_ns).items():
        if v is None:
            continue
        cursor, *rest = k.split("_")
        target = args
        # 只支持一层前缀：<group>_<name>
        if hasattr(target, cursor) and isinstance(getattr(target, cursor), Namespace):
            setattr(getattr(target, cursor), "_".join(rest), v)
        else:          # 找不到分组就直接写顶层
            setattr(args, k, v)

    # ------- 3. 同步叶子到顶层 -------
    _bubble_up(args, args)

    # ------- 4. 别名 -------
    # args.ddp        = args.distributed.enable_ddp
    # args.device     = args.distributed.device
    # args.local_rank = args.distributed.local_rank

    return args

