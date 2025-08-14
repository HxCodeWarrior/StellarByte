"""工业级KV缓存实现，用于基于Transformer的大型语言模型。

本模块实现了一个高效的键值(KV)缓存系统，支持流式生成、束搜索和分布式训练。
"""

from typing import List, Optional, Dict, Tuple
import torch
import os


class ByteKVCache:
    """Transformer注意力层的预分配键值缓存。
    
    设计特点：
    - 每层预分配固定形状的缓冲区，避免自回归解码中的重复内存分配
    - 支持单步追加和块追加操作
    - 提供束搜索重排序、剪枝和设备管理功能
    - 支持状态保存/加载和分布式分片操作

    参数：
        num_layers: 需要KV缓存的Transformer层数
        num_heads: 注意力头数量
        head_dim: 每个注意力头的维度
        max_seq_len: 最大序列长度（缓存容量）
        batch_size: 支持的最大批处理大小
        dtype: 张量数据类型（如torch.float16）
        device: 缓存存储设备
        memory_format: 内存布局格式（默认为连续格式）
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch_size: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        memory_format: Optional[torch.memory_format] = None,
    ):
        # 初始化缓存参数
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.memory_format = memory_format or torch.contiguous_format

        # 每层的缓存数据结构
        self._layers: List[Dict[str, torch.Tensor]] = []
        self._init_buffers()

    def _init_buffers(self) -> None:
        """为每层初始化键/值缓存空间。"""
        self._layers = []
        for _ in range(self.num_layers):
            # 创建未初始化的键/值张量
            k = torch.empty(
                (self.batch_size, self.num_heads, self.max_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            v = torch.empty(
                (self.batch_size, self.num_heads, self.max_seq_len, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            # 确保内存布局符合要求
            k = k.contiguous(memory_format=self.memory_format)
            v = v.contiguous(memory_format=self.memory_format)
            # 存储缓存和当前序列长度
            self._layers.append({"k": k, "v": v, "seq_len": 0})

    # ------------------------ 基本访问方法 ------------------------
    def get_device(self) -> torch.device:
        """返回缓存所在的设备。"""
        return self.device

    def get_dtype(self) -> torch.dtype:
        """返回缓存的数据类型。"""
        return self.dtype

    def current_seq_len(self, layer_idx: int) -> int:
        """获取指定层当前缓存的序列长度。"""
        return int(self._layers[layer_idx]["seq_len"])

    def capacity(self) -> int:
        """返回缓存的最大容量。"""
        return self.max_seq_len

    # ------------------------ 数据写入操作 ------------------------
    def append_batch(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
    ) -> int:
        """向指定层的缓存追加键/值数据块。
        
        参数：
            layer_idx: 目标层索引
            k: 键张量 [B, num_heads, L_block, head_dim]
            v: 值张量 [B, num_heads, L_block, head_dim]
            batch_index: 批处理索引张量 [B]
        
        返回：
            更新后的序列长度
        
        异常：
            RuntimeError: 当追加后超出缓存容量时抛出
        """
        # 验证层索引有效性
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError("层索引超出范围")
        layer = self._layers[layer_idx]
        
        # 验证输入形状
        batch_size, num_heads, block_len, head_dim = k.shape
        if num_heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError("输入张量维度与缓存不匹配")
        
        # 处理默认批处理索引
        if batch_index is None:
            batch_index = torch.arange(batch_size, dtype=torch.long, device=k.device)
        elif batch_index.shape[0] != batch_size:
            raise ValueError("批处理索引维度不匹配")
        
        # 检查缓存容量
        current_len = int(layer["seq_len"])
        new_len = current_len + block_len
        if new_len > self.max_seq_len:
            raise RuntimeError(
                f"缓存溢出: 当前长度={current_len}, 追加块={block_len}, 最大容量={self.max_seq_len}"
            )
        
        # 转换设备/数据类型
        k = k.to(device=layer["k"].device, dtype=self.dtype)
        v = v.to(device=layer["v"].device, dtype=self.dtype)
        
        # 获取目标存储位置
        k_dest = layer["k"][batch_index, :, current_len:new_len, :]
        v_dest = layer["v"][batch_index, :, current_len:new_len, :]
        
        # 执行数据复制
        k_dest.copy_(k)
        v_dest.copy_(v)
        
        # 更新序列长度
        layer["seq_len"] = new_len
        return new_len

    def set_slice(
        self,
        layer_idx: int,
        start: int,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_index: Optional[torch.Tensor] = None,
    ) -> None:
        """在指定位置写入键/值数据块（覆盖已有数据）。
        
        参数：
            layer_idx: 目标层索引
            start: 写入起始位置
            k: 键张量 [B, num_heads, L_block, head_dim]
            v: 值张量 [B, num_heads, L_block, head_dim]
            batch_index: 批处理索引张量 [B]
        """
        # 验证参数有效性
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError("层索引超出范围")
        if start < 0:
            raise ValueError("起始位置不能为负")
        
        layer = self._layers[layer_idx]
        batch_size, num_heads, block_len, head_dim = k.shape
        
        # 处理批处理索引
        if batch_index is None:
            batch_index = torch.arange(batch_size, dtype=torch.long, device=k.device)
        
        # 检查写入边界
        end_pos = start + block_len
        if end_pos > self.max_seq_len:
            raise RuntimeError("写入位置超出缓存边界")
        
        # 转换设备/数据类型
        k = k.to(device=layer["k"].device, dtype=self.dtype)
        v = v.to(device=layer["v"].device, dtype=self.dtype)
        
        # 执行数据写入
        layer["k"][batch_index, :, start:end_pos, :].copy_(k)
        layer["v"][batch_index, :, start:end_pos, :].copy_(v)
        
        # 更新序列长度（如有必要）
        if end_pos > layer["seq_len"]:
            layer["seq_len"] = end_pos

    # ------------------------ 数据读取操作 ------------------------
    def get_kv(
        self, 
        layer_idx: int, 
        max_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定层缓存的键/值对。
        
        参数：
            layer_idx: 目标层索引
            max_len: 返回的最大序列长度（默认到当前长度）
        
        返回：
            (k, v) 元组，形状为 [batch, num_heads, seq_len, head_dim]
        """
        layer = self._layers[layer_idx]
        current_len = int(layer["seq_len"])
        if current_len == 0:
            # 返回空张量
            return torch.empty(0), torch.empty(0)
        seq_len = current_len if max_len is None else min(max_len, current_len)
        
        if seq_len > current_len:
            raise ValueError("请求长度超过当前缓存长度")
        
        k = layer["k"][:, :, :seq_len, :]
        v = layer["v"][:, :, :seq_len, :]
        return k, v

    # ------------------------ 缓存管理操作 ------------------------
    def reorder(self, new_order: torch.Tensor) -> None:
        """按新顺序重排缓存数据（用于束搜索）。
        
        参数：
            new_order: 新顺序索引张量 [batch_size]
        """
        if new_order.dtype != torch.long:
            raise TypeError("重排索引必须是长整型")
        
        batch_size = new_order.shape[0]
        if batch_size > self.batch_size:
            raise ValueError("重排批处理大小超过缓存容量")
        
        for layer in self._layers:
            seq_len = int(layer["seq_len"])
            if seq_len == 0:
                continue
                
            # 重排键/值数据
            layer["k"][:batch_size] = layer["k"][new_order]
            layer["v"][:batch_size] = layer["v"][new_order]

    def prune(self, new_max_len: int) -> None:
        """剪枝缓存到指定长度。
        
        参数：
            new_max_len: 新的最大序列长度
        """
        if not 0 < new_max_len <= self.max_seq_len:
            raise ValueError("新长度必须在(0, 当前最大长度]范围内")
        
        # 更新每层序列长度
        for layer in self._layers:
            if layer["seq_len"] > new_max_len:
                layer["seq_len"] = new_max_len
        
        # 更新全局最大长度
        self.max_seq_len = new_max_len

    # ------------------------ 设备管理 ------------------------
    def to(self, 
           device: torch.device, 
           dtype: Optional[torch.dtype] = None, 
           non_blocking: bool = False) -> "ByteKVCache":
        """移动缓存到新设备和/或更改数据类型。
        
        参数：
            device: 目标设备
            dtype: 目标数据类型（可选）
            non_blocking: 是否使用非阻塞传输
        
        返回：
            self（支持链式调用）
        """
        dtype = dtype or self.dtype
        device = torch.device(device)
        
        # 迁移每层数据
        for layer in self._layers:
            layer["k"] = layer["k"].to(device, dtype, non_blocking=non_blocking)
            layer["v"] = layer["v"].to(device, dtype, non_blocking=non_blocking)
        
        # 更新元数据
        self.device = device
        self.dtype = dtype
        return self

    def cpu_offload(self) -> "ByteKVCache":
        """将缓存移动到CPU（释放GPU内存）。
        
        返回：
            self（支持链式调用）
        """
        return self.to(device=torch.device("cpu"))

    # ------------------------ 序列化操作 ------------------------
    def state_dict(self) -> Dict:
        """生成可序列化的状态字典（仅包含有效数据）。
        
        返回：
            包含缓存元数据和有效数据的字典
        """
        state = {
            "meta": {
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "max_seq_len": self.max_seq_len,
                "batch_size": self.batch_size,
                "dtype": str(self.dtype),
            },
            "layers": [],
        }
        
        # 保存每层有效数据（非整个缓存）
        for layer in self._layers:
            seq_len = int(layer["seq_len"])
            state["layers"].append({
                "seq_len": seq_len,
                "k": layer["k"][:, :, :seq_len, :].cpu(),
                "v": layer["v"][:, :, :seq_len, :].cpu(),
            })
            
        return state

    def load_state_dict(self, state: Dict, strict: bool = True) -> None:
        """从状态字典加载缓存数据。
        
        参数：
            state: 状态字典
            strict: 是否严格检查维度一致性
        
        异常：
            RuntimeError: 当加载的数据与缓存不兼容时
        """
        meta = state.get("meta", {})
        
        # 严格模式下的维度检查
        if strict:
            for param in ["num_layers", "num_heads", "head_dim"]:
                if meta.get(param) != getattr(self, param):
                    raise RuntimeError(f"{param}不匹配")
        
        # 逐层加载数据
        for idx, layer_state in enumerate(state["layers"]):
            if idx >= self.num_layers:
                break
                
            seq_len = int(layer_state["seq_len"])
            if seq_len > self.max_seq_len:
                raise RuntimeError("加载的序列长度超过缓存容量")
            
            # 迁移数据到当前设备
            k = layer_state["k"].to(device=self.device, dtype=self.dtype)
            v = layer_state["v"].to(device=self.device, dtype=self.dtype)
            
            # 写入缓存
            self.set_slice(idx, 0, k, v)
            self._layers[idx]["seq_len"] = seq_len

    def save(self, path: str) -> None:
        """保存缓存状态到文件。
        
        参数：
            path: 文件保存路径
        """
        # 获取目录路径
        dir_path = os.path.dirname(path)
        
        # 仅在目录路径非空时创建目录
        if dir_path and dir_path.strip():
            os.makedirs(dir_path, exist_ok=True)
        
        # 保存状态字典
        torch.save(self.state_dict(), path)

    def load(self, path: str, strict: bool = True) -> None:
        """从文件加载缓存状态。
        
        参数：
            path: 文件路径
            strict: 是否严格检查维度
        """
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state, strict=strict)

    # ------------------------ 分布式操作 ------------------------
    def split_for_model_parallel(self, shards: int) -> List["ByteKVCache"]:
        """将缓存分片用于模型并行。
        
        参数：
            shards: 分片数量
        
        返回：
            分片后的缓存列表
        
        异常：
            ValueError: 当注意力头数不能被分片数整除时
        """
        if self.num_heads % shards != 0:
            raise ValueError("注意力头数必须能被分片数整除")
        
        heads_per_shard = self.num_heads // shards
        cache_shards = []
        
        for shard_idx in range(shards):
            # 创建新分片缓存
            shard = ByteKVCache(
                num_layers=self.num_layers,
                num_heads=heads_per_shard,
                head_dim=self.head_dim,
                max_seq_len=self.max_seq_len,
                batch_size=self.batch_size,
                dtype=self.dtype,
                device=self.device,
            )
            
            # 复制数据子集
            start_idx = shard_idx * heads_per_shard
            end_idx = (shard_idx + 1) * heads_per_shard
            
            for layer_idx in range(self.num_layers):
                seq_len = int(self._layers[layer_idx]["seq_len"])
                shard._layers[layer_idx]["k"].copy_(
                    self._layers[layer_idx]["k"][:, start_idx:end_idx, :seq_len, :]
                )
                shard._layers[layer_idx]["v"].copy_(
                    self._layers[layer_idx]["v"][:, start_idx:end_idx, :seq_len, :]
                )
                shard._layers[layer_idx]["seq_len"] = seq_len
            
            cache_shards.append(shard)
        
        return cache_shards

    def merge_from_shards(self, shards: List["ByteKVCache"]) -> None:
        """合并模型并行分片。
        
        参数：
            shards: 要合并的缓存分片列表
        """
        total_heads = sum(shard.num_heads for shard in shards)
        if total_heads != self.num_heads:
            raise ValueError("分片头数总和与缓存不匹配")
        
        head_offset = 0
        for shard in shards:
            # 复制每个分片的数据
            for layer_idx in range(self.num_layers):
                seq_len = int(shard._layers[layer_idx]["seq_len"])
                self._layers[layer_idx]["k"][
                    :, head_offset:head_offset+shard.num_heads, :seq_len, :
                ].copy_(shard._layers[layer_idx]["k"][:, :, :seq_len, :])
                
                self._layers[layer_idx]["v"][
                    :, head_offset:head_offset+shard.num_heads, :seq_len, :
                ].copy_(shard._layers[layer_idx]["v"][:, :, :seq_len, :])
                
                # 更新序列长度
                self._layers[layer_idx]["seq_len"] = max(
                    self._layers[layer_idx]["seq_len"], seq_len
                )
            
            head_offset += shard.num_heads

    # ------------------------ 工具方法 ------------------------
    def clear(self) -> None:
        """清空缓存（重置序列长度指针）。"""
        for layer in self._layers:
            layer["seq_len"] = 0

    def summary(self) -> str:
        """生成缓存的摘要信息。
        
        返回：
            描述缓存状态的字符串
        """
        info = [
            f"KVCache(层数={self.num_layers}, 头数={self.num_heads}, 头维度={self.head_dim}",
            f"批大小={self.batch_size}, 最大长度={self.max_seq_len}, 类型={self.dtype}, 设备={self.device})"
        ]
        for idx, layer in enumerate(self._layers):
            info.append(f"  层[{idx}].当前长度={layer['seq_len']}")
        return "\n".join(info)


if __name__ == '__main__':
    # 单元测试
    cache = ByteKVCache(
        num_layers=2,
        num_heads=4,
        head_dim=16,
        max_seq_len=64,
        batch_size=2,
        dtype=torch.float32,
        device='cpu'
    )
    print("初始缓存状态:")
    print(cache.summary())

    # 测试数据追加
    k = torch.randn((2, 4, 1, 16))
    v = torch.randn((2, 4, 1, 16))
    cache.append_batch(0, k, v)
    cache.append_batch(1, k, v)
    print("\n追加数据后:")
    print(cache.summary())

    # 测试重排序
    new_order = torch.tensor([1, 0], dtype=torch.long)
    cache.reorder(new_order)
    print("\n重排序后")

    # 测试序列化
    cache.save("test_cache.pt")
    cache2 = ByteKVCache(
        num_layers=2,
        num_heads=4,
        head_dim=16,
        max_seq_len=64,
        batch_size=2,
        dtype=torch.float32,
        device='cpu'
    )
    cache2.load("test_cache.pt")
    print("\n加载缓存后:")
    print(cache2.summary())