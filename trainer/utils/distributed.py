###############################################################################
# 文件: utils/distributed.py
###############################################################################
"""\
分布式训练相关工具
- 包括初始化、全局同步、获取本地设备、barrier 等函数
"""

import os
import torch


def init_distributed_mode(backend: str = 'nccl') -> int:
    """初始化分布式环境（若环境变量表明处于分布式运行）。

    约定使用环境变量：RANK, WORLD_SIZE, LOCAL_RANK

    Returns:
        int: 本进程 local_rank（-1 表示未启用分布式）。
    """
    try:
        rank = int(os.environ.get('RANK', -1))
    except Exception:
        rank = -1
    if rank == -1:
        return -1
    torch.distributed.init_process_group(backend=backend)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def get_world_size() -> int:
    """获取分布式世界大小（GPU 数量）。"""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank() -> int:
    """获取当前进程 rank（非分布式返回 0）。"""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def broadcast_object(obj, src=0):
    """在分布式进程间广播任意可 picklable 的 Python 对象。

    Args:
        obj: 源对象（只有 src 进程的 obj 会被广播），其它进程可传 None。
        src (int): 源进程 rank。

    Returns:
        object: 广播后的对象副本。
    """
    import pickle
    if get_world_size() == 1:
        return obj
    if get_rank() == src:
        data = pickle.dumps(obj)
    else:
        data = None
    data = torch.tensor(bytearray(pickle.dumps(obj) if get_rank() == src else b''), dtype=torch.uint8, device='cuda')
    # 先广播长度
    if get_rank() == src:
        length = torch.tensor([len(data)], device='cuda')
    else:
        length = torch.tensor([0], device='cuda')
    torch.distributed.broadcast(length, src)
    buf = torch.empty((length.item(),), dtype=torch.uint8, device='cuda')
    if get_rank() == src:
        buf.copy_(data)
    torch.distributed.broadcast(buf, src)
    if get_rank() != src:
        import pickle as p
        obj = p.loads(bytes(buf.tolist()))
    return obj