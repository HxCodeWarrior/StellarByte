import math
import torch
import torch.nn as nn
import torch.distributed as dist


class DropPath(nn.Module):
    """DropPath (随机深度) 模块的工业级实现。

    该模块在训练过程中以概率丢弃整个样本路径，在推理时保持完整路径。
    支持概率衰减计划和分布式训练同步，适用于大型模型训练。

    功能特性：
    - 支持线性/余弦衰减计划：随训练步数动态调整丢弃概率
    - 分布式数据并行(DDP)一致性：确保多GPU训练中所有进程使用相同丢弃掩码
    - 自动混合精度(AMP)兼容：正确处理半精度输入
    - 维度无关：适用于任意维度的输入张量

    属性:
        drop_prob (float): 基础丢弃概率，范围[0.0, 1.0]
        schedule_type (str): 衰减计划类型（'linear'/'cosine'）
        total_steps (int): 衰减计划总步数（用于衰减计划）
        sync_ddp (bool): 是否在DDP进程中同步随机掩码
    """

    def __init__(
        self,
        drop_prob: float = 0.0,
        schedule_type: str = "linear",
        total_steps: int = None,
        sync_ddp: bool = False,
    ):
        """初始化DropPath模块。

        参数:
            drop_prob: 基础丢弃概率
            schedule_type: 衰减计划类型（None表示固定概率）,可选: "linear" | "cosine" | None
            total_steps: 衰减计划总步数（无计划时可设为None）
            sync_ddp: 是否在分布式训练中同步掩码
        """
        super().__init__()  # 调用父类nn.Module的构造函数
        assert 0.0 <= drop_prob <= 1.0, "丢弃概率必须在[0,1]范围内"
        self.drop_prob = drop_prob
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        self.sync_ddp = sync_ddp
        # 注册非持久化缓冲区_step用于记录前进步数（不参与模型序列化）
        self.register_buffer("_step", torch.zeros(1, dtype=torch.long), persistent=False)

    def _compute_current_prob(self) -> float:
        """根据衰减计划计算当前丢弃概率。

        返回:
            float: 当前时间步的丢弃概率
        """
        # 未设置计划时返回基础概率
        if self.total_steps is None or self.schedule_type is None:
            return self.drop_prob

        # 计算当前进度 [0, 1]
        progress = self._step.item() / max(1, self.total_steps)
        
        # 线性衰减计划
        if self.schedule_type == "linear":
            return self.drop_prob * progress
        # 余弦衰减计划
        elif self.schedule_type == "cosine":
            return self.drop_prob * (1 - math.cos(math.pi * progress)) / 2
        # 无效计划类型回退到基础概率
        else:
            return self.drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播函数。

        参数:
            x: 输入张量，任意维度形状

        返回:
            torch.Tensor: 应用DropPath后的输出张量
        """
        # 评估模式或丢弃概率为0时直接返回输入
        if not self.training or self.drop_prob == 0.0:
            return x

        # 更新前进步数（每次前向传播递增）
        self._step += 1
        # 计算当前实际丢弃概率
        drop_prob = self._compute_current_prob()
        
        # 概率为0时跳过计算
        if drop_prob == 0.0:
            return x

        keep_prob = 1.0 - drop_prob  # 计算保留概率
        # 创建与输入张量批次维度匹配的掩码形状
        # 格式: (batch_size, 1, 1, ...) 保持其他维度为1
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        
        # 生成随机张量（均匀分布[keep_prob, 1+keep_prob)）
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)

        # 分布式训练同步掩码（确保所有进程使用相同掩码）
        if self.sync_ddp and dist.is_initialized():
            # 从主进程（rank0）广播随机张量
            dist.broadcast(random_tensor, src=0)

        # 二值化掩码（大于1的置1，小于1的置0）
        mask = random_tensor.floor()
        # 缩放输入并应用掩码（注意：训练时需缩放以保持期望值）
        return x.div(keep_prob) * mask

    def extra_repr(self) -> str:
        """生成模块的额外表示信息（用于print展示）。
        
        返回:
            str: 包含模块参数的格式化字符串
        """
        return (
            f"drop_prob={self.drop_prob}, "
            f"schedule_type={self.schedule_type}, "
            f"total_steps={self.total_steps}, "
            f"sync_ddp={self.sync_ddp}"
        )

if __name__ == "__main__":
    torch.manual_seed(42)  # 保证可复现

    # 创建 DropPath（drop_prob=0.5）
    drop_path = DropPath(drop_prob=0.5, schedule_type="linear", total_steps=10)
    
    # 模拟输入 [batch=4, seq=5, hidden=8]
    x = torch.ones((4, 5, 8))

    print("=== 推理模式 (eval) ===")
    drop_path.eval()  # eval 模式下不丢路径
    y_eval = drop_path(x)
    print(y_eval)

    print("\n=== 训练模式 (train) ===")
    drop_path.train()  # train 模式才会生效
    for step in range(1, 6):  # 模拟5个训练step
        y_train = drop_path(x)
        kept_ratio = y_train.sum().item() / x.sum().item()
        print(f"Step {step}: kept_ratio={kept_ratio:.2f}")
        print(y_train)