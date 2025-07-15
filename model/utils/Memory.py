from __future__ import annotations

import math
import uuid
from typing import List, Tuple, Optional, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# 尝试导入 FAISS（GPU/CPU 向量检索库）
# ------------------------------
try:
    import faiss  # 若安装失败仍可运行
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------

def _normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """对最后一个维度做 L2 归一化，避免除零。"""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _default_device() -> torch.device:
    """优先使用 GPU。若无 CUDA，则退回 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# 1. 向量记忆数据库（MemoryDB）
# -----------------------------------------------------------------------------

class MemoryDB:
    """带有可选 FAISS 加速的向量存储。

    * 支持 add / search / delete 基本操作
    * value 部分可存放任意 Python dict，常见为原始文本和元数据
    """

    def __init__(self, dim: int, device: Optional[torch.device] = None,
                 enable_faiss: bool = True):
        self.dim = dim
        self.device = device or _default_device()
        self.enable_faiss = enable_faiss and _FAISS_AVAILABLE

        # _values 用于保存实际内容（文本、元信息等）
        self._values: Dict[int, Dict] = {}
        self._ids: List[int] = []  # 按照插入顺序自增的内部 ID

        if self.enable_faiss:
            # 对余弦相似度，先做归一化，可使用内积近似
            index = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIDMap(index)
        else:
            # 无 FAISS 时，仅用 torch 矩阵保存 key
            self._keys = torch.empty(0, dim, device=self.device)

    # ------------------------------------------------------------------
    # 写入操作
    # ------------------------------------------------------------------
    def add(self, keys: torch.Tensor, values: Sequence[Dict]):
        """向记忆库批量新增 *B* 条记录。

        参数
        ----
        keys : Tensor(B, D)  已 L2 归一化，需与 MemoryDB.device 一致
        values : 包含原文 / 元数据的 dict 列表
        """
        # 确保 tensor 在正确设备、格式为 float32
        keys = keys.detach().to(self.device, dtype=torch.float32)
        # 生成连续 ID
        ids = torch.arange(len(self._ids), len(self._ids) + len(keys)).tolist()

        if self.enable_faiss:
            self._index.add_with_ids(keys.cpu().numpy(), np.array(ids, dtype="int64"))
        else:
            self._keys = torch.cat([self._keys, keys], dim=0)

        for _id, v in zip(ids, values):
            self._values[_id] = v
        self._ids.extend(ids)

    # ------------------------------------------------------------------
    # 读取 / 检索操作
    # ------------------------------------------------------------------
    @torch.no_grad()
    def search(self, queries: torch.Tensor, top_k: int = 8) -> List[List[Dict]]:
        """给定查询向量，返回最相似的 *top_k* 记忆条目。

        返回值为嵌套列表：`List[batch, List[value_dict]]`
        """
        queries = _normalize(queries.to(self.device))
        if self.enable_faiss:
            D, I = self._index.search(queries.cpu().numpy(), top_k)
            results: List[List[Dict]] = [
                [self._values[i] for i in idx if i != -1] for idx in I
            ]
            return results
        else:
            if len(self._keys) == 0:
                return [[] for _ in range(len(queries))]
            # (B, N) 余弦相似度
            sims = torch.matmul(queries, self._keys.t())
            topk = sims.topk(min(top_k, sims.size(1)), dim=-1).indices  # (B, k)
            return [[self._values[self._ids[i]] for i in row.tolist()] for row in topk]

    # ------------------------------------------------------------------
    # 删除操作（按内部 ID）
    # ------------------------------------------------------------------
    def delete(self, ids: List[int]):
        if self.enable_faiss:
            self._index.remove_ids(np.array(ids, dtype="int64"))
        else:
            mask = torch.ones(len(self._ids), dtype=torch.bool, device=self.device)
            for i in ids:
                if i in self._ids:
                    idx = self._ids.index(i)
                    mask[idx] = False
            self._keys = self._keys[mask]
        for i in ids:
            self._values.pop(i, None)
            if i in self._ids:
                self._ids.remove(i)


# -----------------------------------------------------------------------------
# 2. ReversibleCompressor – 可逆压缩器
# -----------------------------------------------------------------------------

class ReversibleCompressor(nn.Module):
    """线性可逆投影，用少量 token 近似表示整段序列隐藏状态。"""

    def __init__(self, hidden_dim: int, wm_tokens: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.wm_tokens = wm_tokens
        # 使用正交矩阵初始化，保证 W · W^T ≈ I，实现近似可逆
        W = torch.empty(wm_tokens, hidden_dim)
        nn.init.orthogonal_(W)
        self.W = nn.Parameter(W)  # 形状 (T, D)

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """压缩： (B, L, D) → (B, 1, T)"""
        pooled = x.mean(dim=1)             # 简单平均池化
        wm = torch.matmul(pooled, self.W.t())  # (B, T)
        return wm.unsqueeze(1)             # 添加 token 维度

    def decompress(self, wm: torch.Tensor, seq_len: int) -> torch.Tensor:
        """解压： (B, 1, T) → (B, L, D') 用于重建评估"""
        wm = wm.squeeze(1)                 # (B, T)
        recon = torch.matmul(wm, self.W)   # (B, D)
        recon = recon.unsqueeze(1).repeat(1, seq_len, 1)  # 广播到序列长度
        return recon


# -----------------------------------------------------------------------------
# 3. MemoryController – 写入门控网络
# -----------------------------------------------------------------------------

class MemoryController(nn.Module):
    """简单的两层线性网络，输出写入概率。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, h_cls: torch.Tensor, h_last: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """根据阈值返回布尔掩码 (B,)，指示哪些样本需要写入记忆库。"""
        g = torch.sigmoid(self.fc(torch.cat([h_cls, h_last], dim=-1))).squeeze(-1)
        return g > threshold


# -----------------------------------------------------------------------------
# 4. MemoryAugmentedLM – 记忆增强语言模型包装器
# -----------------------------------------------------------------------------

class MemoryAugmentedLM(nn.Module):
    """在前向传播中透明地执行“检索 + 写回”。

    条件：backbone 至少实现
    * `embed(input_ids)` → (B, L, D)
    * `forward(input_ids, **kwargs)`  语言模型常规输出
    并有 `config.hidden_size` 属性。
    """

    def __init__(self,
                 backbone: nn.Module,
                 tokenizer,
                 memory_db: MemoryDB,
                 wm_tokens: int = 64,
                 mem_token_id: Optional[int] = None,
                 top_k: int = 8):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        # 若未传入，自行在 tokenizer 中查找 <MEM> 特殊 token
        self.mem_token_id = mem_token_id or tokenizer.convert_tokens_to_ids("<MEM>")
        self.memory_db = memory_db
        self.top_k = top_k
        self.hidden_dim = backbone.config.hidden_size
        self.comp = ReversibleCompressor(self.hidden_dim, wm_tokens)
        self.controller = MemoryController(self.hidden_dim)

    # ------------------------------------------------------------------
    # 前向传播入口
    # ------------------------------------------------------------------
    def forward(self, input_ids: torch.LongTensor, **kwargs):
        B, L = input_ids.shape
        device = input_ids.device

        # ---- 1) 获取查询向量（用 [CLS] 与最后一个 token） ----
        with torch.no_grad():
            embeddings = self.backbone.embed(input_ids)  # (B, L, D)
            h_cls = embeddings[:, 0]    # 句首
            h_last = embeddings[:, -1]  # 句尾
            queries = _normalize(h_cls)

        # ---- 2) 检索向量记忆库 ----
        memory_chunks = self.memory_db.search(queries, top_k=self.top_k)

        # 将检索结果编码为 token 序列，并统计最大长度
        mem_token_lists: List[List[int]] = []
        max_mem_len = 0
        for chunk_list in memory_chunks:
            ids: List[int] = []
            for c in chunk_list:
                # 以特殊 <MEM> 分隔
                ids += [self.mem_token_id] + self.tokenizer.encode(c["text"], add_special_tokens=False)
            mem_token_lists.append(ids)
            max_mem_len = max(max_mem_len, len(ids))

        # 若存在检索结果，则把它们 *前置* 到输入序列
        if max_mem_len > 0:
            mem_tokens = torch.full((B, max_mem_len), self.tokenizer.pad_token_id, device=device)
            for i, ids in enumerate(mem_token_lists):
                if ids:
                    mem_tokens[i, :len(ids)] = torch.tensor(ids, device=device)
            input_ids = torch.cat([mem_tokens, input_ids], dim=1)

        # ---- 3) 继续常规前向 ----
        outputs = self.backbone(input_ids, **kwargs)

        # ---- 4) 写入记忆（根据门控判定） ----
        with torch.no_grad():
            # 最后一层隐藏状态 (B, L_total, D)
            embeddings_last = outputs["hidden_states"][-1]
            write_mask = self.controller(h_cls, h_last)  # (B,)
            if write_mask.any():
                wm = self.comp.compress(embeddings_last[write_mask])      # (b, 1, T)
                wm_flat = _normalize(wm.squeeze(1))                       # (b, T)
                keys = wm_flat.mean(dim=1)                                # (b, D)
                # 将完整原文（无 padding）作为 value 存储
                values = [{
                    "text": self.tokenizer.decode(sample_ids.tolist(), skip_special_tokens=True),
                    "uuid": str(uuid.uuid4())
                } for sample_ids in input_ids[write_mask]]
                self.memory_db.add(keys, values)

        return outputs


# -----------------------------------------------------------------------------
# 5. 最小示例：直接运行查看效果
# -----------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover – demo only
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "facebook/opt-350m"  # 示例骨干
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModelForCausalLM.from_pretrained(model_name)

    mem_dim = backbone.config.hidden_size
    memory_db = MemoryDB(dim=mem_dim)  # 默认启用 FAISS（若可用）
    lmm = MemoryAugmentedLM(backbone, tokenizer, memory_db)

    prompt = "Hello, my name is Ada and I love programming."
    ids = tokenizer.encode(prompt, return_tensors="pt")

    lmm.eval()
    with torch.no_grad():
        output = lmm(input_ids=ids)
    print("Logits shape:", output.logits.shape)
