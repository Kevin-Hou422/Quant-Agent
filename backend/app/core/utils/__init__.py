"""
utils — 通用向量化算子工具库

集中管理跨模块共用的纯函数（无状态）：
  - fast_ops: 全向量化 ts_*/cs_* 算子（NumPy/Bottleneck）
"""

from .fast_ops import (
    ts_mean, ts_std, ts_delta, ts_rank,
    ts_decay_linear, ind_neutralize,
)

__all__ = [
    "ts_mean", "ts_std", "ts_delta", "ts_rank",
    "ts_decay_linear", "ind_neutralize",
]

