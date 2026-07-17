"""
alpha_lifecycle.py — 因子生命周期状态机（Task 5.2）

状态流转图：

    CANDIDATE ──→ VALIDATED ──→ PAPER ──→ ACTIVE ──→ DECAYING ──→ RETIRED
        │             │           │          │            │
        └──→ RETIRED  └─→ RETIRED └→ RETIRED └→ DECAYING  └──→ ACTIVE   (恢复)
                                               └→ SUPERSEDED
    任何状态 ──→ RETIRED（人工强制下线永远合法）
    ACTIVE  ──→ SUPERSEDED（被新版本因子替代）

设计说明
--------
- 纯函数式校验：`validate_transition(old, new)` 抛 `IllegalTransition` 或静默通过。
- 兼容历史数据：旧记录的 "active" 状态映射为 ACTIVE。
- 状态语义：
    CANDIDATE  — GP/Agent 刚产出，未经过完整验证
    VALIDATED  — 通过 WalkForward + DSR 门槛（见 OPERATIONS 规则）
    PAPER      — 每日模拟交易验证中（PaperBroker 跟踪）
    ACTIVE     — 通过 paper 验证，参与正式组合
    DECAYING   — 监控检测到 IC 衰减，观察期
    RETIRED    — 已下线（终态；不可再流转）
    SUPERSEDED — 被同假设的新版本替代（终态）
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet


class AlphaStatus(str, Enum):
    CANDIDATE  = "candidate"
    VALIDATED  = "validated"
    PAPER      = "paper"
    ACTIVE     = "active"
    DECAYING   = "decaying"
    RETIRED    = "retired"
    SUPERSEDED = "superseded"


class IllegalTransition(ValueError):
    """非法状态流转。"""


# 合法流转表：old → 允许的 new 集合
_ALLOWED: Dict[AlphaStatus, FrozenSet[AlphaStatus]] = {
    AlphaStatus.CANDIDATE:  frozenset({AlphaStatus.VALIDATED, AlphaStatus.RETIRED}),
    AlphaStatus.VALIDATED:  frozenset({AlphaStatus.PAPER, AlphaStatus.RETIRED}),
    AlphaStatus.PAPER:      frozenset({AlphaStatus.ACTIVE, AlphaStatus.RETIRED}),
    AlphaStatus.ACTIVE:     frozenset({AlphaStatus.DECAYING, AlphaStatus.SUPERSEDED,
                                       AlphaStatus.RETIRED}),
    AlphaStatus.DECAYING:   frozenset({AlphaStatus.ACTIVE, AlphaStatus.RETIRED}),
    AlphaStatus.RETIRED:    frozenset(),          # 终态
    AlphaStatus.SUPERSEDED: frozenset(),          # 终态
}

TERMINAL_STATES: FrozenSet[AlphaStatus] = frozenset(
    {AlphaStatus.RETIRED, AlphaStatus.SUPERSEDED}
)


def coerce_status(raw: str) -> AlphaStatus:
    """
    字符串 → AlphaStatus，兼容历史数据。

    旧 AlphaStore 只有 "active"/"retired" 两个值，直接映射；
    未知字符串抛 ValueError。
    """
    try:
        return AlphaStatus(raw.strip().lower())
    except ValueError:
        raise ValueError(
            f"未知 Alpha 状态 '{raw}'；合法值：{[s.value for s in AlphaStatus]}"
        ) from None


def validate_transition(old: AlphaStatus | str, new: AlphaStatus | str) -> None:
    """
    校验 old → new 是否合法；非法时抛 IllegalTransition。

    规则：
      - old == new 视为幂等操作，永远合法（重复 PATCH 不报错）
      - 流转表之外的组合非法
      - 终态（RETIRED / SUPERSEDED）不可再流转
    """
    o = coerce_status(old) if isinstance(old, str) else old
    n = coerce_status(new) if isinstance(new, str) else new

    if o == n:
        return                                    # 幂等
    if n not in _ALLOWED[o]:
        raise IllegalTransition(
            f"非法流转 {o.value} → {n.value}；"
            f"{o.value} 只允许流转到 {sorted(s.value for s in _ALLOWED[o]) or '（终态，不可流转）'}"
        )


def allowed_next(status: AlphaStatus | str) -> list[str]:
    """返回给定状态的所有合法下一状态（前端流转按钮用）。"""
    s = coerce_status(status) if isinstance(status, str) else status
    return sorted(x.value for x in _ALLOWED[s])
