"""
ReasoningLog — Agent 推理过程的结构化记录。

每次 LLM 修改 DSL 时追加一条 ChangeEntry，
整个推理会话序列化为 JSON 字符串后写入 AlphaStore。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ChangeEntry:
    """单次 DSL 变更记录。"""
    step:         int
    old_dsl:      str
    new_dsl:      str
    reason:       str
    metrics:      Dict[str, float] = field(default_factory=dict)
    timestamp:    str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ReasoningLog:
    """
    一次 Agent 推理会话的完整日志。

    Attributes
    ----------
    hypothesis   : 原始市场假设（自然语言）
    initial_dsls : LLM 首轮生成的 DSL 候选列表
    changes      : 逐步修正记录
    final_dsl    : 最终采纳的 DSL
    final_metrics: 最终回测指标摘要
    """
    hypothesis:    str
    initial_dsls:  List[str]
    changes:       List[ChangeEntry] = field(default_factory=list)
    final_dsl:     str = ""
    final_metrics: Dict[str, float] = field(default_factory=dict)
    created_at:    str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_change(
        self,
        old_dsl: str,
        new_dsl: str,
        reason: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        entry = ChangeEntry(
            step=len(self.changes) + 1,
            old_dsl=old_dsl,
            new_dsl=new_dsl,
            reason=reason,
            metrics=metrics or {},
        )
        self.changes.append(entry)

    def to_json(self) -> str:
        """序列化为 JSON 字符串（存入数据库 reasoning 列）。"""
        data = asdict(self)
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "ReasoningLog":
        """从 JSON 字符串反序列化。"""
        data = json.loads(raw)
        changes = [ChangeEntry(**c) for c in data.pop("changes", [])]
        obj = cls(**data)
        obj.changes = changes
        return obj

    def summary(self) -> str:
        lines = [
            f"Hypothesis : {self.hypothesis}",
            f"Initial DSLs: {len(self.initial_dsls)}",
            f"Changes    : {len(self.changes)}",
            f"Final DSL  : {self.final_dsl}",
        ]
        if self.final_metrics:
            for k, v in self.final_metrics.items():
                lines.append(f"  {k}: {v:.4f}")
        return "\n".join(lines)
