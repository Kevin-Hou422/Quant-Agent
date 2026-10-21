"""
quality.py — 基本面的来源 / 期间质量门（复审 2026-10-19 §5.4）

复审原话："启用基本面交易前应加来源/期间质量门。"§3："保留来源、有效期间、fallback 和缺失标记。"

字段存在不等于可以交易：末日覆盖率低（大量过期 / 缺事实 / 异常隔离）、市值用的是估计股本（稀释加权平均）
的票占比高，都会让横截面排序建立在一批不可比的数上。质量门只管**用到基本面字段的策略**：

  · 提案：不过 → `passed=False`，原因写进 verdict；
  · 每日循环：active 策略不过 → 本轮不调仓（只对账 / 熔断维护）。

质量报告由 `attach_fundamentals` 每次挂字段时落盘（`FundamentalsStore.save_report`），必须是**这份面板末日**的。
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .panels import ALL_FIELDS, VALUATION_FIELDS

logger = logging.getLogger(__name__)


def fundamental_fields_in(expressions: Iterable[Optional[str]]) -> List[str]:
    """表达式里用到的基本面字段（按标识符整词匹配）。"""
    found = set()
    for e in expressions:
        if not e:
            continue
        for f in ALL_FIELDS:
            if re.search(rf"(?<![A-Za-z0-9_]){re.escape(f)}(?![A-Za-z0-9_])", str(e)):
                found.add(f)
    return sorted(found)


def quality_problems(report, fields: Iterable[str], min_coverage: Optional[float] = None,
                     max_estimated: Optional[float] = None) -> List[str]:
    """逐条列出质量问题（空 = 过门）。门槛缺省读 settings（`fund_min_coverage` / `fund_max_estimated_shares`）。"""
    fields = list(fields)
    if not fields:
        return []
    if min_coverage is None or max_estimated is None:
        from app.config import settings
        min_coverage = float(getattr(settings, "fund_min_coverage", 0.8)) if min_coverage is None else min_coverage
        max_estimated = (float(getattr(settings, "fund_max_estimated_shares", 0.1))
                         if max_estimated is None else max_estimated)
    out: List[str] = []
    for f in fields:
        cov = float(report.coverage.get(f, 0.0))
        if cov < min_coverage:
            out.append(f"{f} 末日覆盖 {cov:.0%} < {min_coverage:.0%}（过期 / 缺事实 / 异常隔离）")
    if any(f in VALUATION_FIELDS for f in fields):
        n = max(int(report.n_tickers or 0), 1)
        share = len(report.estimated_shares) / n
        if share > max_estimated:
            out.append(f"{share:.0%} 的标的市值用的是稀释加权平均股数（估计值，不是期末股本）> {max_estimated:.0%}")
    return out


def strategy_quality_problem(factor_dsl: Dict[str, Optional[str]], last_day) -> str:
    """策略（成分表达式）用到基本面字段时的质量门结论；没用到 → 空串。"""
    fields = fundamental_fields_in(factor_dsl.values())
    if not fields:
        return ""
    try:
        from app.config import settings
        from .service import fundamentals_dir
        from .store import FundamentalsStore
        got = FundamentalsStore(fundamentals_dir(settings)).load_report()
    except Exception as exc:
        logger.error("[fundamentals] 读不到质量报告 → 按不过门处理: %s", exc)
        return f"读不到基本面质量报告（{exc}）"
    if got is None:
        return f"用到 {fields}，但没有基本面质量报告（这份面板没挂过基本面字段）"
    report, asof = got
    if pd.Timestamp(asof).normalize() != pd.Timestamp(last_day).normalize():
        return f"基本面质量报告是 {pd.Timestamp(asof).date()} 的，面板末日是 {pd.Timestamp(last_day).date()}"
    return "; ".join(quality_problems(report, fields))
