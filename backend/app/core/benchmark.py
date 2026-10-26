"""
benchmark.py — 正式前向每日记录的基准（2026-10-22 用户决定 D16）

用户原话："首轮用等权宇宙。预先固定股票池时点、再平衡、收益窗口、缺价和公司行动处理。不能事后删除缺价股票再平均。
SPY 第二基准后排；低仓位策略对满仓基准的差额不要直接称为 alpha。"

规则（`BENCHMARK_RULES`，带版本号写进每天的记录；改规则 = 换版本）：

    股票池时点   正式前向**启动时**冻结成分名单（`ForwardRecordStore.start(benchmark_universe=...)`，
                 存进 forward_runs.benchmark）；启动之前的记录用当天数据集的列，并标 universe_frozen=False
    再平衡       每日收盘再平衡到等权（每个成分 1/N，N = 冻结名单的长度）
    收益窗口     收盘(T−1) → 收盘(T)：与模拟账本 / 券商的收盘估值同一窗口（实际持仓收益；预测 IC 另记，见 clock.py）
    价格         数据集的复权收盘价（拆股 / 分红按数据源复权）
    缺价         成分当天缺价 → 按**最后可信价**持有、当日收益记 0，**留在分母**；价格恢复的那天按
                 今收 / 最后可信价结清缺价期间的变化（b2，2026-10-25 审计 R5：b1 要求相邻两天都有价，
                 100、缺失、120 两天都记 0、累计 0%，恢复段被永久漏记）。最后可信价只往回找，不用未来价格回填
    公司行动     有成分被公司行动检查隔离（未知公司行动）→ 估值不可信，**当天暂停发布**（return = None，
                 另给 provisional_return 标明是暂定的、隔离成分按 0）；b1 把它当成稳定的 0 收益发布
    第二基准     SPY 后排，本版不提供
    差额的叫法   "相对满仓等权基准的差额"；策略净敞口低于满仓时它不是 alpha —— 另给按实际净敞口折算的差额，
                 它用日末敞口乘全天基准收益，只是**近似代理**，不是精确的 alpha
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

BENCHMARK_NAME = "ew_universe"

BENCHMARK_RULES: Dict[str, str] = {
    "version": "b2",
    "universe": "正式前向启动时冻结成分名单（forward_runs.benchmark）；启动前用当天数据集的列，标 universe_frozen=False",
    "rebalance": "每日收盘再平衡到等权，每个成分 1/N，N = 冻结名单长度",
    "return_window": "收盘(T−1) → 收盘(T)，与模拟账本 / 券商的收盘估值同一窗口",
    "price": "数据集的复权收盘价（拆股 / 分红按数据源复权）",
    "missing_price": ("缺价成分按最后可信价持有、当日收益记 0、留在分母；价格恢复当天按 今收 / 最后可信价 结清；"
                      "缺价名单、比例与各自的最后可信日写进记录（不删后再平均，不用未来价回填）"),
    "corporate_actions": "有成分处于未知公司行动隔离 → 估值不可信，当天暂停发布（return=None），另给暂定值与名单",
    "secondary": "SPY 第二基准后排，本版不提供",
    "excess": "相对满仓等权基准的差额不是 alpha；另给按日末净敞口折算的差额（近似代理）",
}

EXCESS_LABEL = ("相对满仓等权宇宙的差额：策略净敞口低于满仓时它不是 alpha；"
                "*_exposure_matched = 策略收益 − 日末净敞口 × 全天基准收益，只是近似代理、不是精确 alpha")


def freeze_universe(tickers: Iterable[str]) -> dict:
    """冻结基准成分（正式前向启动时调用）。名单去重排序，带摘要与规则版本。"""
    names = sorted({str(t) for t in tickers if str(t).strip()})
    if not names:
        raise ValueError("基准成分名单为空")
    return {"name": BENCHMARK_NAME, "rules_version": BENCHMARK_RULES["version"], "tickers": names,
            "n": len(names), "digest": hashlib.sha256("\x1f".join(names).encode("utf-8")).hexdigest()[:16]}


def ew_universe_return(close: pd.DataFrame, day, universe: Optional[List[str]],
                       quarantined: Iterable[str] = ()) -> dict:
    """
    第 `day` 天的等权宇宙收益（收盘(T−1) → 收盘(T)）。universe=None → 用当天数据集的列（未冻结）。
    缺价成分按最后可信价持有（当天 0）、恢复当天结清；有成分被隔离 → 暂停发布（return=None，给暂定值）。
    day 不在数据里、或是第一行 → return=None（并说明）。
    """
    day = pd.Timestamp(day).normalize()
    frozen = universe is not None
    names = list(universe) if frozen else [str(c) for c in close.columns]
    q = sorted({str(t) for t in quarantined})
    out = {"name": BENCHMARK_NAME, "rules_version": BENCHMARK_RULES["version"], "universe_frozen": frozen,
           "n": len(names), "return": None, "missing": [], "missing_share": None, "quarantined": [],
           "window": None, "secondary": None, "secondary_note": BENCHMARK_RULES["secondary"]}
    if day not in close.index:
        out["why_none"] = "数据里没有这一天"
        return out
    pos = close.index.get_loc(day)
    if pos == 0 or not names:
        out["why_none"] = "没有前一天的收盘价" if pos == 0 else "基准成分为空"
        return out
    prev_day = close.index[pos - 1]
    out["window"] = [str(pd.Timestamp(prev_day).date()), str(day.date())]
    rets, missing, held, settled = [], [], [], {}
    for t in names:
        if t in q:
            held.append(t)
            rets.append(0.0)
            continue
        if t not in close.columns:
            missing.append(t)
            rets.append(0.0)
            continue
        b = close.at[day, t]
        last = _last_trusted(close[t], pos - 1)            # 只往回找：T−1 及更早
        if pd.notna(b) and float(b) > 0 and last is not None:
            rets.append(float(b) / last[1] - 1.0)
            if last[0] != prev_day:                         # 缺价之后恢复：这一天结清缺价期间的变化
                settled[t] = str(pd.Timestamp(last[0]).date())
        else:
            missing.append(t)
            rets.append(0.0)
    value = float(np.mean(rets))
    out["missing"] = sorted(missing)
    out["missing_share"] = len(missing) / len(names)
    out["quarantined"] = sorted(held)
    out["settled_since"] = settled
    out["valuation_trusted"] = not held
    if held:
        out["provisional_return"] = value
        out["why_none"] = f"成分 {sorted(held)} 处于未知公司行动隔离：估值不可信，当天暂停发布（暂定值另列）"
    else:
        out["return"] = value
    return out


def _last_trusted(series: pd.Series, upto: int):
    """位置 `upto` 及之前最后一个有效（正、有限）的收盘价 → (日期, 价)；没有 → None。"""
    s = pd.to_numeric(series.iloc[: upto + 1], errors="coerce")
    s = s[(s > 0) & np.isfinite(s)]
    return (s.index[-1], float(s.iloc[-1])) if len(s) else None
