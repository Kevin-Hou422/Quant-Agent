"""
corporate_actions.py — 未知公司行动的识别与"停止新增风险"（2026-10-22 用户决定第 18 条 / F02′）

用户原话："公司行动账本：完整功能可以后排。但价量策略同样依赖拆股、分红、复权及券商股数 / 现金正确。
未知公司行动必须隔离、停止新增风险并对账，不能以'没基本面因子'豁免。"

这里只做**最小但不可绕过**的一层（完整账本后排）：

    识别     最近几根 bar 里，单日 |对数收益| ≥ log(1.4)（跌 ≥ 28.6% 或涨 ≥ 40%）→ 疑似未知公司行动
             （未复权的拆股 / 合股、特别分红、分拆……）；离某个拆股比例近的标出最近的比例（2:1、1:2 …）。
             单看价格分不清"真暴跌"与"没复权的拆股" —— 误报的代价只是这只票暂时不加风险，宁可误报。
    隔离     标记写进 `corporate_action_flags`（app/db/corporate_actions.py），一直有效到**人**对账后结案
    不加风险 模拟账本的目标：标记日起，这只票的权重不能超过标记前的仓位（同向收缩可以，加仓 / 反手不行）；
             券商下单：盘前风控门拒绝这只票的加风险单（减仓 / 平仓照常放行）
    对账     券商持仓与账本推算对不上时，差异按股数比例标出疑似的拆股比例，方便人核对
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: 单日 |对数收益| 超过它 → 疑似公司行动
JUMP_LOG_THRESHOLD = math.log(1.4)
#: 往回看几根 bar（停机几天后也能补上）
LOOKBACK_BARS = 5
#: 常见拆股 / 合股比例（新股数 : 旧股数）
_SPLITS = [(2, 1), (3, 1), (4, 1), (5, 1), (10, 1), (20, 1), (3, 2),
           (1, 2), (1, 3), (1, 4), (1, 5), (1, 10), (1, 20), (2, 3)]
#: 价格比例离拆股比例多近算"像"（对数距离；含当天的正常波动）
_SPLIT_TOL = math.log(1.15)


def nearest_split(price_ratio: float) -> Optional[str]:
    """
    价格比例（今 / 昨）最像哪种拆股，按惯例写成"新股数:旧股数"（2:1 拆股 → 价格约减半）。
    都不像 → None。拆 / 合股后价格 ≈ 原价 × 旧/新。
    """
    if not (price_ratio > 0 and math.isfinite(price_ratio)):
        return None
    best, dist = None, _SPLIT_TOL
    for new, old in _SPLITS:
        d = abs(math.log(price_ratio) - math.log(old / new))
        if d < dist:
            best, dist = f"{new}:{old}", d
    return best


def detect_suspects(close: pd.DataFrame, volume: Optional[pd.DataFrame] = None,
                    lookback: int = LOOKBACK_BARS) -> List[dict]:
    """最近 `lookback` 根 bar 上的疑似未知公司行动。每条 = {ticker, bar_date, ratio, nearest_split, volume_ratio}。"""
    if close is None or close.empty or len(close.index) < 2:
        return []
    c = close.astype(float)
    ratio = c / c.shift(1)
    out: List[dict] = []
    for day in c.index[-lookback:]:
        if day == c.index[0]:
            continue
        r = ratio.loc[day]
        for tk, v in r.items():
            if not (np.isfinite(v) and v > 0) or abs(math.log(v)) < JUMP_LOG_THRESHOLD:
                continue
            vr = None
            if volume is not None and tk in volume.columns:
                prev = volume[tk].shift(1).loc[day]
                vr = float(volume[tk].loc[day] / prev) if prev and np.isfinite(prev) and prev > 0 else None
            out.append({"ticker": str(tk), "bar_date": str(pd.Timestamp(day).date()), "ratio": float(v),
                        "nearest_split": nearest_split(float(v)), "volume_ratio": vr})
    return out


def no_new_risk(weights: pd.DataFrame, quarantined: Dict[str, str]) -> pd.DataFrame:
    """
    模拟账本目标的"停止新增风险"：对每只被隔离的票，标记日（含）起的每一行
    不能超过标记日前一行的仓位 —— 同向且绝对值不超过它；异号 → 0（只平不反手）。标记日前的行不动。
    """
    out = weights.copy()
    for tk, since in quarantined.items():
        if tk not in out.columns:
            continue
        mask = out.index >= pd.Timestamp(since)
        if not mask.any():
            continue
        first = int(np.argmax(mask))
        ref = float(out[tk].iloc[first - 1]) if first > 0 else 0.0
        col = out.loc[mask, tk].astype(float)
        same = np.sign(col) == np.sign(ref)
        capped = np.where(same, np.sign(col) * np.minimum(col.abs(), abs(ref)), 0.0)
        out.loc[mask, tk] = capped
    return out


def annotate_discrepancies(discrepancies: List[dict]) -> List[dict]:
    """券商持仓 / 账本推算的股数比例像拆股 → 标 suspected_corporate_action（"新:旧"），方便对账。"""
    out = []
    for d in discrepancies:
        d = dict(d)
        e, b = d.get("expected"), d.get("broker")
        try:
            e, b = float(e), float(b)
        except (TypeError, ValueError) as exc:
            logger.warning("[corporate_action] 对账差异 %s 的股数不是数（不标拆股比例）: %s", d, exc)
            out.append(d)
            continue
        if e and b and np.sign(e) == np.sign(b):
            share_ratio = b / e                                   # 股数比例 = 新 / 旧
            for new, old in _SPLITS:
                if abs(math.log(share_ratio) - math.log(new / old)) < math.log(1.02):
                    d["suspected_corporate_action"] = f"{new}:{old}"
                    break
        out.append(d)
    return out
