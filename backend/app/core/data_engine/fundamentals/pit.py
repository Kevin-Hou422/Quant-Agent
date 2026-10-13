"""
pit.py — 基本面事实的时点（point-in-time）计算（Phase 10.2 / 10.3）

本模块里每个函数都只接收**一只股票、截至某个披露日已经公开**的事实（调用方先按
`filed <= as_of` 过滤），所以它们在构造上不可能看到未来。时点语义集中在这一个地方：

    某期间的数值 = 该期间在 as_of 之前披露的**最新一版**（重述覆盖原值，但只从重述的披露日起）。

期间的种类只看 `start` / `end` 的跨度（`fp` / `fy` 描述的是文件，不是事实，见 edgar.py）：
    全年      340–380 天        年初至今 / 季度   < 340 天        时点（资产负债表）  start 为空

财年不是整 365 天（52/53 周制，AAPL 的季末每年挪几天），所以"上年同期"按 ±DATE_TOL 天匹配。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ANNUAL_MIN, ANNUAL_MAX = 340, 380
QUARTER_MIN, QUARTER_MAX = 80, 100
#: 上年同期 / 上一财年末的日期匹配容差（52/53 周财年）
DATE_TOL = pd.Timedelta(days=10)
YEAR = pd.Timedelta(days=365)
#: SUE 的标准化窗口：过去 8 个季度的同比差；少于 MIN 个就不给数
SUE_WINDOW, SUE_MIN = 8, 6


def _latest_versions(rows: pd.DataFrame) -> pd.DataFrame:
    """每个 (start, end) 只留披露日最晚的一版。"""
    if rows.empty:
        return rows
    return (rows.sort_values(["filed", "accn"])
                .drop_duplicates(subset=["start", "end"], keep="last"))


def flows(visible: pd.DataFrame, concept: str) -> pd.DataFrame:
    """某流量概念的可见事实（最新版本），附 `days` = 期间天数。"""
    r = visible[(visible["concept"] == concept) & visible["start"].notna()]
    r = _latest_versions(r).copy()
    r["days"] = (r["end"] - r["start"]).dt.days
    return r


def latest_instant(visible: pd.DataFrame, concept: str) -> Optional[pd.Series]:
    """某时点概念：期末日最晚的那一期的最新版本。没有 → None。"""
    r = visible[(visible["concept"] == concept) & visible["start"].isna()]
    if r.empty:
        return None
    r = r[r["end"] == r["end"].max()].sort_values(["filed", "accn"])
    return r.iloc[-1]


def instant_near(visible: pd.DataFrame, concept: str, when: pd.Timestamp) -> Optional[float]:
    """期末日在 `when` ± DATE_TOL 内的时点值（最新版本）；没有 → None。"""
    r = visible[(visible["concept"] == concept) & visible["start"].isna()]
    r = r[(r["end"] - when).abs() <= DATE_TOL]
    if r.empty:
        return None
    return float(r.sort_values(["end", "filed", "accn"]).iloc[-1]["val"])


def _match(rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    m = rows[((rows["start"] - start).abs() <= DATE_TOL) & ((rows["end"] - end).abs() <= DATE_TOL)]
    if m.empty:
        return None
    return float(m.sort_values(["filed", "accn"]).iloc[-1]["val"])


def ttm(visible: pd.DataFrame, concept: str) -> Optional[Tuple[float, pd.Timestamp]]:
    """
    最近十二个月（TTM）= 上一财年全年 + 本财年年初至今 − 上年同期年初至今。
    三个数都是财报里**直接披露**的（10-Q 的利润表同时给单季与年初至今两列），不做任何拆分估计。
    最新一期恰好是全年 → 直接取全年数。返回 (数值, 期末日)；凑不齐 → None。
    """
    f = flows(visible, concept)
    if f.empty:
        return None
    end = f["end"].max()
    at_end = f[f["end"] == end]
    annual = at_end[(at_end["days"] >= ANNUAL_MIN) & (at_end["days"] <= ANNUAL_MAX)]
    if not annual.empty:
        return float(annual.iloc[-1]["val"]), end
    ytd = at_end[at_end["days"] < ANNUAL_MIN]
    if ytd.empty:
        return None
    cur = ytd.sort_values("days").iloc[-1]                 # 跨度最长的那条 = 年初至今
    fy_start = cur["start"]
    years = f[(f["days"] >= ANNUAL_MIN) & (f["days"] <= ANNUAL_MAX)]
    prev_fy = years[(years["end"] - (fy_start - pd.Timedelta(days=1))).abs() <= DATE_TOL]
    prev_ytd = _match(f[f["days"] < ANNUAL_MIN], fy_start - YEAR, end - YEAR)
    if prev_fy.empty or prev_ytd is None:
        return None
    return float(prev_fy.sort_values(["filed", "accn"]).iloc[-1]["val"] + cur["val"] - prev_ytd), end


def quarterly(visible: pd.DataFrame, concept: str) -> Dict[pd.Timestamp, float]:
    """
    单季数值 {季末日: 值}。直接披露的 3 个月窗口优先；第四季度财报不单列，
    用 全年 − 前三季度年初至今 得到（两个数都是直接披露的）。
    """
    f = flows(visible, concept)
    out: Dict[pd.Timestamp, float] = {}
    q = f[(f["days"] >= QUARTER_MIN) & (f["days"] <= QUARTER_MAX)]
    for _, r in q.iterrows():
        out[r["end"]] = float(r["val"])
    years = f[(f["days"] >= ANNUAL_MIN) & (f["days"] <= ANNUAL_MAX)]
    nine = f[(f["days"] >= 260) & (f["days"] <= 290)]
    for _, y in years.iterrows():
        if y["end"] in out:
            continue
        m = nine[(nine["start"] - y["start"]).abs() <= DATE_TOL]
        if not m.empty:
            out[y["end"]] = float(y["val"] - m.sort_values(["filed", "accn"]).iloc[-1]["val"])
    return out


def sue(visible: pd.DataFrame, concept: str = "net_income") -> Optional[Tuple[float, pd.Timestamp]]:
    """
    标准化盈利超预期（季节性随机游走）：(本季 − 去年同季) / 过去 8 个季度同比差的标准差。
    用净利润（美元）而不是每股收益：不受拆股影响，且第四季度可由 全年 − 前三季 精确得到。
    返回 (SUE, 季末日)；历史不足 SUE_MIN 个同比差 → None。
    """
    q = quarterly(visible, concept)
    if not q:
        return None
    ends = sorted(q)

    def yoy(e: pd.Timestamp) -> Optional[float]:
        prior = [p for p in ends if abs(p - (e - YEAR)) <= DATE_TOL]
        return None if not prior else q[e] - q[prior[-1]]

    last = ends[-1]
    surprise = yoy(last)
    hist = [d for d in (yoy(e) for e in ends[:-1][-SUE_WINDOW:]) if d is not None]
    if surprise is None or len(hist) < SUE_MIN:
        return None
    sd = float(np.std(hist, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        return None
    return float(surprise / sd), last


def filing_dates(facts: pd.DataFrame) -> List[pd.Timestamp]:
    """这只股票所有出现过的披露日（升序去重）—— 时点指标只在这些日子可能变化。"""
    return sorted(pd.DatetimeIndex(facts["filed"].dropna().unique()))
