"""
股本：报告写明股本来源、市值随股本事实过期、用估计股本估值的标的计数可见。

来源（原测试文件的说明，保留出处）：

外部审查 2026-10-18 F13：基本面字段的时间正确不等于经济口径正确。

  · SUE 等 10-Q，days_since_earnings 按 8-K 2.02 归零 —— 新 8-K 已到、10-Q 未到时，
    `days_since_earnings < 60` 会选中**上一季**的 SUE。
  · 过期按"最近一次披露"统一算：只更新股本的披露会让一年前的利润数字继续算新鲜。
  · 稀释加权平均股数是兜底，不是期末股本 —— 来源要看得见。
  · 拆股识别是启发式：复权因子跳了、不复权价没动（数据拼接问题）不能当拆股。

---

复审 2026-10-19 F13 余项 / §5.4 / D8（基本面）：

  "`_shares` 的 diluted weighted-average fallback 不等于期末股本；shares 的事实期间没有与其他指标同样的
   逐字段过期检查。小于 1.5 倍的疑似送股部分只告警，仍会影响市值/估值；SUE 事件仅按日期关系屏蔽，
   未明确匹配盈利期间。启用基本面交易前应加来源/期间质量门。"
  §3 模块表："保留来源、有效期间、fallback 和缺失标记；小比例公司行动必须进入异常隔离。"
  D8："先把 filing-based 因子名称、期间和事件配对写清。"
"""

from __future__ import annotations

import pandas as pd

from app.core.data_engine.fundamentals.panels import (
    MAX_PERIOD_AGE_DAYS,
    SHARES_SOURCES,
    build_panels,
    metric_events,
)
from tests.unit.data_engine import fundamentals_fixtures as fx

IDX = pd.bdate_range("2023-07-03", "2024-11-27")


def _aapl():
    return fx.facts("AAPL"), fx.events("AAPL")


def _raw(idx=IDX, px=180.0):
    return pd.DataFrame({"AAPL": px}, index=idx)


def test_the_report_names_the_share_count_source():
    facts, events = _aapl()
    _, rep = build_panels(IDX, ["AAPL"], {"AAPL": facts}, {"AAPL": events})
    assert rep.shares_source["AAPL"] in SHARES_SOURCES.values()
    no_cover = facts[~facts["concept"].astype(str).isin(["shares_cover", "shares_bs"])]
    _, rep2 = build_panels(IDX, ["AAPL"], {"AAPL": no_cover}, {"AAPL": events})
    assert rep2.shares_source["AAPL"] == "diluted_weighted_average"
    assert "shares_source" in rep2.to_dict() and "sue_pending" in rep2.to_dict()


# ---------------------------------------------------------------------------
# 股本：逐字段过期；fallback 要看得见
# ---------------------------------------------------------------------------

def test_market_cap_expires_with_the_share_count_fact():
    facts, events = _aapl()
    cut = pd.Timestamp("2023-12-31")
    sh = facts["concept"].astype(str).str.startswith("shares")
    keep = ~sh | (pd.to_datetime(facts["filed"]) <= cut)               # 股本在 cut 之后不再更新
    sub = facts[keep]
    p, _ = build_panels(IDX, ["AAPL"], {"AAPL": sub}, {"AAPL": events}, raw_close=_raw())
    ends = metric_events(sub)["sh_end"].dropna()
    last = ends[ends <= cut].max()
    expired = IDX[IDX > last + pd.Timedelta(days=MAX_PERIOD_AGE_DAYS)]
    assert len(expired) > 0
    assert p["market_cap"]["AAPL"][expired].isna().all(), "股本事实早已过期，市值还在用它"
    assert p["market_cap"]["AAPL"][IDX[IDX < cut + pd.Timedelta(days=30)]].notna().any()


def test_the_report_counts_tickers_valued_with_an_estimated_share_count():
    facts, events = _aapl()
    no_cover = facts[~facts["concept"].astype(str).isin(["shares_cover", "shares_bs"])]
    _, rep = build_panels(IDX, ["AAPL"], {"AAPL": no_cover}, {"AAPL": events}, raw_close=_raw())
    assert rep.estimated_shares == ["AAPL"]
