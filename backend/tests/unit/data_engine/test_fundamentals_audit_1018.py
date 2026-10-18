"""
外部审查 2026-10-18 F13：基本面字段的时间正确不等于经济口径正确。

  · SUE 等 10-Q，days_since_earnings 按 8-K 2.02 归零 —— 新 8-K 已到、10-Q 未到时，
    `days_since_earnings < 60` 会选中**上一季**的 SUE。
  · 过期按"最近一次披露"统一算：只更新股本的披露会让一年前的利润数字继续算新鲜。
  · 稀释加权平均股数是兜底，不是期末股本 —— 来源要看得见。
  · 拆股识别是启发式：复权因子跳了、不复权价没动（数据拼接问题）不能当拆股。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.data_engine.fundamentals.panels import (
    MAX_PERIOD_AGE_DAYS, SHARES_SOURCES, build_panels, metric_events,
)
from app.core.data_engine.fundamentals.prices import detect_splits
from tests.unit.data_engine import fundamentals_fixtures as fx

IDX = pd.bdate_range("2023-07-03", "2024-11-27")


def _aapl():
    return fx.facts("AAPL"), fx.events("AAPL")


def test_sue_is_blank_between_a_new_earnings_release_and_its_filing():
    facts, events = _aapl()
    p, rep = build_panels(IDX, ["AAPL"], {"AAPL": facts}, {"AAPL": events})
    sue, dse = p["sue"]["AAPL"], p["days_since_earnings"]["AAPL"]
    released = [d for d in IDX if dse.get(d) == 0]                     # 新公告生效的那天
    assert released
    filed = sorted(pd.to_datetime(facts["filed"]).unique())
    checked = 0
    for d in released:
        prev_10q = max(f for f in filed if f < d)
        nxt = [f for f in filed if f >= d - pd.Timedelta(days=1)]
        if nxt and nxt[0] > d - pd.Timedelta(days=1):                  # 申报晚于公告：中间这几天 SUE 为空
            assert np.isnan(sue[d]), f"{d.date()} 公告已出、申报未到，却给出了上一季（{prev_10q.date()}）的 SUE"
            checked += 1
    assert checked >= 3
    assert rep.sue_pending["AAPL"] > 0
    assert sue.notna().mean() > 0.95                                   # 只空公告到申报之间那一两天


def test_old_earnings_do_not_stay_fresh_because_a_later_filing_updated_shares():
    facts, events = _aapl()
    cut = pd.Timestamp("2023-12-31")
    keep = (pd.to_datetime(facts["filed"]) <= cut) | facts["concept"].astype(str).str.startswith("shares")
    p, _ = build_panels(IDX, ["AAPL"], {"AAPL": facts[keep]}, {"AAPL": events})
    ends = metric_events(facts[keep])["ni_end"].dropna()
    last_ni_end = ends[ends <= cut].max()
    roe = p["roe"]["AAPL"]
    expired = IDX[IDX > last_ni_end + pd.Timedelta(days=MAX_PERIOD_AGE_DAYS)]
    assert len(expired) > 0 and roe[expired].isna().all(), "期末日早已过期的利润仍在给 ROE"
    assert roe[IDX[IDX < cut + pd.Timedelta(days=30)]].notna().any()


def test_the_report_names_the_share_count_source():
    facts, events = _aapl()
    _, rep = build_panels(IDX, ["AAPL"], {"AAPL": facts}, {"AAPL": events})
    assert rep.shares_source["AAPL"] in SHARES_SOURCES.values()
    no_cover = facts[~facts["concept"].astype(str).isin(["shares_cover", "shares_bs"])]
    _, rep2 = build_panels(IDX, ["AAPL"], {"AAPL": no_cover}, {"AAPL": events})
    assert rep2.shares_source["AAPL"] == "diluted_weighted_average"
    assert "shares_source" in rep2.to_dict() and "sue_pending" in rep2.to_dict()


def _series(vals):
    return pd.Series(vals, index=pd.bdate_range("2024-06-05", periods=len(vals)), name="X", dtype=float)


def test_a_real_split_moves_the_raw_price_the_other_way():
    raw, adj = _series([1200, 1210, 121, 122]), _series([120, 121, 121, 122])
    s = detect_splits(raw, adj)
    assert list(s.round(2)) == [10.0] and s.index[0] == raw.index[2]


def test_a_factor_jump_without_a_raw_price_jump_is_not_a_split(caplog):
    raw, adj = _series([100, 101, 100, 101]), _series([10, 10.1, 100, 101])     # 两个 vintage 拼在一起
    with caplog.at_level("WARNING"):
        assert detect_splits(raw, adj).empty
    assert any("没有伴随不复权价的反向跳变" in r.getMessage() for r in caplog.records)


def test_an_ambiguous_small_jump_is_reported_not_applied(caplog):
    raw, adj = _series([100, 100, 77, 77]), _series([100, 100, 100, 100])       # 1.3 倍：送股？特别分红？
    with caplog.at_level("WARNING"):
        assert detect_splits(raw, adj).empty
    assert any("小比例送股或特别分红" in r.getMessage() for r in caplog.records)
