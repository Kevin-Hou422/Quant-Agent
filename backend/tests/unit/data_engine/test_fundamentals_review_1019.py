"""
复审 2026-10-19 F13 余项 / §5.4 / D8（基本面）：

  "`_shares` 的 diluted weighted-average fallback 不等于期末股本；shares 的事实期间没有与其他指标同样的
   逐字段过期检查。小于 1.5 倍的疑似送股部分只告警，仍会影响市值/估值；SUE 事件仅按日期关系屏蔽，
   未明确匹配盈利期间。启用基本面交易前应加来源/期间质量门。"
  §3 模块表："保留来源、有效期间、fallback 和缺失标记；小比例公司行动必须进入异常隔离。"
  D8："先把 filing-based 因子名称、期间和事件配对写清。"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.fundamentals.panels import MAX_PERIOD_AGE_DAYS, build_panels, metric_events
from tests.unit.data_engine import fundamentals_fixtures as fx

IDX = pd.bdate_range("2023-07-03", "2024-11-27")


def _aapl():
    return fx.facts("AAPL"), fx.events("AAPL")


def _raw(idx=IDX, px=180.0):
    return pd.DataFrame({"AAPL": px}, index=idx)


# ---------------------------------------------------------------------------
# SUE：按盈利期间配对，不靠时间先后
# ---------------------------------------------------------------------------

def test_a_correction_8k_for_the_quarter_already_filed_does_not_blank_sue():
    """10-Q 已到之后、同一季的一份更正 8-K 2.02：按时间先后会把 SUE 置空到下一份申报（两个多月）。"""
    facts, events = _aapl()
    filed = sorted(pd.to_datetime(facts["filed"]).unique())
    q_filing = [f for f in filed if pd.Timestamp("2024-05-01") < f < pd.Timestamp("2024-06-30")][0]
    extra = events.iloc[[-1]].copy()
    extra["filing_date"] = q_filing + pd.Timedelta(days=10)
    extra["accn"] = "0000000000-24-999999"
    ev2 = pd.concat([events, extra]).sort_values("filing_date")
    p, rep = build_panels(IDX, ["AAPL"], {"AAPL": facts}, {"AAPL": ev2})
    window = IDX[(IDX > q_filing + pd.Timedelta(days=12)) & (IDX < pd.Timestamp("2024-07-25"))]
    assert len(window) > 20
    assert p["sue"]["AAPL"][window].notna().all(), "同一季的更正公告把已申报那季的 SUE 置空了"
    # 面板期内的公告都要配上财季（最早几条在任何财季披露之前，按时点推不出 —— 那是对的）
    inside = [e for e in rep.sue_pairing["AAPL"]["events"] if e["announced"] >= IDX[0]]
    assert inside and all(e["quarter_end"] is not None for e in inside)
    extra_q = [e for e in inside if e["announced"] == q_filing + pd.Timedelta(days=10)]
    assert extra_q and extra_q[0]["quarter_end"] < q_filing, "更正公告没有配到已申报的那一季"


def test_a_new_quarters_release_before_its_filing_still_blanks_sue():
    facts, events = _aapl()
    p, rep = build_panels(IDX, ["AAPL"], {"AAPL": facts}, {"AAPL": events})
    assert rep.sue_pending["AAPL"] > 0
    assert p["sue"]["AAPL"].notna().mean() > 0.95


def test_every_announcement_is_mapped_to_a_fiscal_quarter_in_the_report():
    facts, events = _aapl()
    _, rep = build_panels(IDX, ["AAPL"], {"AAPL": facts}, {"AAPL": events})
    m = [e for e in rep.sue_pairing["AAPL"]["events"] if e["announced"] >= IDX[0]]
    assert m and all(e["quarter_end"] is not None and e["quarter_end"] < e["announced"] for e in m)
    assert all(e["announced"] - e["quarter_end"] <= pd.Timedelta(days=100) for e in m)


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


# ---------------------------------------------------------------------------
# 小比例公司行动：异常隔离
# ---------------------------------------------------------------------------

def test_an_ambiguous_small_jump_quarantines_valuation_until_new_shares_arrive():
    facts, events = _aapl()
    jump = pd.Timestamp("2024-03-04")
    amb = {"AAPL": pd.Series({jump: 1.25})}
    p, rep = build_panels(IDX, ["AAPL"], {"AAPL": facts}, {"AAPL": events}, raw_close=_raw(), ambiguous=amb)
    mev = metric_events(facts)
    basis_after = mev.loc[mev["shares_basis"] > jump, "filed"].min()
    q = IDX[(IDX >= jump) & (IDX <= basis_after)]
    assert len(q) > 5
    for f in ("market_cap", "earnings_yield", "book_to_price"):
        assert p[f]["AAPL"][q].isna().all(), f"{f} 在疑似送股之后、新股本到来之前仍然有值"
    assert p["market_cap"]["AAPL"][IDX[IDX > basis_after + pd.Timedelta(days=5)]].notna().any()
    assert p["roe"]["AAPL"][q].notna().any(), "不依赖股本的字段不该被隔离"
    assert rep.quarantined["AAPL"]["events"] == [str(jump.date())]


def test_the_moomoo_path_reports_ambiguous_jumps_and_the_store_keeps_them(tmp_path):
    from app.core.data_engine.fundamentals import prices
    from app.core.data_engine.fundamentals.store import FundamentalsStore
    idx = pd.bdate_range("2024-06-03", periods=6)
    raw = pd.Series([100, 100, 100, 80, 80, 80], index=idx, dtype=float, name="X")
    adj = pd.Series([100, 100, 100, 100, 100, 100], index=idx, dtype=float, name="X")
    amb = prices.detect_ambiguous(raw, adj)
    assert list(amb.index) == [idx[3]] and amb.iloc[0] == pytest.approx(1.25)
    st = FundamentalsStore(tmp_path)
    st.save_raw_prices("moomoo", raw.to_frame(), {}, {"X": ["2024-06-03", "2024-06-10"]}, ambiguous={"X": amb})
    got = st.load_ambiguous("moomoo")
    assert list(got["X"].index) == [idx[3]]


# ---------------------------------------------------------------------------
# §5.4 来源 / 期间质量门
# ---------------------------------------------------------------------------

def test_the_quality_gate_names_each_problem():
    from app.core.data_engine.fundamentals.panels import FundamentalsReport
    from app.core.data_engine.fundamentals.quality import quality_problems
    rep = FundamentalsReport(coverage={"earnings_yield": 0.5, "roe": 0.95}, estimated_shares=["A", "B"],
                             n_tickers=10)
    probs = quality_problems(rep, ["earnings_yield"])
    assert any("覆盖" in p for p in probs) and any("股本" in p for p in probs)
    assert quality_problems(rep, ["roe"]) == []
    assert quality_problems(rep, []) == []


def test_fields_used_by_an_expression_are_found():
    from app.core.data_engine.fundamentals.quality import fundamental_fields_in
    assert fundamental_fields_in(["rank(earnings_yield) - rank(ts_delta(close, 5))"]) == ["earnings_yield"]
    assert fundamental_fields_in(["rank(close)"]) == []


def test_attach_saves_the_report_for_the_gate(tmp_path):
    from app.core.data_engine.fundamentals.store import FundamentalsStore
    from app.core.data_engine.fundamentals.panels import FundamentalsReport
    st = FundamentalsStore(tmp_path)
    st.save_report(FundamentalsReport(coverage={"roe": 0.9}, n_tickers=3), pd.Timestamp("2024-11-27"))
    rep, asof = st.load_report()
    assert asof == pd.Timestamp("2024-11-27") and rep.coverage == {"roe": 0.9} and rep.n_tickers == 3


def test_a_strategy_using_fundamentals_does_not_trade_without_a_passing_quality_report(tmp_path, monkeypatch):
    from app.config import settings
    from app.core.data_engine.fundamentals.quality import strategy_quality_problem
    monkeypatch.setattr(settings, "fundamentals_dir", str(tmp_path / "f"))
    last = pd.Timestamp("2024-11-27")
    assert "质量报告" in strategy_quality_problem({"1": "rank(earnings_yield)"}, last)
    assert strategy_quality_problem({"1": "rank(close)"}, last) == ""
