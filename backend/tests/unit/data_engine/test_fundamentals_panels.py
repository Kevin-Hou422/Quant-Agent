"""
Phase 10.1 / 10.2 / 10.3 — 事实 → 日频面板。

核心是**发布日无前视**（路线图 Phase 10 验收）：一条 F 日披露的事实，F 日及之前的面板值
不能因它而变。这里用三种方式钉它：
  1. 具体日期：AAPL 2023-11-03（周五）披露年报 → 11-03 仍是旧值，11-06（周一）才是新值；
  2. 前缀不变性：把"未来的事实"整个删掉，过去的面板逐位不变；
  3. 重述：只改重述披露日之后的值。
价格夹具是 moomoo / yahoo 在 NVDA 2024-06-10 一拆十前后的真实收盘价。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.fundamentals import panels as P
from app.core.data_engine.fundamentals import prices
from app.core.data_engine.fundamentals.panels import build_panels, earnings_days, metric_events
from tests.unit.data_engine import fundamentals_fixtures as fx

B = 1e9


@pytest.fixture(scope="module")
def facts():
    return {t: fx.facts(t) for t in ("AAPL", "NVDA")}


@pytest.fixture(scope="module")
def events():
    return {t: fx.events(t) for t in ("AAPL", "NVDA")}


@pytest.fixture(scope="module")
def daily(facts, events):
    idx = pd.bdate_range("2023-06-01", "2024-09-30")
    panels, rep = build_panels(idx, ["AAPL", "NVDA"], facts, events)
    return idx, panels, rep


# ---------------------------------------------------------------------------
# 可见性：披露日之后的第一个交易日
# ---------------------------------------------------------------------------

def test_filing_becomes_visible_on_the_next_trading_day_not_the_filing_day(daily):
    _, panels, _ = daily
    roe = panels["roe"]["AAPL"]
    old = 94.760 / 60.274          # 2023-08-04 的三季报：TTM 净利润 / 权益
    new = 96.995 / 62.146          # 2023-11-03（周五）的年报
    assert roe["2023-11-02"] == pytest.approx(old, rel=1e-5)
    assert roe["2023-11-03"] == pytest.approx(old, rel=1e-5)      # 披露当天：还看不到
    assert roe["2023-11-06"] == pytest.approx(new, rel=1e-5)      # 下一个交易日（跨周末）


def test_values_are_constant_between_filings(daily):
    _, panels, _ = daily
    seg = panels["gross_margin"]["AAPL"]["2023-11-06":"2024-02-02"]
    assert seg.nunique() == 1
    assert seg.iloc[0] == pytest.approx(169.148 / 383.285, rel=1e-5)


def test_no_lookahead_future_facts_do_not_change_the_past(facts, events, daily):
    """把 cut 之后披露的事实全部删掉，cut 及之前的面板必须逐位相同。"""
    idx, full, _ = daily
    for cut in ("2023-08-04", "2023-11-03", "2024-02-21", "2024-05-29"):
        cut_ts = pd.Timestamp(cut)
        known = {t: f[f["filed"] <= cut_ts] for t, f in facts.items()}
        ev_known = {t: e[e["filing_date"] <= cut_ts] for t, e in events.items()}
        past, _ = build_panels(idx, ["AAPL", "NVDA"], known, ev_known)
        for name in full:
            pd.testing.assert_frame_equal(past[name].loc[:cut_ts], full[name].loc[:cut_ts],
                                          check_exact=True, obj=f"{name}@{cut}")


def test_no_lookahead_has_teeth_the_day_after_the_cut_does_change(facts, events, daily):
    """上一条的反面：删掉的事实确实会改变 cut 之后的值 —— 否则那条测试什么都没验。"""
    idx, full, _ = daily
    cut = pd.Timestamp("2023-11-02")
    known = {t: f[f["filed"] <= cut] for t, f in facts.items()}
    past, _ = build_panels(idx, ["AAPL", "NVDA"], known, events)
    assert past["roe"]["AAPL"]["2023-11-06"] != pytest.approx(full["roe"]["AAPL"]["2023-11-06"])


def test_shifting_a_filing_date_shifts_visibility_by_exactly_that_much(facts, events):
    idx = pd.bdate_range("2023-09-01", "2023-12-29")
    f = facts["AAPL"].copy()
    late = f.copy()
    late.loc[late["filed"] == "2023-11-03", "filed"] = pd.Timestamp("2023-11-08")
    a, _ = build_panels(idx, ["AAPL"], {"AAPL": f}, {})
    b, _ = build_panels(idx, ["AAPL"], {"AAPL": late}, {})
    changed = idx[(a["roe"]["AAPL"] != b["roe"]["AAPL"]).values]
    assert [d.strftime("%m-%d") for d in changed] == ["11-06", "11-07", "11-08"]


def test_restatement_only_affects_dates_after_it_was_filed(facts):
    """NVDA 2024-01-28 的股本一年后被重述为拆股后口径：重述披露日（2025-02-26）之前的面板不受影响。"""
    idx = pd.bdate_range("2024-01-02", "2025-06-30")
    f = facts["NVDA"]
    restated = (f["concept"] == "shares_bs") & (f["end"] == "2024-01-28") & (f["filed"] == "2025-02-26")
    assert restated.sum() == 1
    raw = pd.DataFrame({"NVDA": 100.0}, index=idx)
    a, _ = build_panels(idx, ["NVDA"], {"NVDA": f}, {}, raw)
    b, _ = build_panels(idx, ["NVDA"], {"NVDA": f[~restated]}, {}, raw)
    for name in a:
        pd.testing.assert_series_equal(a[name]["NVDA"][:"2025-02-26"], b[name]["NVDA"][:"2025-02-26"])


def test_before_enough_history_is_filed_the_field_is_nan(facts):
    """夹具最早的披露日是 2019-01-30，但 TTM 要等到第一份年报（2019-10-31）才凑得齐。"""
    idx = pd.bdate_range("2018-06-01", "2019-12-31")
    p, _ = build_panels(idx, ["AAPL"], facts, {})
    roe = p["roe"]["AAPL"]
    assert roe[:"2019-10-31"].isna().all()
    assert roe["2019-11-01":].notna().all()
    assert roe["2019-11-01"] == pytest.approx(55.256 / 90.488, rel=1e-5)     # FY2019 净利润 / 权益


def test_stale_fundamentals_expire(facts):
    """最后一份财报之后超过 MAX_STALE_DAYS 没有新披露 → NaN（不能永远挂着最后一期的数）。"""
    f = facts["AAPL"][facts["AAPL"]["filed"] <= "2023-11-03"]
    idx = pd.bdate_range("2023-10-02", "2024-08-30")
    p, _ = build_panels(idx, ["AAPL"], {"AAPL": f}, {})
    roe = p["roe"]["AAPL"]
    last_ok = pd.Timestamp("2023-11-03") + pd.Timedelta(days=P.MAX_STALE_DAYS)
    assert roe[:last_ok].notna().all()
    assert roe[last_ok + pd.Timedelta(days=1):].isna().all()
    assert roe[last_ok] == roe["2023-11-06"]


def test_two_filings_on_the_same_day_use_everything_known_that_day():
    f = fx.facts("AAPL")
    extra = f[(f["concept"] == "equity") & (f["filed"] == "2023-11-03") & (f["end"] == "2023-09-30")].copy()
    extra["val"], extra["accn"] = 70.0 * B, "zz-amended"
    idx = pd.bdate_range("2023-11-01", "2023-11-10")
    p, _ = build_panels(idx, ["AAPL"], {"AAPL": pd.concat([f, extra])}, {})
    assert p["roe"]["AAPL"]["2023-11-06"] == pytest.approx(96.995 / 70.0, rel=1e-5)


# ---------------------------------------------------------------------------
# 各字段的定义
# ---------------------------------------------------------------------------

def test_metric_events_match_the_reported_numbers(facts):
    m = metric_events(facts["AAPL"]).set_index("filed")
    r = m.loc["2024-02-02"]
    assert r["ni_ttm"] == pytest.approx(100.913 * B)
    assert r["rev_ttm"] == pytest.approx(385.706 * B)
    assert r["gp_ttm"] == pytest.approx(173.671 * B)
    assert r["equity"] == pytest.approx(74.100 * B)
    assert r["assets"] == pytest.approx(353.514 * B)
    assert r["assets_prev"] == pytest.approx(346.747 * B)
    assert r["shares"] == pytest.approx(15.44188e9, rel=1e-6)
    assert r["shares_basis"] == pd.Timestamp("2024-01-19")       # 封面页股本的"截至日"


def test_asset_growth_is_year_over_year(daily):
    _, panels, _ = daily
    assert panels["asset_growth"]["AAPL"]["2024-02-05"] == pytest.approx(353.514 / 346.747 - 1, rel=1e-6)


def test_negative_equity_gives_nan_roe_not_a_flipped_sign():
    f = fx.facts("AAPL").copy()
    f.loc[f["concept"] == "equity", "val"] = -5.0 * B
    idx = pd.bdate_range("2024-01-02", "2024-03-29")
    p, _ = build_panels(idx, ["AAPL"], {"AAPL": f}, {})
    assert p["roe"]["AAPL"].isna().all()
    assert p["gross_margin"]["AAPL"].notna().all()


def test_ticker_without_facts_is_nan_and_reported(facts, events):
    idx = pd.bdate_range("2024-01-02", "2024-03-29")
    p, rep = build_panels(idx, ["AAPL", "SPY"], facts, events)
    assert p["roe"]["SPY"].isna().all()
    assert rep.no_facts == ["SPY"]
    assert rep.coverage["roe"] == 0.5
    assert list(p["roe"].columns) == ["AAPL", "SPY"]


def test_empty_fact_frame_counts_as_no_facts(facts):
    idx = pd.bdate_range("2024-01-02", "2024-01-31")
    _, rep = build_panels(idx, ["AAPL", "NVDA"], {"AAPL": facts["AAPL"], "NVDA": facts["NVDA"].iloc[0:0]}, {})
    assert rep.no_facts == ["NVDA"]


# ---------------------------------------------------------------------------
# 财报事件日
# ---------------------------------------------------------------------------

def test_days_since_earnings_resets_the_trading_day_after_the_announcement(daily):
    _, panels, _ = daily
    d = panels["days_since_earnings"]["NVDA"]
    # NVDA 2024-05-22 盘后公告（8-K 申报日 05-22）→ 05-23 记 0
    assert d["2024-05-22"] > 50
    assert d["2024-05-23"] == 0
    assert d["2024-05-24"] == 1
    assert d["2024-06-07"] == 11         # 数的是面板的行（这里是工作日序列），不是日历日


def test_days_since_earnings_counts_index_rows_not_calendar_days():
    idx = pd.DatetimeIndex(["2024-05-22", "2024-05-23", "2024-05-24", "2024-05-28", "2024-05-29"])
    ev = pd.DataFrame({"filing_date": [pd.Timestamp("2024-05-22")]})
    assert earnings_days(ev, idx).tolist() == [pytest.approx(np.nan, nan_ok=True), 0, 1, 2, 3]


def test_announcement_on_a_non_trading_day_takes_effect_next_session():
    idx = pd.bdate_range("2024-05-20", "2024-05-31")
    ev = pd.DataFrame({"filing_date": [pd.Timestamp("2024-05-25")]})      # 周六
    d = earnings_days(ev, idx)
    assert d["2024-05-24"] != 0 and d["2024-05-27"] == 0


def test_events_long_before_the_panel_start_are_not_counted():
    """起点之前的公告隔了多少个交易日数不出来 → 不用，等面板内第一次公告。"""
    idx = pd.bdate_range("2024-05-20", "2024-06-20")
    old = pd.DataFrame({"filing_date": [pd.Timestamp("2024-02-21")]})
    assert earnings_days(old, idx).isna().all()
    both = pd.DataFrame({"filing_date": pd.to_datetime(["2024-02-21", "2024-05-22"])})
    d = earnings_days(both, idx)
    assert d[:"2024-05-22"].isna().all() and d["2024-05-23"] == 0


def test_announcement_just_before_the_panel_start_is_day_zero():
    idx = pd.bdate_range("2024-05-20", "2024-05-24")                      # 周一起
    fri = pd.DataFrame({"filing_date": [pd.Timestamp("2024-05-17")]})     # 上周五
    assert earnings_days(fri, idx).tolist() == [0, 1, 2, 3, 4]
    five_days = pd.DataFrame({"filing_date": [pd.Timestamp("2024-05-15")]})
    assert earnings_days(five_days, idx).isna().all()


def test_announcement_after_the_last_row_is_ignored():
    idx = pd.bdate_range("2024-05-20", "2024-05-24")
    ev = pd.DataFrame({"filing_date": pd.to_datetime(["2024-05-24", "2024-06-30"])})
    assert earnings_days(ev, idx).isna().all()
    assert earnings_days(ev.iloc[0:0], idx).isna().all()


# ---------------------------------------------------------------------------
# 市值与拆股（真实价格）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mm():
    df = fx.moomoo_prices()
    raw, qfq = df["raw"], df["qfq"]
    splits = {t: prices.detect_splits(raw[t], qfq[t]) for t in raw.columns}
    return raw, qfq, {t: s for t, s in splits.items() if len(s)}


def test_valuation_fields_are_absent_without_unadjusted_prices(facts, events):
    idx = pd.bdate_range("2024-05-20", "2024-06-20")
    p, rep = build_panels(idx, ["AAPL", "NVDA"], facts, events)
    assert "earnings_yield" not in p and "book_to_price" not in p
    assert set(p) == set(P.ALL_FIELDS) - set(P.VALUATION_FIELDS)
    assert "earnings_yield" not in rep.coverage


def test_earnings_yield_matches_hand_computed_market_cap(facts, events, mm):
    raw, _, splits = mm
    p, _ = build_panels(raw.index, ["AAPL", "NVDA"], facts, events, raw, splits)
    # 2024-05-31：可见的是 05-29 的 10-Q —— TTM 净利润 42.598B，封面页股本 24.6 亿（截至 05-24）
    mcap = 1096.33 * 2.46e9
    assert p["earnings_yield"]["NVDA"]["2024-05-31"] == pytest.approx(42.598e9 / mcap, rel=1e-6)
    assert p["book_to_price"]["NVDA"]["2024-05-31"] == pytest.approx(49.142e9 / mcap, rel=1e-6)


def test_market_cap_is_continuous_across_the_ten_for_one_split(facts, events, mm):
    """
    06-07 收 1208.88，06-10 收 121.79（一拆十）。已知股本还是拆前的 24.6 亿，下一份财报 08-28 才出。
    不调整的话 06-10 的市值掉到十分之一、E/P 跳 10 倍。
    """
    raw, qfq, splits = mm
    p, rep = build_panels(raw.index, ["NVDA"], facts, events, raw, splits)
    ey = p["earnings_yield"]["NVDA"]
    implied_return = ey["2024-06-07"] / ey["2024-06-10"]           # 市值之比
    true_return = qfq["NVDA"]["2024-06-10"] / qfq["NVDA"]["2024-06-07"]
    assert implied_return == pytest.approx(true_return, rel=1e-4)
    assert rep.splits == {"NVDA": {"2024-06-10": 10.0}}

    naive, _ = build_panels(raw.index, ["NVDA"], facts, events, raw, {})
    jump = naive["earnings_yield"]["NVDA"]["2024-06-10"] / naive["earnings_yield"]["NVDA"]["2024-06-07"]
    assert jump == pytest.approx(10.0, rel=0.02)                    # 没有拆股记录时确实会错 10 倍


def test_split_before_the_share_count_date_is_not_applied_twice(facts, events):
    """08-28 的 10-Q 封面页股本（截至 08-23）已是拆后的 245 亿 —— 不能再乘一次 10。"""
    idx = pd.bdate_range("2024-08-01", "2024-09-30")
    raw = pd.DataFrame({"NVDA": 120.0}, index=idx)
    splits = {"NVDA": pd.Series([10.0], index=pd.DatetimeIndex(["2024-06-10"]))}
    p, _ = build_panels(idx, ["NVDA"], facts, events, raw, splits)
    assert p["earnings_yield"]["NVDA"]["2024-08-28"] == pytest.approx(42.598e9 / (120.0 * 2.46e9 * 10), rel=1e-6)
    assert p["earnings_yield"]["NVDA"]["2024-08-29"] == pytest.approx(53.008e9 / (120.0 * 24.53e9), rel=1e-6)


def test_split_on_the_share_count_date_itself_is_treated_as_already_reflected():
    basis = np.array(["2024-06-10", "2024-06-07"], dtype="datetime64[ns]")
    dates = pd.DatetimeIndex(["2024-06-11", "2024-06-11"])
    splits = pd.Series([10.0], index=pd.DatetimeIndex(["2024-06-10"]))
    assert P._split_factor(splits, basis, dates).tolist() == [1.0, 10.0]


def test_multiple_splits_compound():
    splits = pd.Series([2.0, 0.5, 4.0], index=pd.DatetimeIndex(["2024-02-01", "2024-03-01", "2024-04-01"]))
    basis = np.array(["2024-01-15"] * 4, dtype="datetime64[ns]")
    dates = pd.DatetimeIndex(["2024-01-31", "2024-02-01", "2024-03-15", "2024-04-01"])
    assert P._split_factor(splits, basis, dates).tolist() == [1.0, 2.0, 1.0, 4.0]
    assert P._split_factor(splits.iloc[::-1], basis, dates).tolist() == [1.0, 2.0, 1.0, 4.0]   # 乱序输入
    assert P._split_factor(None, basis, dates).tolist() == [1.0] * 4


def test_absurd_valuation_is_rejected_and_reported(facts, events, caplog):
    """股本口径错了（多类别股只报了一类等）会让 E/P 离谱 —— 置 NaN 并记入报告，不让它进截面排名。"""
    idx = pd.bdate_range("2024-03-01", "2024-03-29")
    raw = pd.DataFrame({"AAPL": 180.0, "NVDA": 0.004}, index=idx)     # NVDA 价格错到市值只有 1000 万
    with caplog.at_level(logging.WARNING):
        p, rep = build_panels(idx, ["AAPL", "NVDA"], facts, events, raw, {})
    assert p["earnings_yield"]["NVDA"].isna().all() and p["book_to_price"]["NVDA"].isna().all()
    assert p["earnings_yield"]["AAPL"].notna().all()
    assert rep.sanity_rejected == {"NVDA": len(idx)}
    assert fx.logged(caplog, "NVDA")


def test_sanity_bounds_are_inclusive():
    """恰好等于上限的不算离谱。"""
    f = fx.facts("AAPL")
    idx = pd.bdate_range("2024-03-01", "2024-03-05")
    m = metric_events(f[f["filed"] <= "2024-02-02"]).iloc[-1]
    price_at_limit = m["ni_ttm"] / (P.MAX_ABS_EARNINGS_YIELD * m["shares"])
    for px, rejected in ((price_at_limit * 1.0001, False), (price_at_limit * 0.999, True)):
        _, rep = build_panels(idx, ["AAPL"], {"AAPL": f}, {}, pd.DataFrame({"AAPL": px}, index=idx), {})
        assert bool(rep.sanity_rejected) is rejected


def test_missing_price_gives_nan_valuation_but_keeps_other_fields(facts, events):
    idx = pd.bdate_range("2024-03-01", "2024-03-29")
    raw = pd.DataFrame({"AAPL": 180.0}, index=idx)                    # 没有 NVDA 列
    p, _ = build_panels(idx, ["AAPL", "NVDA"], facts, events, raw, {})
    assert p["earnings_yield"]["NVDA"].isna().all()
    assert p["roe"]["NVDA"].notna().all()
    assert p["earnings_yield"]["AAPL"].notna().all()


def test_report_serialises():
    _, rep = build_panels(pd.bdate_range("2024-03-01", "2024-03-05"), ["AAPL"], {"AAPL": fx.facts("AAPL")}, {})
    d = rep.to_dict()
    assert d["coverage"]["roe"] == 1.0 and d["no_facts"] == [] and d["splits"] == {}


def test_precomputed_metrics_give_the_same_panels(facts, events):
    idx = pd.bdate_range("2023-06-01", "2024-06-28")
    a, _ = build_panels(idx, ["AAPL", "NVDA"], facts, events)
    pre = {t: metric_events(f) for t, f in facts.items()}
    b, _ = build_panels(idx, ["AAPL", "NVDA"], facts, events, metrics=pre)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


# ---------------------------------------------------------------------------
# market_cap 字段（size 因子 / 风险模型的 size 风格要用）
# ---------------------------------------------------------------------------

def test_market_cap_is_unadjusted_price_times_split_adjusted_shares(facts, events, mm):
    raw, _, splits = mm
    p, _ = build_panels(raw.index, ["AAPL", "NVDA"], facts, events, raw, splits)
    mc = p["market_cap"]["NVDA"]
    assert mc["2024-05-31"] == pytest.approx(1096.33 * 2.46e9, rel=1e-9)          # 拆股前
    assert mc["2024-06-10"] == pytest.approx(121.79 * 2.46e9 * 10, rel=1e-9)      # 拆股日：价格 /10，股本 ×10
    assert mc["2024-06-10"] / mc["2024-06-07"] == pytest.approx(121.79 * 10 / 1208.88, rel=1e-9)
    # 与另外两个估值字段是同一个市值
    assert p["earnings_yield"]["NVDA"]["2024-05-31"] * mc["2024-05-31"] == pytest.approx(42.598e9, rel=1e-9)


def test_market_cap_exists_even_when_earnings_history_is_too_short(mm):
    """市值只要股本和价格；TTM 净利润凑不齐（E/P 为空）不影响它。"""
    raw, _, splits = mm
    n = fx.facts("NVDA")
    no_income = n[n["concept"] != "net_income"]
    p, _ = build_panels(raw.index, ["NVDA"], {"NVDA": no_income}, {}, raw, splits)
    assert p["earnings_yield"]["NVDA"].isna().all()
    assert p["market_cap"]["NVDA"].notna().all()


def test_market_cap_is_blanked_together_with_the_ratios_on_rejected_days(facts, events):
    idx = pd.bdate_range("2024-03-01", "2024-03-29")
    raw = pd.DataFrame({"AAPL": 180.0, "NVDA": 0.004}, index=idx)
    p, rep = build_panels(idx, ["AAPL", "NVDA"], facts, events, raw, {})
    assert rep.sanity_rejected == {"NVDA": len(idx)}
    assert p["market_cap"]["NVDA"].isna().all()
    assert p["market_cap"]["AAPL"].iloc[-1] == pytest.approx(180.0 * 15.44188e9, rel=1e-6)


def test_market_cap_is_absent_without_unadjusted_prices(facts, events):
    p, rep = build_panels(pd.bdate_range("2024-03-01", "2024-03-08"), ["AAPL"], facts, events)
    assert "market_cap" not in p and "market_cap" not in rep.coverage
