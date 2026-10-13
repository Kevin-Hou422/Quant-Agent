"""
Phase 10 — 第一次对 us_broad_large 全部 95 只票跑真实 EDGAR（2026-10-12）时暴露的问题，逐条钉住。

只用 AAPL / NVDA 两家做夹具时，下面这些一个都看不出来：
  1. 多数公司不报 `GrossProfit`（95 只里毛利率覆盖率只有 29.5%）→ 用 收入 − 营业成本 兜底；
  2. BKNG 2016 年后的净利润只在 `NetIncomeLossAvailableToCommonStockholdersBasic` 下；
  3. 净资产为负的公司（BKNG、MCD、SBUX、PM …）B/P 是负数，混进截面排名没有意义；
  4. GOOGL 是多类别股：没有封面页股本，只有资产负债表上的合计股数；
  5. XOM 2026-07 重组换了 CIK，历史财报在旧主体名下；
  6. Citi 的 2026 年季报 SEC 的 companyfacts 接口没有收进来 —— 过期保护把它置空是对的，
     但必须在报告里点名，否则只是覆盖率悄悄少了一格；
  7. 落盘的是**解析后**的事实：解析口径变了（加科目）而缓存没过期时，必须重新解析。
夹具仍是裁剪后的真实响应。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.fundamentals import edgar, pit
from app.core.data_engine.fundamentals.panels import MAX_STALE_DAYS, build_panels, metric_events
from app.core.data_engine.fundamentals.service import sync
from app.core.data_engine.fundamentals.store import FundamentalsStore
from tests.unit.data_engine import fundamentals_fixtures as fx

B = 1e9


# ---------------------------------------------------------------------------
# 1. 毛利：收入 − 营业成本
# ---------------------------------------------------------------------------

def test_gross_profit_falls_back_to_revenue_minus_cost_of_revenue():
    """Alphabet 2023：收入 307.394B，营业成本 133.332B（10-K 原数）→ 毛利 174.062B。"""
    g = fx.facts("GOOGL")
    assert "gross_profit" not in set(g["concept"])                 # 这家公司确实不报 GrossProfit
    m = metric_events(g).set_index("filed").loc["2024-01-31"]
    assert m["rev_ttm"] == pytest.approx(307.394 * B)
    assert m["gp_ttm"] == pytest.approx((307.394 - 133.332) * B)
    idx = pd.bdate_range("2024-02-01", "2024-02-29")
    p, _ = build_panels(idx, ["GOOGL"], {"GOOGL": g}, {})
    assert p["gross_margin"]["GOOGL"].iloc[-1] == pytest.approx(174.062 / 307.394, rel=1e-6)


def test_reported_gross_profit_wins_over_the_fallback():
    """AAPL 两个都报 —— 直接披露的毛利优先，不用减法重算。"""
    a = fx.facts("AAPL")
    assert {"gross_profit", "cost_of_revenue"} <= set(a["concept"])
    tampered = a.copy()
    tampered.loc[tampered["concept"] == "cost_of_revenue", "val"] = 1.0     # 减法会得出离谱的数
    m = metric_events(tampered).set_index("filed").loc["2024-02-02"]
    assert m["gp_ttm"] == pytest.approx(173.671 * B)


def test_fallback_requires_cost_and_revenue_for_the_same_period():
    g = fx.facts("GOOGL")
    # 去掉最新一期的营业成本：收入到了 2023-12-31，成本只到 2023-09-30 → 不能相减
    stale_cost = g[~((g["concept"] == "cost_of_revenue") & (g["end"] == "2023-12-31"))]
    vis = stale_cost[stale_cost["filed"] <= "2024-01-31"]
    assert pit.ttm(vis, "revenue")[1] != pit.ttm(vis, "cost_of_revenue")[1]
    m = metric_events(stale_cost).set_index("filed").loc["2024-01-31"]
    assert np.isnan(m["gp_ttm"])


def test_no_cost_and_no_gross_profit_means_no_margin():
    b = fx.facts("BKNG")
    assert not {"gross_profit", "cost_of_revenue"} & set(b["concept"])
    assert metric_events(b)["gp_ttm"].isna().all()


# ---------------------------------------------------------------------------
# 2. 净利润的第三个科目
# ---------------------------------------------------------------------------

def test_net_income_is_found_under_the_available_to_common_tag():
    """Booking 2024 年净利润 5.882B —— 只出现在第三顺位的科目下。"""
    b = fx.facts("BKNG")
    assert set(b[b["concept"] == "net_income"]["tag"]) == {"NetIncomeLossAvailableToCommonStockholdersBasic"}
    val, end = pit.ttm(b[b["filed"] <= "2025-02-20"], "net_income")
    assert val == pytest.approx(5.882 * B) and end == pd.Timestamp("2024-12-31")


def test_plain_net_income_still_takes_priority_when_both_exist():
    """GOOGL 两个科目都报 → 用 NetIncomeLoss。"""
    raw = fx.payload("companyfacts", "GOOGL")["facts"]["us-gaap"]
    assert {"NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"} <= set(raw)
    g = fx.facts("GOOGL")
    assert set(g[g["concept"] == "net_income"]["tag"]) == {"NetIncomeLoss"}
    assert pit.ttm(g[g["filed"] <= "2024-01-31"], "net_income")[0] == pytest.approx(73.795 * B)


# ---------------------------------------------------------------------------
# 3. 负净资产
# ---------------------------------------------------------------------------

def test_negative_book_equity_blanks_book_to_price_and_roe_but_not_earnings_yield():
    b, ev = fx.facts("BKNG"), fx.events("BKNG")
    idx = pd.bdate_range("2025-03-03", "2025-03-31")
    raw = pd.DataFrame({"BKNG": 4800.0}, index=idx)
    p, rep = build_panels(idx, ["BKNG"], {"BKNG": b}, {"BKNG": ev}, raw, {})
    assert p["book_to_price"]["BKNG"].isna().all()                 # 权益 −4.02B
    assert p["roe"]["BKNG"].isna().all()
    assert p["earnings_yield"]["BKNG"].iloc[-1] == pytest.approx(5.882e9 / (4800.0 * 32_815_201), rel=1e-6)
    assert rep.sanity_rejected == {}


def test_zero_equity_is_treated_like_negative():
    a = fx.facts("AAPL").copy()
    a.loc[a["concept"] == "equity", "val"] = 0.0
    idx = pd.bdate_range("2024-03-01", "2024-03-08")
    p, _ = build_panels(idx, ["AAPL"], {"AAPL": a}, {}, pd.DataFrame({"AAPL": 180.0}, index=idx), {})
    assert p["book_to_price"]["AAPL"].isna().all() and p["earnings_yield"]["AAPL"].notna().all()


# ---------------------------------------------------------------------------
# 4. 多类别股、以及第二次真实拆股
# ---------------------------------------------------------------------------

def test_multi_class_company_uses_the_balance_sheet_total():
    """GOOGL 没有封面页股本（按类别分开报，companyfacts 不收带维度的事实）→ 用资产负债表合计。"""
    g = fx.facts("GOOGL")
    assert "shares_cover" not in set(g["concept"])
    m = metric_events(g).set_index("filed")
    assert m.loc["2024-01-31", "shares"] == pytest.approx(12.460e9)       # A+B+C 合计
    assert m.loc["2024-01-31", "shares_basis"] == pd.Timestamp("2024-01-31")   # 重述口径 → 披露日


def test_alphabet_twenty_for_one_split_keeps_market_cap_continuous():
    """
    2022-04-27 的 10-Q：6.59 亿股（拆前）。07-18 一拆二十，07-27 的 10-Q 才报出 130.78 亿股。
    这九天里股价已是拆后的。
    """
    g = fx.facts("GOOGL")
    idx = pd.bdate_range("2022-07-11", "2022-08-05")
    px = pd.Series(np.where(idx < "2022-07-18", 2300.0, 115.0), index=idx)
    splits = {"GOOGL": pd.Series([20.0], index=pd.DatetimeIndex(["2022-07-18"]))}
    p, _ = build_panels(idx, ["GOOGL"], {"GOOGL": g}, {}, pd.DataFrame({"GOOGL": px}), splits)
    ey = p["earnings_yield"]["GOOGL"]
    mcap = 74.539e9 / ey                                            # TTM 净利润 / E/P
    assert mcap["2022-07-15"] == pytest.approx(2300.0 * 658.763e6, rel=1e-6)
    assert mcap["2022-07-18"] == pytest.approx(115.0 * 658.763e6 * 20, rel=1e-6)
    assert mcap["2022-07-18"] / mcap["2022-07-15"] == pytest.approx(1.0, rel=1e-9)   # 价格 /20，股本 ×20
    # 07-28 起用新 10-Q：净利润与股本都换了，股本已是拆后口径，不再乘 20
    assert ey["2022-07-28"] == pytest.approx(72.016e9 / (115.0 * 13.078e9), rel=1e-6)


def test_booking_twenty_five_for_one_split():
    b = fx.facts("BKNG")
    idx = pd.bdate_range("2026-03-23", "2026-05-08")
    px = pd.Series(np.where(idx < "2026-04-06", 5000.0, 200.0), index=idx)
    splits = {"BKNG": pd.Series([25.0], index=pd.DatetimeIndex(["2026-04-06"]))}
    p, _ = build_panels(idx, ["BKNG"], {"BKNG": b}, {}, pd.DataFrame({"BKNG": px}), splits)
    ey = p["earnings_yield"]["BKNG"]
    assert ey["2026-04-02"] == pytest.approx(ey["2026-04-06"], rel=1e-9)
    assert ey["2026-04-06"] == pytest.approx(5.404e9 / (200.0 * 31_673_346 * 25), rel=1e-6)
    # 04-28 的 10-Q 封面页已是拆后的 7.75 亿股（截至 04-20）
    assert ey["2026-04-29"] == pytest.approx(6.154e9 / (200.0 * 774_878_436), rel=1e-6)


def test_no_share_count_at_all_gives_nan_valuation_not_a_guess():
    """BRK-B、V 这类公司在 companyfacts 里没有可用的股本 → 估值字段为空，不拿别的数凑。"""
    a = fx.facts("AAPL")
    no_shares = a[~a["concept"].str.startswith("shares")]
    idx = pd.bdate_range("2024-03-01", "2024-03-08")
    p, rep = build_panels(idx, ["AAPL"], {"AAPL": no_shares}, {}, pd.DataFrame({"AAPL": 180.0}, index=idx), {})
    assert p["earnings_yield"]["AAPL"].isna().all() and p["book_to_price"]["AAPL"].isna().all()
    assert p["roe"]["AAPL"].notna().all()
    assert rep.coverage["earnings_yield"] == 0.0 and rep.coverage["roe"] == 1.0


# ---------------------------------------------------------------------------
# 5. 继任主体（换 CIK）
# ---------------------------------------------------------------------------

def test_predecessor_registrant_facts_are_merged(tmp_path, monkeypatch):
    """
    XOM 现在指向 2026 年新设的控股公司，历史在旧 CIK 34088 名下。
    这里用 BKNG / AAPL 的真实响应分别充当"新主体 / 旧主体"来验合并机制本身。
    """
    assert edgar.PREDECESSOR_CIKS["XOM"] == (34088,)
    sess = fx.FakeSession()
    new_cik = fx.CIK["XOM"]
    assert new_cik == 2115436
    sess.routes[edgar.FACTS_URL.format(cik=new_cik)] = fx.payload("companyfacts", "BKNG")
    sess.routes[edgar.SUBMISSIONS_URL.format(cik=new_cik)] = fx.payload("submissions", "BKNG")
    sess.routes[edgar.FACTS_URL.format(cik=34088)] = fx.payload("companyfacts", "AAPL")
    st = FundamentalsStore(tmp_path)
    rep = sync(["XOM"], fx.client(sess), st, now=1.0)
    assert rep.synced == ["XOM"]
    facts, _ = st.load("XOM")
    assert len(facts) == len(fx.facts("BKNG")) + len(fx.facts("AAPL"))
    assert (facts["ticker"] == "XOM").all()
    assert edgar.FACTS_URL.format(cik=34088) in sess.calls


def test_tickers_without_a_registered_predecessor_make_no_extra_request(tmp_path):
    sess = fx.FakeSession()
    sync(["AAPL"], fx.client(sess), FundamentalsStore(tmp_path), now=1.0)
    assert [u for u in sess.calls if "companyfacts" in u] == [edgar.FACTS_URL.format(cik=fx.CIK["AAPL"])]
    assert fx.client().predecessor_facts("AAPL") == []


def test_missing_predecessor_fails_that_ticker_loudly(tmp_path):
    """登记了前任却取不到 → 这只票同步失败（不悄悄只用新主体那几个月的数据）。"""
    sess = fx.FakeSession()
    new_cik = fx.CIK["XOM"]
    sess.routes[edgar.FACTS_URL.format(cik=new_cik)] = fx.payload("companyfacts", "BKNG")
    sess.routes[edgar.SUBMISSIONS_URL.format(cik=new_cik)] = fx.payload("submissions", "BKNG")
    rep = sync(["XOM"], fx.client(sess), FundamentalsStore(tmp_path), now=1.0)
    assert "XOM" in rep.failed and "404" in rep.failed["XOM"]


# ---------------------------------------------------------------------------
# 6. 上游断档 → 过期 → 点名
# ---------------------------------------------------------------------------

def test_stale_ticker_is_named_in_the_report():
    a, n = fx.facts("AAPL"), fx.facts("NVDA")
    stopped = a[a["filed"] <= "2025-01-31"]                       # 之后再没有财报进来（Citi 的情形）
    idx = pd.bdate_range("2025-06-02", "2025-12-31")
    p, rep = build_panels(idx, ["AAPL", "NVDA"], {"AAPL": stopped, "NVDA": n}, {})
    assert rep.stale == {"AAPL": "2025-01-31"}
    assert np.isnan(p["roe"]["AAPL"].iloc[-1]) and not np.isnan(p["roe"]["NVDA"].iloc[-1])
    assert rep.to_dict()["stale"] == {"AAPL": "2025-01-31"}
    assert rep.coverage["roe"] == 0.5


def test_stale_boundary_is_exactly_max_stale_days():
    a = fx.facts("AAPL")
    stopped = a[a["filed"] <= "2025-01-31"]
    last = pd.Timestamp("2025-01-31")
    for extra, is_stale in ((0, False), (1, True)):
        end = last + pd.Timedelta(days=MAX_STALE_DAYS + extra)
        idx = pd.date_range(end - pd.Timedelta(days=5), end, freq="D")
        _, rep = build_panels(idx, ["AAPL"], {"AAPL": stopped}, {})
        assert bool(rep.stale) is is_stale


def test_stale_tickers_are_logged_by_the_service(tmp_path, caplog):
    from types import SimpleNamespace
    from app.core.data_engine.fundamentals.service import attach_fundamentals
    a = fx.facts("AAPL")
    FundamentalsStore(tmp_path).save("AAPL", a[a["filed"] <= "2025-01-31"], fx.events("AAPL"))
    idx = pd.bdate_range("2025-10-01", "2025-12-31")
    settings = SimpleNamespace(pit_store_dir=str(tmp_path), fundamentals_dir=str(tmp_path),
                               edgar_user_agent="", fundamentals_max_age_h=20.0, price_source="yahoo")
    data = {"close": pd.DataFrame({"AAPL": 200.0}, index=idx)}
    with caplog.at_level(logging.WARNING):
        rep = attach_fundamentals(data, ["AAPL"], settings=settings,
                                  price_fetch=lambda: (data["close"], {}))
    assert rep.stale == {"AAPL": "2025-01-31"}
    assert fx.logged(caplog, "已过期") and fx.logged(caplog, "2025-01-31")


# ---------------------------------------------------------------------------
# 7. 解析口径变了 → 重新解析
# ---------------------------------------------------------------------------

def test_facts_stored_under_an_old_parser_version_are_reparsed_even_if_fresh(tmp_path, monkeypatch):
    st = FundamentalsStore(tmp_path)
    sync(["AAPL"], fx.client(), st, now=100.0)
    assert st.parser_version("AAPL") == edgar.PARSER_VERSION

    again = fx.FakeSession()
    assert sync(["AAPL"], fx.client(again), st, now=101.0).cached == ["AAPL"]     # 版本相同 + 没过期 → 跳过
    assert again.calls == []

    from app.core.data_engine.fundamentals import service, store
    monkeypatch.setattr(service, "PARSER_VERSION", edgar.PARSER_VERSION + 1)
    monkeypatch.setattr(store, "PARSER_VERSION", edgar.PARSER_VERSION + 1)
    rep = sync(["AAPL"], fx.client(again), st, now=102.0)
    assert rep.synced == ["AAPL"] and again.calls != []
    assert st.parser_version("AAPL") == edgar.PARSER_VERSION + 1


def test_store_written_before_versioning_counts_as_outdated(tmp_path):
    import json
    st = FundamentalsStore(tmp_path)
    sync(["AAPL"], fx.client(), st, now=100.0)
    meta = tmp_path / "AAPL" / "meta.json"
    d = json.loads(meta.read_text(encoding="utf-8"))
    del d["parser_version"]
    meta.write_text(json.dumps(d), encoding="utf-8")
    assert st.parser_version("AAPL") is None
    assert sync(["AAPL"], fx.client(), st, now=101.0).synced == ["AAPL"]


# ---------------------------------------------------------------------------
# 受理时间戳：口径因申报人而异
# ---------------------------------------------------------------------------

def test_acceptance_timestamp_means_different_things_for_different_filers():
    """
    BKNG：按 UTC 读是美东 16:02，当日申报 —— 说得通。
    AAPL：按 UTC 读是美东 20:30，却仍记当日申报 —— 说不通（见 test_fundamentals_edgar）。
    同一个字段两种口径，所以事件日只认 filingDate。
    """
    ev = fx.events("BKNG")
    row = ev[ev["filing_date"] == "2026-02-18"].iloc[0]
    local = pd.Timestamp(row["accepted_raw"]).tz_convert("America/New_York")
    assert local.strftime("%Y-%m-%d %H:%M") == "2026-02-18 16:02"
    a = fx.events("AAPL")
    arow = a[a["filing_date"] == "2026-01-29"].iloc[0]
    alocal = pd.Timestamp(arow["accepted_raw"]).tz_convert("America/New_York")
    assert alocal.strftime("%Y-%m-%d %H:%M") == "2026-01-29 21:30"            # 晚于 17:30 截止
