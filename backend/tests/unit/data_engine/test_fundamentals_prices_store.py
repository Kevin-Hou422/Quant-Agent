"""
Phase 10.2 — 不复权价 / 拆股识别（两个价格源的真实口径）、事实存储、同步与挂载。
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.fundamentals import prices, service
from app.core.data_engine.fundamentals.panels import ALL_FIELDS, METRICS_VERSION, VALUATION_FIELDS, metric_events
from app.core.data_engine.fundamentals.service import (
    FundamentalsUnavailable, attach_fundamentals, cached_metrics, sync,
)
from app.core.data_engine.fundamentals.store import FundamentalsStore
from tests.unit.data_engine import fundamentals_fixtures as fx


# ---------------------------------------------------------------------------
# 拆股识别 / 不复权价 —— 两个源各自的真实口径
# ---------------------------------------------------------------------------

def test_the_price_fixtures_are_the_documented_synthetic_ones():
    """D4：仓库里的行情夹具必须是生成器逐位造出来的合成数据（不是哪天取回的真实行情）。"""
    import importlib.util
    spec = importlib.util.spec_from_file_location("synthetic_split_prices", fx.PRICE_DIR / "synthetic_split_prices.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    pd.testing.assert_frame_equal(fx.moomoo_prices(), gen.moomoo_frame(), check_freq=False, check_names=False)
    pd.testing.assert_frame_equal(fx.yahoo_download_frame(), gen.yahoo_frame(), check_freq=False,
                                  check_names=False, check_dtype=False)
    assert sorted(p.name for p in fx.PRICE_DIR.glob("*.csv")) == ["split_dividend_moomoo.csv", "split_dividend_yahoo.csv"]


def test_moomoo_split_is_read_from_the_raw_over_adjusted_ratio():
    df = fx.moomoo_prices()
    s = prices.detect_splits(df["raw"]["NVDA"], df["qfq"]["NVDA"])
    assert s.to_dict() == {pd.Timestamp("2024-06-10"): 10.0}


def test_dividends_and_ordinary_moves_are_not_splits():
    """AAPL 同期没有拆股；NVDA 06-11 除息（比值只动 0.008%）也不算。"""
    df = fx.moomoo_prices()
    assert prices.detect_splits(df["raw"]["AAPL"], df["qfq"]["AAPL"]).empty
    ratio = df["raw"]["NVDA"] / df["qfq"]["NVDA"]
    assert ratio["2024-06-10"] != ratio["2024-06-11"]            # 除息确实让比值动了
    assert len(prices.detect_splits(df["raw"]["NVDA"], df["qfq"]["NVDA"])) == 1


@pytest.mark.parametrize("mult,detected", [(1.5, True), (1.49, False), (1 / 1.5, True), (1 / 1.49, False)])
def test_split_threshold_boundaries(mult, detected):
    idx = pd.bdate_range("2024-01-02", periods=4)
    adj = pd.Series(100.0, index=idx)
    raw = pd.Series([100.0 * mult, 100.0 * mult, 100.0, 100.0], index=idx)
    s = prices.detect_splits(raw, adj)
    expected = {idx[2]: round(mult, 4)} if detected else {}
    assert s.to_dict() == pytest.approx(expected, abs=1e-4)


def test_reverse_split_has_a_multiplier_below_one():
    idx = pd.bdate_range("2024-01-02", periods=3)
    s = prices.detect_splits(pd.Series([2.0, 20.0, 20.0], index=idx), pd.Series(20.0, index=idx))
    assert s.to_dict() == {idx[1]: 0.1}


def test_yahoo_close_is_already_split_adjusted_and_gets_restored():
    """yfinance auto_adjust=False 的 Close 在拆股前是 120.0 而不是当时成交的 1200.0。"""
    frame = fx.yahoo_download_frame()
    assert frame["Close"]["NVDA"]["2024-06-07"] == pytest.approx(120.0, abs=1e-3)
    seen = {}

    def download(**kw):
        seen.update(kw)
        return frame.copy()

    raw, splits = prices.yahoo_raw_and_splits(["NVDA"], "2024-05-20", "2024-06-20", download=download)
    assert raw["NVDA"]["2024-06-07"] == pytest.approx(1200.0, abs=1e-3)
    assert raw["NVDA"]["2024-06-10"] == pytest.approx(121.0, abs=1e-3)      # 拆股日当天已是拆后价
    assert splits["NVDA"].to_dict() == {pd.Timestamp("2024-06-10"): 10.0}
    assert seen["auto_adjust"] is False and seen["actions"] is True
    assert seen["end"] == "2024-06-21"                                       # yfinance 的 end 是排他的


def test_both_sources_agree_on_the_unadjusted_price():
    """同一只票、同一段日期：moomoo 的不复权价与 yahoo 还原出来的应当是同一个数。"""
    mm = fx.moomoo_prices()["raw"]["NVDA"]
    yh, _ = prices.yahoo_raw_and_splits(["NVDA"], "2024-05-20", "2024-06-20",
                                        download=lambda **k: fx.yahoo_download_frame())
    pd.testing.assert_series_equal(yh["NVDA"].reindex(mm.index), mm, check_names=False,
                                   check_freq=False, atol=1e-3, rtol=0)


def test_unsplit_multiplies_by_later_splits_only():
    idx = pd.DatetimeIndex(["2024-01-31", "2024-02-01", "2024-03-15", "2024-04-01", "2024-04-02"])
    splits = pd.Series([2.0, 5.0], index=pd.DatetimeIndex(["2024-02-01", "2024-04-01"]))
    out = prices.unsplit(pd.Series(10.0, index=idx), splits)
    assert out.tolist() == [100.0, 50.0, 50.0, 10.0, 10.0]
    assert prices.unsplit(pd.Series(10.0, index=idx), splits.iloc[::-1]).tolist() == out.tolist()
    assert prices.unsplit(pd.Series(10.0, index=idx), splits.iloc[0:0]).tolist() == [10.0] * 5


def test_yahoo_without_split_column_is_an_error():
    frame = fx.yahoo_download_frame().drop(columns="Stock Splits", level=0)
    with pytest.raises(ValueError, match="Stock Splits"):
        prices.yahoo_raw_and_splits(["NVDA"], "2024-05-20", "2024-06-20", download=lambda **k: frame)
    with pytest.raises(ValueError, match="空"):
        prices.yahoo_raw_and_splits(["NVDA"], "2024-05-20", "2024-06-20", download=lambda **k: pd.DataFrame())


class _Provider:
    """MoomooProvider 的替身：按 autype 返回合成夹具里对应的那一列。"""
    calls: list = []

    def __init__(self, autype):
        self.autype = autype

    def fetch(self, tickers, start, end, fields=None):
        _Provider.calls.append((self.autype, tuple(tickers), start, end, tuple(fields or ())))
        df = fx.moomoo_prices()["raw" if self.autype is None else "qfq"]
        return {"close": df[[t for t in tickers if t in df.columns]]}


def test_moomoo_fetches_unadjusted_and_reuses_the_adjusted_close_it_is_given():
    _Provider.calls = []
    qfq = fx.moomoo_prices()["qfq"]
    raw, splits = prices.moomoo_raw_and_splits(["NVDA", "AAPL"], "2024-05-20", "2024-06-20", host="h", port=1,
                                               adj_close=qfq, provider_factory=_Provider)
    assert [c[0] for c in _Provider.calls] == [None]                 # 只取了一次：不复权
    assert _Provider.calls[0][4] == ("close",)
    assert raw["NVDA"]["2024-06-07"] == 1200.0
    assert list(splits) == ["NVDA"] and splits["NVDA"].iloc[0] == 10.0


def test_moomoo_fetches_its_own_adjusted_close_when_none_is_given():
    _Provider.calls = []
    _, splits = prices.moomoo_raw_and_splits(["NVDA"], "2024-05-20", "2024-06-20", host="h", port=1,
                                             provider_factory=_Provider)
    assert [c[0] for c in _Provider.calls] == [None, "qfq"]
    assert splits["NVDA"].to_dict() == {pd.Timestamp("2024-06-10"): 10.0}


def test_stitched_multi_vintage_close_hides_the_split():
    """
    为什么 PIT 增量面板不能拿来识别拆股：拆股前写入的那些 bar 是**当时的**前复权价（≈ 当时成交价），
    拆股后写入的是新口径。拼起来的序列与不复权价几乎处处相等，比值不跳。
    """
    df = fx.moomoo_prices()
    raw = df["raw"]["NVDA"]
    stitched = raw.copy()                       # 每根 bar 都是"写入当天"的复权价 = 当天成交价
    assert prices.detect_splits(raw, stitched).empty


# ---------------------------------------------------------------------------
# 存储
# ---------------------------------------------------------------------------

def test_store_roundtrip_preserves_facts_and_events(tmp_path):
    st = FundamentalsStore(tmp_path)
    assert st.load("AAPL") is None and st.synced_at("AAPL") is None
    f, e = fx.facts("AAPL"), fx.events("AAPL")
    st.save("AAPL", f, e, now=1234.0)
    f2, e2 = st.load("AAPL")
    pd.testing.assert_frame_equal(f2, f, check_dtype=False)
    pd.testing.assert_frame_equal(e2, e, check_dtype=False)
    assert st.synced_at("AAPL") == 1234.0
    assert pd.api.types.is_datetime64_any_dtype(f2["filed"])


def test_store_is_append_only_across_syncs(tmp_path):
    """重新同步时上游少了几条历史事实 —— 本地"当时看到过什么"不能跟着丢。"""
    st = FundamentalsStore(tmp_path)
    f, e = fx.facts("AAPL"), fx.events("AAPL")
    st.save("AAPL", f, e, now=1.0)
    newer = f[f["filed"] >= "2024-01-01"]
    st.save("AAPL", newer, e.iloc[-3:], now=2.0)
    f2, e2 = st.load("AAPL")
    assert len(f2) == len(f) and len(e2) == len(e)
    assert st.synced_at("AAPL") == 2.0


def test_store_merge_takes_the_newer_copy_of_the_same_fact(tmp_path):
    st = FundamentalsStore(tmp_path)
    f, e = fx.facts("AAPL"), fx.events("AAPL")
    st.save("AAPL", f, e)
    changed = f.iloc[[0]].copy()
    changed["val"] = -1.0
    st.save("AAPL", changed, e)
    f2, _ = st.load("AAPL")
    assert len(f2) == len(f)
    hit = f2[(f2["concept"] == changed["concept"].iloc[0]) & (f2["end"] == changed["end"].iloc[0])
             & (f2["filed"] == changed["filed"].iloc[0])]
    assert hit["val"].tolist() == [-1.0]


def test_store_normalises_ticker_directories(tmp_path):
    st = FundamentalsStore(tmp_path)
    st.save("brk.b", fx.facts("AAPL"), fx.events("AAPL"))
    assert (tmp_path / "BRK-B" / "facts.parquet").exists()
    assert st.load("BRK-B") is not None
    assert not list(tmp_path.rglob("*.tmp"))


def test_load_many_skips_missing_tickers(tmp_path):
    st = FundamentalsStore(tmp_path)
    st.save("AAPL", fx.facts("AAPL"), fx.events("AAPL"))
    facts, events = st.load_many(["AAPL", "NVDA"])
    assert list(facts) == ["AAPL"] and list(events) == ["AAPL"]


# ---------------------------------------------------------------------------
# 同步
# ---------------------------------------------------------------------------

def test_sync_fetches_parses_and_stores(tmp_path):
    sess = fx.FakeSession()
    rep = sync(["AAPL", "NVDA"], fx.client(sess), FundamentalsStore(tmp_path), now=100.0)
    assert rep.synced == ["AAPL", "NVDA"] and rep.cached == [] and rep.failed == {}
    f, e = FundamentalsStore(tmp_path).load("NVDA")
    assert len(f) == len(fx.facts("NVDA")) and len(e) == len(fx.events("NVDA"))
    assert rep.unchanged == []
    assert rep.to_dict() == {"synced": 2, "unchanged": 0, "cached": 0, "failed": {}}


def test_sync_skips_recently_synced_tickers(tmp_path):
    st = FundamentalsStore(tmp_path)
    sync(["AAPL"], fx.client(), st, now=0.0)
    again = fx.FakeSession()
    at_limit = sync(["AAPL"], fx.client(again), st, max_age_h=20.0, now=20 * 3600.0)
    assert at_limit.cached == ["AAPL"] and again.calls == []
    past = sync(["AAPL"], fx.client(again), st, max_age_h=20.0, now=20 * 3600.0 + 1)
    assert past.cached == [] and again.calls != []          # 过了有效期 → 至少去看一眼文件清单


def test_sync_without_a_new_report_does_not_redownload_company_facts(tmp_path):
    """财报一季才一份：每天只取很小的文件清单，没有新的年报 / 季报就不碰 5–8 MB 的 companyfacts。"""
    st = FundamentalsStore(tmp_path)
    sync(["AAPL", "NVDA"], fx.client(), st, now=0.0)
    day2 = fx.FakeSession()
    rep = sync(["AAPL", "NVDA"], fx.client(day2), st, now=86400.0)
    assert rep.unchanged == ["AAPL", "NVDA"] and rep.synced == [] and rep.failed == {}
    assert not [u for u in day2.calls if "companyfacts" in u]
    assert len([u for u in day2.calls if "submissions" in u]) == 2
    assert st.synced_at("AAPL") == 86400.0                   # 记为"今天看过了"
    assert len(st.load("AAPL")[0]) == len(fx.facts("AAPL"))


def test_sync_redownloads_when_a_newer_report_has_been_filed(tmp_path):
    """本地事实只到 2026-01-30，文件清单里已有 2026-05-01 和 2026-07-31 的 10-Q → 重下。"""
    st = FundamentalsStore(tmp_path)
    a = fx.facts("AAPL")
    st.save("AAPL", a[a["filed"] <= "2026-01-30"], fx.events("AAPL").iloc[:-2], now=0.0)
    sess = fx.FakeSession()
    rep = sync(["AAPL"], fx.client(sess), st, now=86400.0)
    assert rep.synced == ["AAPL"] and rep.unchanged == []
    assert [u for u in sess.calls if "companyfacts" in u]
    facts, events = st.load("AAPL")
    assert facts["filed"].max() == pd.Timestamp("2026-07-31") and len(events) == len(fx.events("AAPL"))


def test_sync_report_filed_on_the_same_day_as_the_latest_local_fact_counts_as_known(tmp_path):
    st = FundamentalsStore(tmp_path)
    st.save("AAPL", fx.facts("AAPL"), fx.events("AAPL"), now=0.0)      # 本地最新披露日 == 清单最新申报日
    sess = fx.FakeSession()
    assert sync(["AAPL"], fx.client(sess), st, now=86400.0).unchanged == ["AAPL"]
    a = fx.facts("AAPL")
    st2 = FundamentalsStore(tmp_path / "behind")
    st2.save("AAPL", a[a["filed"] < "2026-07-31"], fx.events("AAPL"), now=0.0)   # 差一天
    assert sync(["AAPL"], fx.client(fx.FakeSession()), st2, now=86400.0).synced == ["AAPL"]


def test_unchanged_path_still_picks_up_new_earnings_announcements(tmp_path):
    """8-K（财报新闻稿）先于 10-Q：没有新 10-Q 也要把新的公告事件记下来。"""
    st = FundamentalsStore(tmp_path)
    st.save("AAPL", fx.facts("AAPL"), fx.events("AAPL").iloc[:-1], now=0.0)
    rep = sync(["AAPL"], fx.client(), st, now=86400.0)
    assert rep.unchanged == ["AAPL"]
    assert len(st.load("AAPL")[1]) == len(fx.events("AAPL"))


def test_company_with_no_annual_or_quarterly_reports_listed_is_downloaded_not_skipped(tmp_path):
    from app.core.data_engine.fundamentals import edgar
    st = FundamentalsStore(tmp_path)
    st.save("AAPL", fx.facts("AAPL"), fx.events("AAPL"), now=0.0)
    sess = fx.FakeSession()
    sess.routes[edgar.SUBMISSIONS_URL.format(cik=fx.CIK["AAPL"])] = {
        "filings": {"recent": {"form": ["8-K"], "filingDate": ["2026-09-01"], "items": ["8.01"],
                               "acceptanceDateTime": ["x"], "accessionNumber": ["z"]}}}
    assert sync(["AAPL"], fx.client(sess), st, now=86400.0).synced == ["AAPL"]


def test_sync_one_failure_does_not_stop_the_rest(tmp_path, caplog):
    st = FundamentalsStore(tmp_path)
    with caplog.at_level(logging.WARNING):
        rep = sync(["ZZZZ", "MSFT", "AAPL"], fx.client(), st, now=1.0)
    assert rep.synced == ["AAPL"]
    assert set(rep.failed) == {"ZZZZ", "MSFT"}
    assert "404" in rep.failed["MSFT"] and "ZZZZ" in rep.failed["ZZZZ"]
    assert st.load("MSFT") is None
    assert fx.logged(caplog, "同步失败")


def test_sync_company_with_no_usable_facts_is_a_failure_not_an_empty_store(tmp_path):
    sess = fx.FakeSession()
    from app.core.data_engine.fundamentals import edgar
    sess.routes[edgar.FACTS_URL.format(cik=789019)] = {"cik": 789019, "facts": {"ifrs-full": {}}}
    sess.routes[edgar.SUBMISSIONS_URL.format(cik=789019)] = {"filings": {"recent": {"form": []}}}
    st = FundamentalsStore(tmp_path)
    rep = sync(["MSFT"], fx.client(sess), st, now=1.0)
    assert "MSFT" in rep.failed and st.load("MSFT") is None


def test_sync_keeps_the_old_copy_when_a_refresh_fails(tmp_path):
    st = FundamentalsStore(tmp_path)
    sync(["AAPL"], fx.client(), st, now=0.0)
    down = fx.FakeSession()
    down.fail_with = ConnectionError("down")
    rep = sync(["AAPL"], fx.client(down), st, now=10 ** 7)
    assert "AAPL" in rep.failed
    assert len(st.load("AAPL")[0]) == len(fx.facts("AAPL"))


# ---------------------------------------------------------------------------
# 每披露日指标的增量缓存
# ---------------------------------------------------------------------------

def test_cached_metrics_equal_a_full_recompute_after_new_filings(tmp_path):
    st = FundamentalsStore(tmp_path)
    f = fx.facts("NVDA")
    early = f[f["filed"] <= "2024-06-30"]
    first = cached_metrics(st, "NVDA", early)
    pd.testing.assert_frame_equal(first, metric_events(early))
    second = cached_metrics(st, "NVDA", f)                      # 多了几份新财报 → 只补算新披露日
    pd.testing.assert_frame_equal(second.reset_index(drop=True), metric_events(f), check_dtype=False)
    stamp = st.load_metrics("NVDA")[1]
    assert stamp["version"] == METRICS_VERSION and stamp["n_facts_upto"] == len(f)


def test_cached_metrics_are_reused_without_recomputation(tmp_path, monkeypatch):
    st = FundamentalsStore(tmp_path)
    f = fx.facts("AAPL")
    cached_metrics(st, "AAPL", f)
    from app.core.data_engine.fundamentals import panels
    monkeypatch.setattr(panels, "metric_events", lambda *a, **k: pytest.fail("不该重算"))
    out = cached_metrics(st, "AAPL", f)
    assert len(out) == 31


def test_cached_metrics_only_compute_the_new_filing_dates(tmp_path, monkeypatch):
    st = FundamentalsStore(tmp_path)
    f = fx.facts("AAPL")
    cached_metrics(st, "AAPL", f[f["filed"] <= "2026-01-30"])
    from app.core.data_engine.fundamentals import panels
    real, seen = panels.metric_events, []

    def spy(facts, since=None):
        seen.append(since)
        return real(facts, since=since)

    monkeypatch.setattr(panels, "metric_events", spy)
    out = cached_metrics(st, "AAPL", f)
    assert seen == [pd.Timestamp("2026-01-31")]
    assert out["filed"].dt.strftime("%Y-%m-%d").tolist()[-3:] == ["2026-01-30", "2026-05-01", "2026-07-31"]


@pytest.mark.parametrize("how", ["version", "backfilled_fact"])
def test_cached_metrics_are_invalidated(tmp_path, monkeypatch, how):
    st = FundamentalsStore(tmp_path)
    f = fx.facts("AAPL")
    cached_metrics(st, "AAPL", f)
    from app.core.data_engine.fundamentals import panels
    if how == "version":
        monkeypatch.setattr(panels, "METRICS_VERSION", METRICS_VERSION + 1)
    else:                                   # 补进一条旧披露日的事实 → 已算过的行可能变
        extra = f[(f["concept"] == "equity") & (f["filed"] == "2023-11-03")].iloc[[0]].copy()
        extra["val"], extra["accn"] = 1.0, "zz-late"
        f = pd.concat([f, extra], ignore_index=True)
    calls = []
    real = panels.metric_events
    monkeypatch.setattr(panels, "metric_events", lambda facts, since=None: calls.append(since) or real(facts, since=since))
    cached_metrics(st, "AAPL", f)
    assert calls == [None]                  # 全量重算


# ---------------------------------------------------------------------------
# 挂载
# ---------------------------------------------------------------------------

def _settings(tmp_path, **kw):
    base = dict(pit_store_dir=str(tmp_path / "pit"), fundamentals_dir="", edgar_user_agent="",
                fundamentals_max_age_h=20.0, price_source="moomoo", moomoo_host="h", moomoo_port=1)
    base.update(kw)
    return SimpleNamespace(**base)


def _price_panel():
    df = fx.moomoo_prices()
    return {"close": df["qfq"].copy()}, df["raw"].copy(), df["qfq"].copy()


def _raw_fetch():
    df = fx.moomoo_prices()
    raw = df["raw"]
    return raw, {"NVDA": prices.detect_splits(raw["NVDA"], df["qfq"]["NVDA"])}


def test_attach_adds_every_field_aligned_to_the_price_panel(tmp_path):
    data, _, _ = _price_panel()
    rep = attach_fundamentals(data, ["NVDA", "AAPL"], settings=_settings(tmp_path),
                              client=fx.client(), price_fetch=_raw_fetch)
    assert set(ALL_FIELDS) <= set(data)
    for name in ALL_FIELDS:
        assert data[name].index.equals(data["close"].index)
        assert list(data[name].columns) == list(data["close"].columns)
    assert rep.splits == {"NVDA": {"2024-06-10": 10.0}}
    assert data["earnings_yield"]["NVDA"]["2024-05-31"] == pytest.approx(42.598e9 / (1100.0 * 2.46e9), rel=1e-6)


def test_attach_default_store_lives_under_the_pit_directory(tmp_path):
    data, _, _ = _price_panel()
    attach_fundamentals(data, ["AAPL"], settings=_settings(tmp_path), client=fx.client(), price_fetch=_raw_fetch)
    assert (tmp_path / "pit" / "fundamentals" / "AAPL" / "facts.parquet").exists()
    data2, _, _ = _price_panel()
    attach_fundamentals(data2, ["AAPL"], settings=_settings(tmp_path, fundamentals_dir=str(tmp_path / "elsewhere")),
                        client=fx.client(), price_fetch=_raw_fetch)
    assert (tmp_path / "elsewhere" / "AAPL" / "facts.parquet").exists()


def test_enabled_without_identity_and_without_local_facts_is_an_error(tmp_path):
    data, _, _ = _price_panel()
    with pytest.raises(FundamentalsUnavailable, match="EDGAR_USER_AGENT"):
        attach_fundamentals(data, ["AAPL"], settings=_settings(tmp_path), price_fetch=_raw_fetch)
    assert set(data) == {"close"}                              # 没挂上任何东西


def test_without_identity_local_facts_are_still_used(tmp_path, caplog):
    FundamentalsStore(tmp_path / "pit" / "fundamentals").save("AAPL", fx.facts("AAPL"), fx.events("AAPL"))
    data, _, _ = _price_panel()
    with caplog.at_level(logging.WARNING):
        attach_fundamentals(data, ["AAPL", "NVDA"], settings=_settings(tmp_path), price_fetch=_raw_fetch)
    assert data["roe"]["AAPL"].notna().all() and data["roe"]["NVDA"].isna().all()
    assert fx.logged(caplog, "未配置 EDGAR_USER_AGENT")


def test_edgar_outage_with_nothing_local_is_an_error(tmp_path):
    down = fx.FakeSession()
    down.fail_with = ConnectionError("down")
    data, _, _ = _price_panel()
    with pytest.raises(FundamentalsUnavailable, match="EDGAR 全部请求失败"):
        attach_fundamentals(data, ["AAPL"], settings=_settings(tmp_path), client=fx.client(down),
                            price_fetch=_raw_fetch)


def test_unadjusted_price_failure_drops_valuation_fields_only(tmp_path, caplog):
    data, _, _ = _price_panel()

    def boom():
        raise RuntimeError("OpenD 未在监听")

    with caplog.at_level(logging.ERROR):
        attach_fundamentals(data, ["AAPL", "NVDA"], settings=_settings(tmp_path), client=fx.client(),
                            price_fetch=boom)
    assert not (set(VALUATION_FIELDS) & set(data))
    assert set(ALL_FIELDS) - set(VALUATION_FIELDS) <= set(data)
    assert fx.logged(caplog, "不提供")


def test_attach_requires_a_close_panel(tmp_path):
    with pytest.raises(FundamentalsUnavailable, match="close"):
        attach_fundamentals({}, ["AAPL"], settings=_settings(tmp_path), client=fx.client())
    with pytest.raises(FundamentalsUnavailable, match="close"):
        attach_fundamentals({"close": pd.DataFrame()}, ["AAPL"], settings=_settings(tmp_path), client=fx.client())


def test_identity_from_settings_builds_a_client(tmp_path, monkeypatch):
    built = {}

    class _Client:
        def __init__(self, ua, cache_dir=None, max_age_h=None):
            built.update(ua=ua, cache_dir=cache_dir, max_age_h=max_age_h)
            self._inner = fx.client()

        def company_facts(self, tk):
            return self._inner.company_facts(tk)

        def submissions(self, tk):
            return self._inner.submissions(tk)

        def predecessor_facts(self, tk):
            return self._inner.predecessor_facts(tk)

    monkeypatch.setattr(service, "EdgarClient", _Client)
    data, _, _ = _price_panel()
    attach_fundamentals(data, ["AAPL"], settings=_settings(tmp_path, edgar_user_agent="  A b@c.io ",
                                                           fundamentals_max_age_h=5.0),
                        price_fetch=_raw_fetch)
    assert built["ua"] == "A b@c.io" and built["max_age_h"] == 5.0
    assert built["cache_dir"] == tmp_path / "pit" / "fundamentals" / "_raw"
    assert data["roe"]["AAPL"].notna().all()


# ambiguous_out：疑似小比例送股的回填口（复审 2026-10-19 F13），fetch 没给就是 None
_MOOMOO_KW = {"host": "h", "port": 1, "adj_close": "ADJ", "ambiguous_out": None}


@pytest.mark.parametrize("source,expected,kwargs", [("moomoo", "moomoo", _MOOMOO_KW), ("yahoo", "yahoo", {}),
                                                    ("", "yahoo", {})])
def test_price_source_setting_picks_the_matching_unadjusted_fetcher(tmp_path, monkeypatch, source, expected,
                                                                    kwargs):
    hit = []
    monkeypatch.setattr(prices, "moomoo_raw_and_splits",
                        lambda tickers, start, end, **kw: hit.append(("moomoo", kw)) or ("r", "s"))
    monkeypatch.setattr(prices, "yahoo_raw_and_splits",
                        lambda tickers, start, end, **kw: hit.append(("yahoo", kw)) or ("r", "s"))
    out = service.fetch_raw_and_splits(["AAPL"], "2024-01-01", "2024-02-01",
                                       _settings(tmp_path, price_source=source), adj_close="ADJ")
    assert out == ("r", "s")
    assert hit == [(expected, kwargs)]


# ---------------------------------------------------------------------------
# 不复权价缓存：取过的区间不再向数据源要
# ---------------------------------------------------------------------------

from app.core.data_engine.fundamentals.service import cached_raw_and_splits   # noqa: E402


class _Source:
    """记录每次被要了哪些票、哪个区间的数据源替身；数据取自 moomoo 形状的合成夹具。"""

    def __init__(self, missing=()):
        self.calls = []
        self.missing = set(missing)

    def __call__(self, tickers, start, end, adj_close):
        self.calls.append((tuple(tickers), start, end))
        df = fx.moomoo_prices()
        raw = df["raw"].loc[start:end, [t for t in tickers if t not in self.missing]]
        qfq = df["qfq"].loc[start:end]
        return raw, {t: prices.detect_splits(raw[t], qfq[t]) for t in raw.columns}


def _idx(start, end):
    return fx.moomoo_prices().loc[start:end].index


def test_raw_prices_are_fetched_once_then_served_from_disk(tmp_path):
    st, src = FundamentalsStore(tmp_path), _Source()
    idx = _idx("2024-05-20", "2024-06-20")
    raw1, sp1 = cached_raw_and_splits(st, ["NVDA", "AAPL"], idx, _settings(tmp_path), fetch=src)
    raw2, sp2 = cached_raw_and_splits(st, ["NVDA", "AAPL"], idx, _settings(tmp_path), fetch=src)
    assert src.calls == [(("NVDA", "AAPL"), "2024-05-20", "2024-06-20")]      # 第二次没有请求
    pd.testing.assert_frame_equal(raw1, raw2)
    assert raw2["NVDA"]["2024-06-07"] == 1200.0
    assert list(sp2) == ["NVDA"] and sp2["NVDA"].to_dict() == {pd.Timestamp("2024-06-10"): 10.0}
    assert raw2.index.equals(idx) and list(raw2.columns) == ["NVDA", "AAPL"]


def test_a_narrower_window_inside_the_cached_range_needs_no_request(tmp_path):
    st, src = FundamentalsStore(tmp_path), _Source()
    cached_raw_and_splits(st, ["NVDA"], _idx("2024-05-20", "2024-06-20"), _settings(tmp_path), fetch=src)
    inner = _idx("2024-05-28", "2024-06-12")
    raw, _ = cached_raw_and_splits(st, ["NVDA"], inner, _settings(tmp_path), fetch=src)
    assert len(src.calls) == 1 and raw.index.equals(inner)


def test_only_the_missing_tail_is_fetched_with_one_day_of_overlap(tmp_path):
    """
    先取到 06-07（拆股前一天），再要到 06-20：只取 06-07 起的尾巴。重叠的那一天不能省 ——
    06-10 的拆股是靠 06-07 → 06-10 的比值跳变认出来的。
    """
    st, src = FundamentalsStore(tmp_path), _Source()
    cached_raw_and_splits(st, ["NVDA"], _idx("2024-05-20", "2024-06-07"), _settings(tmp_path), fetch=src)
    raw, splits = cached_raw_and_splits(st, ["NVDA"], _idx("2024-05-20", "2024-06-20"), _settings(tmp_path), fetch=src)
    assert src.calls[1] == (("NVDA",), "2024-06-07", "2024-06-20")
    assert splits["NVDA"].to_dict() == {pd.Timestamp("2024-06-10"): 10.0}
    assert raw["NVDA"].notna().all() and raw["NVDA"]["2024-05-20"] == 950.25


def test_an_earlier_start_or_a_new_ticker_triggers_a_full_fetch_for_those_names_only(tmp_path):
    st, src = FundamentalsStore(tmp_path), _Source()
    cached_raw_and_splits(st, ["NVDA"], _idx("2024-06-03", "2024-06-20"), _settings(tmp_path), fetch=src)
    cached_raw_and_splits(st, ["NVDA", "AAPL"], _idx("2024-06-03", "2024-06-20"), _settings(tmp_path), fetch=src)
    assert src.calls[1] == (("AAPL",), "2024-06-03", "2024-06-20")           # NVDA 已有，不再要
    cached_raw_and_splits(st, ["NVDA", "AAPL"], _idx("2024-05-20", "2024-06-20"), _settings(tmp_path), fetch=src)
    assert src.calls[2] == (("NVDA", "AAPL"), "2024-05-20", "2024-06-20")    # 起点更早 → 整段重取
    _, splits = cached_raw_and_splits(st, ["NVDA"], _idx("2024-05-20", "2024-06-20"), _settings(tmp_path), fetch=src)
    assert len(src.calls) == 3
    assert splits["NVDA"].to_dict() == {pd.Timestamp("2024-06-10"): 10.0}    # 重取不会把拆股记成两次


def test_ticker_the_source_did_not_return_is_retried_next_time(tmp_path):
    st = FundamentalsStore(tmp_path)
    idx = _idx("2024-05-20", "2024-06-20")
    flaky = _Source(missing={"AAPL"})
    raw, _ = cached_raw_and_splits(st, ["NVDA", "AAPL"], idx, _settings(tmp_path), fetch=flaky)
    assert list(raw.columns) == ["NVDA"]
    ok = _Source()
    raw, _ = cached_raw_and_splits(st, ["NVDA", "AAPL"], idx, _settings(tmp_path), fetch=ok)
    assert ok.calls == [(("AAPL",), "2024-05-20", "2024-06-20")]
    assert list(raw.columns) == ["NVDA", "AAPL"] and raw["AAPL"].notna().all()


def test_last_bar_not_yet_published_is_fetched_again_later(tmp_path):
    """末日那根还没出来（盘中 / 数据源延迟）：不能把那天记成取过了。"""
    st = FundamentalsStore(tmp_path)
    idx = _idx("2024-05-20", "2024-06-20")

    def early(tickers, start, end, adj_close):                              # 少返回最后一天
        df = fx.moomoo_prices()
        return df["raw"].loc[start:"2024-06-18", list(tickers)], {}

    raw, _ = cached_raw_and_splits(st, ["AAPL"], idx, _settings(tmp_path), fetch=early)
    assert np.isnan(raw["AAPL"]["2024-06-20"])
    src = _Source()
    raw, _ = cached_raw_and_splits(st, ["AAPL"], idx, _settings(tmp_path), fetch=src)
    assert src.calls == [(("AAPL",), "2024-06-18", "2024-06-20")]
    assert raw["AAPL"]["2024-06-20"] == 209.70


def test_caches_are_kept_apart_by_price_source(tmp_path):
    st, src = FundamentalsStore(tmp_path), _Source()
    idx = _idx("2024-05-20", "2024-06-20")
    cached_raw_and_splits(st, ["NVDA"], idx, _settings(tmp_path, price_source="moomoo"), fetch=src)
    cached_raw_and_splits(st, ["NVDA"], idx, _settings(tmp_path, price_source="yahoo"), fetch=src)
    assert len(src.calls) == 2
    assert (tmp_path / "_prices" / "moomoo" / "raw_close.parquet").exists()
    assert (tmp_path / "_prices" / "yahoo" / "raw_close.parquet").exists()


def test_source_returning_nothing_at_all_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="没有返回"):
        cached_raw_and_splits(FundamentalsStore(tmp_path), ["NVDA"], _idx("2024-05-20", "2024-06-20"),
                              _settings(tmp_path), fetch=_Source(missing={"NVDA"}))


def test_cached_values_are_not_overwritten_by_a_later_fetch(tmp_path):
    """不复权价是历史事实：已落盘的值以先到的为准，后来的抓取只补缺。"""
    st = FundamentalsStore(tmp_path)
    idx = _idx("2024-05-20", "2024-06-20")
    cached_raw_and_splits(st, ["NVDA"], idx, _settings(tmp_path), fetch=_Source())
    tampered = fx.moomoo_prices()["raw"][["NVDA"]] * 0 + 1.0
    st.save_raw_prices("moomoo", tampered, {}, {})
    raw, _ = cached_raw_and_splits(st, ["NVDA"], idx, _settings(tmp_path), fetch=_Source())
    assert raw["NVDA"]["2024-06-07"] == 1200.0


def test_attach_uses_the_price_cache_on_the_default_path(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(tickers, start, end, settings, adj_close=None, ambiguous_out=None):
        calls.append((tuple(tickers), start, end))
        df = fx.moomoo_prices()
        return df["raw"][list(tickers)], {"NVDA": prices.detect_splits(df["raw"]["NVDA"], df["qfq"]["NVDA"])}

    monkeypatch.setattr(service, "fetch_raw_and_splits", fake_fetch)
    for _ in range(2):
        data, _, _ = _price_panel()
        attach_fundamentals(data, ["NVDA", "AAPL"], settings=_settings(tmp_path), client=fx.client())
        assert data["market_cap"]["NVDA"]["2024-06-10"] == pytest.approx(121.0 * 2.46e9 * 10, rel=1e-9)
    assert calls == [(("NVDA", "AAPL"), "2024-05-20", "2024-06-20")]
