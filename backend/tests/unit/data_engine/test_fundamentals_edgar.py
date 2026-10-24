"""
Phase 10.2 — EDGAR 客户端与事实解析。夹具是裁剪后的**真实响应**（见 fundamentals_fixtures）。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from app.core.data_engine.fundamentals import edgar
from app.core.data_engine.fundamentals.edgar import EdgarClient, EdgarError
from tests.unit.data_engine import fundamentals_fixtures as fx


# ---------------------------------------------------------------------------
# 身份声明：SEC 要求，且系统不替用户编
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ua", ["", "   ", None, "QuantAgent/1.0", "Kevin Hou"])
def test_client_refuses_to_exist_without_a_contact_email(ua):
    with pytest.raises(EdgarError, match="EDGAR_USER_AGENT"):
        EdgarClient(ua, session=fx.FakeSession())


def test_every_request_carries_the_declared_identity():
    sess = fx.FakeSession()
    fx.client(sess).company_facts("AAPL")
    assert len(sess.headers_seen) == 2                       # ticker 表 + 事实
    assert all(h["User-Agent"] == fx.UA for h in sess.headers_seen)


# ---------------------------------------------------------------------------
# ticker → CIK（真实的 company_tickers.json）
# ---------------------------------------------------------------------------

def test_cik_lookup_uses_the_real_ticker_table():
    c = fx.client()
    assert c.cik_for("AAPL") == 320193
    assert c.cik_for("nvda") == 1045810


def test_share_class_tickers_are_normalised_to_sec_style():
    """SEC 的表里是 BRK-B；数据集里可能写成 BRK.B。"""
    c = fx.client()
    assert c.cik_for("BRK.B") == c.cik_for("BRK-B") == 1067983


def test_unknown_ticker_is_an_error_not_an_empty_result():
    with pytest.raises(EdgarError, match="ZZZZ"):
        fx.client().cik_for("ZZZZ")


def test_ticker_table_is_fetched_once_per_client():
    sess = fx.FakeSession()
    c = fx.client(sess)
    c.cik_for("AAPL"), c.cik_for("NVDA"), c.cik_for("MSFT")
    assert sess.calls.count(edgar.TICKERS_URL) == 1


# ---------------------------------------------------------------------------
# HTTP 失败、缓存、限速
# ---------------------------------------------------------------------------

def test_company_without_xbrl_facts_is_a_404_error():
    """MSFT 在 ticker 表里，但替身没登记它的事实 URL → 404（EDGAR 对无事实的 CIK 就这么回）。"""
    with pytest.raises(EdgarError, match="404"):
        fx.client().company_facts("MSFT")


def test_network_failure_surfaces_as_edgar_error():
    sess = fx.FakeSession()
    sess.fail_with = ConnectionError("dns down")
    with pytest.raises(EdgarError, match="dns down"):
        fx.client(sess).company_facts("AAPL")


def test_fresh_cache_is_served_without_touching_the_network(tmp_path):
    first = fx.FakeSession()
    a = fx.client(first, cache_dir=tmp_path).submissions("AAPL")
    second = fx.FakeSession()
    b = fx.client(second, cache_dir=tmp_path).submissions("AAPL")
    assert second.calls == []
    assert a == b


def test_stale_cache_is_refetched(tmp_path):
    fx.client(fx.FakeSession(), cache_dir=tmp_path).submissions("AAPL")
    later = fx.FakeSession()
    now = [10 ** 10]                                          # 远晚于缓存文件的 mtime
    fx.client(later, cache_dir=tmp_path, max_age_h=20.0, clock=lambda: now[0]).submissions("AAPL")
    assert edgar.SUBMISSIONS_URL.format(cik=320193) in later.calls


def test_company_facts_are_never_written_to_the_raw_cache(tmp_path):
    """companyfacts 每家 5–8 MB：原文不落盘（要存的是解析后的事实），每次调用都是一次真请求。"""
    sess = fx.FakeSession()
    c = fx.client(sess, cache_dir=tmp_path)
    c.company_facts("AAPL"), c.company_facts("AAPL")
    assert sess.calls.count(edgar.FACTS_URL.format(cik=320193)) == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["company_tickers.json"]
    c.submissions("AAPL")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["company_tickers.json", "submissions_0000320193.json"]


def test_latest_report_date_looks_at_annual_and_quarterly_forms_only():
    assert edgar.latest_report_date(fx.payload("submissions", "AAPL")) == pd.Timestamp("2026-07-31")
    payload = {"filings": {"recent": {"form": ["8-K", "10-Q", "10-K/A", "4"],
                                      "filingDate": ["2024-09-01", "2024-05-02", "2024-06-15", "2024-12-01"]}}}
    assert edgar.latest_report_date(payload) == pd.Timestamp("2024-06-15")
    assert edgar.latest_report_date({"filings": {"recent": {"form": ["8-K"], "filingDate": ["2024-09-01"]}}}) is None
    assert edgar.latest_report_date(None) is None and edgar.latest_report_date({}) is None


def test_cache_boundary_exactly_max_age_is_still_fresh(tmp_path):
    fx.client(fx.FakeSession(), cache_dir=tmp_path).cik_for("AAPL")
    mtime = (tmp_path / "company_tickers.json").stat().st_mtime
    at_limit, past_limit = fx.FakeSession(), fx.FakeSession()
    fx.client(at_limit, cache_dir=tmp_path, max_age_h=1.0, clock=lambda: mtime + 3600.0).cik_for("AAPL")
    fx.client(past_limit, cache_dir=tmp_path, max_age_h=1.0, clock=lambda: mtime + 3600.5).cik_for("AAPL")
    assert at_limit.calls == []
    assert past_limit.calls == [edgar.TICKERS_URL]


def test_failed_response_is_not_cached(tmp_path):
    with pytest.raises(EdgarError):
        fx.client(fx.FakeSession(), cache_dir=tmp_path).company_facts("MSFT")
    assert [p.name for p in tmp_path.iterdir()] == ["company_tickers.json"]


def test_requests_are_spaced_to_respect_the_sec_rate_limit():
    """SEC 限 10 次/秒。连续两次请求之间必须睡够 MIN_INTERVAL_S。"""
    slept, t = [], [1000.0]

    def sleep(s):
        slept.append(s)
        t[0] += s

    c = EdgarClient(fx.UA, session=fx.FakeSession(), sleep=sleep, clock=lambda: t[0])
    c.cik_for("AAPL")            # 第一次：距"上次请求"(0) 很久，不用睡
    c.company_facts("AAPL")      # 紧接着第二次：时钟没走 → 睡满间隔
    assert slept == [pytest.approx(edgar.MIN_INTERVAL_S)]
    t[0] += 5.0
    c.submissions("AAPL")        # 隔了 5 秒：不用睡
    assert len(slept) == 1


# ---------------------------------------------------------------------------
# 事实解析（真实 companyfacts）
# ---------------------------------------------------------------------------

def test_parsed_facts_have_the_contract_columns_and_types():
    f = fx.facts("AAPL")
    assert list(f.columns) == edgar.FACT_COLUMNS
    assert len(f) > 500
    for c in ("end", "filed"):
        assert pd.api.types.is_datetime64_any_dtype(f[c]) and f[c].notna().all()
    assert set(f["concept"]) == set(edgar.CONCEPTS)


def test_only_annual_and_quarterly_reports_are_kept():
    """NVDA 的真实响应里有 5 条来自 DEF 14A（委托书）的净利润事实 —— 不收。"""
    for tk in ("AAPL", "NVDA"):
        assert set(fx.facts(tk)["form"]) <= edgar.FORMS
    raw = fx.payload("companyfacts", "NVDA")
    forms_in_raw = [x["form"] for tax in raw["facts"].values() for tag in tax.values()
                    for xs in tag["units"].values() for x in xs]
    assert forms_in_raw.count("DEF 14A") == 5


def test_instant_facts_have_no_start_and_flow_facts_do():
    f = fx.facts("AAPL")
    assert f[f["concept"].isin(["equity", "assets", "shares_cover", "shares_bs"])]["start"].isna().all()
    assert f[f["concept"].isin(["net_income", "revenue", "gross_profit"])]["start"].notna().all()


def test_revenue_is_assembled_across_the_tags_a_company_has_used():
    """NVDA 早年用 `Revenues`，后来换成 ASC 606 的科目 —— 两段都得接上。"""
    rev = fx.facts("NVDA").query("concept == 'revenue'")
    assert set(rev["tag"]) == {"Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax"}


def test_one_value_per_period_and_filing_when_tags_overlap():
    f = fx.facts("NVDA")
    assert not f.duplicated(subset=["concept", "start", "end", "filed"]).any()
    by_tag = f[f["concept"] == "revenue"].groupby("tag").size()
    assert (by_tag > 0).all() and len(by_tag) == 2


def test_tag_priority_decides_when_two_tags_report_the_same_period():
    payload = {"facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            {"start": "2023-01-01", "end": "2023-03-31", "val": 100, "form": "10-Q", "filed": "2023-05-01", "accn": "a"}]}},
        "SalesRevenueNet": {"units": {"USD": [
            {"start": "2023-01-01", "end": "2023-03-31", "val": 999, "form": "10-Q", "filed": "2023-05-01", "accn": "a"}]}},
    }}}
    f = edgar.parse_company_facts("X", payload)
    assert f[["tag", "val"]].values.tolist() == [["Revenues", 100.0]]


def test_restatements_are_kept_as_separate_rows_by_filing_date():
    """拆股后重述的股本：同一期末日，两个披露日，两个值 —— 两条都在，谁也不覆盖谁。"""
    f = fx.facts("NVDA")
    r = f[(f["concept"] == "shares_bs") & (f["end"] == "2024-01-28")].sort_values("filed")
    assert r["filed"].dt.strftime("%Y-%m-%d").tolist() == ["2024-02-21", "2025-02-26"]
    assert r["val"].tolist() == [2_464_000_000.0, 24_643_000_000.0]


@pytest.mark.parametrize("payload", [None, {}, {"facts": {}}, {"facts": {"us-gaap": {}}}])
def test_empty_payloads_give_an_empty_frame_with_the_contract_columns(payload):
    f = edgar.parse_company_facts("X", payload)
    assert f.empty and list(f.columns) == edgar.FACT_COLUMNS


def test_facts_without_value_or_dates_are_dropped():
    def fact(**kw):
        base = {"start": "2023-01-01", "end": "2023-03-31", "val": 1, "form": "10-Q",
                "filed": "2023-05-01", "accn": "a"}
        base.update(kw)
        return base
    payload = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
        fact(val=None), fact(end=None), fact(filed=None), fact(filed="not-a-date"),
        fact(form="8-K"), fact(val=7, accn="keep")]}}}}}
    f = edgar.parse_company_facts("X", payload)
    assert f["accn"].tolist() == ["keep"] and f["val"].tolist() == [7.0]


# ---------------------------------------------------------------------------
# 财报公告事件（真实 submissions）
# ---------------------------------------------------------------------------

def test_earnings_events_are_8k_item_202_only():
    ev = fx.events("AAPL")
    assert list(ev.columns) == edgar.EVENT_COLUMNS
    assert (ev["form"] == "8-K").all()
    assert ev["items"].str.contains("2.02").all()
    raw = fx.payload("submissions", "AAPL")["filings"]["recent"]
    n_8k = sum(1 for f in raw["form"] if f == "8-K")
    assert 0 < len(ev) < n_8k, "夹具里应当有不含 2.02 的 8-K，否则筛选没有被检验"


def test_earnings_events_are_about_quarterly_and_sorted():
    ev = fx.events("NVDA")
    d = ev["filing_date"]
    assert d.is_monotonic_increasing
    gaps = d.diff().dt.days.dropna()
    assert 84 <= gaps.median() <= 98
    # 真实数据里有一次相隔 16 天的 2.02（2022-08 的业绩预告 + 正式财报）：预告也是盈利信息，照收
    assert gaps.min() == 16 and (gaps < 60).sum() == 1


def test_item_match_is_exact_not_substring():
    """'12.02' 之类不能被当成 2.02。"""
    payload = {"filings": {"recent": {
        "form": ["8-K", "8-K", "10-Q"], "items": ["12.02,9.01", "5.02, 2.02", "2.02"],
        "filingDate": ["2024-01-02", "2024-01-03", "2024-01-04"],
        "acceptanceDateTime": ["x", "y", "z"], "accessionNumber": ["a", "b", "c"]}}}
    ev = edgar.parse_earnings_events("X", payload)
    assert ev["accn"].tolist() == ["b"]


def test_event_day_is_the_filing_date_and_the_timestamp_is_kept_verbatim():
    """
    受理时间戳按 UTC 读会与申报日自相矛盾（见 edgar.py）。这条测试把那个矛盾钉在真实数据上：
    哪天 SEC 把口径改对了，这里会先红。
    """
    ev = fx.events("AAPL")
    row = ev[ev["filing_date"] == "2024-08-01"].iloc[0]
    assert row["accepted_raw"] == "2024-08-02T00:30:26.000Z"
    as_utc = pd.Timestamp(row["accepted_raw"]).tz_convert("America/New_York")
    assert as_utc.strftime("%Y-%m-%d %H:%M") == "2024-08-01 20:30"       # 晚于 17:30 截止
    assert as_utc.hour * 60 + as_utc.minute > 17 * 60 + 30
    assert row["filing_date"] == pd.Timestamp("2024-08-01")              # 申报日却没顺延


@pytest.mark.parametrize("payload", [None, {}, {"filings": {}}, {"filings": {"recent": {"form": []}}}])
def test_empty_submissions_give_an_empty_event_frame(payload):
    ev = edgar.parse_earnings_events("X", payload)
    assert ev.empty and list(ev.columns) == edgar.EVENT_COLUMNS


def test_fixture_files_are_real_shaped_json():
    for tk in ("AAPL", "NVDA"):
        p = fx.payload("companyfacts", tk)
        assert {"cik", "entityName", "facts"} <= set(p)
        assert json.dumps(p)[:1] == "{"


# ---------------------------------------------------------------------------
# 真连 SEC 的契约测试 —— 夹具是某一天的快照，这里验"今天的 EDGAR 还是这个形状"。
# 需要环境变量 EDGAR_USER_AGENT（"Name email"）；没有就跳过，不会替你编身份。
# ---------------------------------------------------------------------------

_LIVE_UA = __import__("os").environ.get("EDGAR_USER_AGENT", "").strip()


@pytest.mark.live_source
@pytest.mark.skipif("@" not in _LIVE_UA, reason="未设置 EDGAR_USER_AGENT —— 不向 SEC 发请求")
def test_live_edgar_still_matches_the_fixture_contract(tmp_path):
    c = EdgarClient(_LIVE_UA, cache_dir=tmp_path)
    assert c.cik_for("AAPL") == fx.CIK["AAPL"]
    live = edgar.parse_company_facts("AAPL", c.company_facts("AAPL"))
    snap = fx.facts("AAPL")
    assert list(live.columns) == list(snap.columns)
    assert set(snap["concept"]) <= set(live["concept"])
    # 快照里的每一条事实，今天的 EDGAR 仍然原样给出（历史披露不会被改写）
    key = ["concept", "start", "end", "filed", "accn"]
    merged = snap.merge(live, on=key, how="left", suffixes=("", "_live"))
    assert merged["val_live"].notna().all()
    assert (merged["val"] == merged["val_live"]).all()
    ev = edgar.parse_earnings_events("AAPL", c.submissions("AAPL"))
    assert len(ev) >= 4 and ev["filing_date"].is_monotonic_increasing
    assert ev["accepted_raw"].str.endswith("Z").all()
