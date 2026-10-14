"""
FE-10 — GET /api/datasets/{name}/fundamentals：基本面字段清单、覆盖率、每只票的最近披露日。

接口只读本地已存的事实：不向 SEC 发请求、不加载价格。夹具是裁剪后的真实 EDGAR 响应。
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.alpha_engine.fields import FIELD_NOTES, FUNDAMENTAL_FIELDS, field_class
from app.core.data_engine.fundamentals.panels import MAX_STALE_DAYS
from app.core.data_engine.fundamentals.service import cached_metrics, store_status
from app.core.data_engine.fundamentals.store import FundamentalsStore
from tests.unit.data_engine import fundamentals_fixtures as fx


@pytest.fixture
def populated(tmp_path, monkeypatch):
    """AAPL / BKNG 有事实且算过指标，NVDA 有事实但没算过指标，MSFT 什么都没有。"""
    from app.config import settings
    from app.core.data_engine import dataset_registry as reg
    st = FundamentalsStore(tmp_path)
    for tk in ("AAPL", "BKNG", "NVDA"):
        st.save(tk, fx.facts(tk), fx.events(tk), now=1_780_000_000.0)
    for tk in ("AAPL", "BKNG"):
        cached_metrics(st, tk, st.load(tk)[0])
    monkeypatch.setattr(settings, "fundamentals_dir", str(tmp_path))
    monkeypatch.setattr(settings, "enable_fundamentals", True)
    monkeypatch.setattr(settings, "edgar_user_agent", "")
    monkeypatch.setitem(reg._SPECS, "fund_status_us",
                        reg.DatasetSpec("fund_status_us", "yfinance", "US", "mixed",
                                        ["AAPL", "BKNG", "NVDA", "MSFT"], "2021-01-01"))
    monkeypatch.setitem(reg._SPECS, "fund_status_crypto",
                        reg.DatasetSpec("fund_status_crypto", "ccxt_binance", "CRYPTO", "l1", ["BTC/USDT"], "2021-01-01"))
    return st


@pytest.fixture
def client(populated, monkeypatch):
    # 接口用真实日期判断"过期"：不钉住的话，夹具里 2026-08 的披露到 2027-02 就会过期，
    # 覆盖率断言随日历自己变红
    from app.core.data_engine.fundamentals import service
    monkeypatch.setattr(service, "_today", lambda: pd.Timestamp("2026-10-12"))
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _get(client, name):
    resp = client.get(f"/api/datasets/{name}/fundamentals")
    assert resp.status_code == 200, resp.text[:400]
    return resp.json()


# ---------------------------------------------------------------------------
# store_status
# ---------------------------------------------------------------------------

def test_status_reports_the_last_filing_date_per_ticker(populated):
    s = store_status(populated, ["AAPL", "BKNG", "NVDA", "MSFT"], today=pd.Timestamp("2026-10-12"))
    by = {r["ticker"]: r for r in s["tickers"]}
    assert by["AAPL"]["last_filed"] == "2026-07-31" and by["AAPL"]["last_period_end"] == "2026-06-27"
    assert by["BKNG"]["last_filed"] == "2026-08-04"
    assert by["NVDA"]["last_filed"] == "2026-08-26"
    assert by["MSFT"] == {"ticker": "MSFT", "has_facts": False, "first_filed": None, "last_filed": None,
                          "last_period_end": None, "stale": False, "fields": None, "synced_at": None}
    assert by["AAPL"]["first_filed"] == "2019-01-30" and by["BKNG"]["first_filed"] == "2024-02-22"
    assert s["n_tickers"] == 4 and s["n_with_facts"] == 3 and s["n_stale"] == 0
    assert by["AAPL"]["synced_at"].startswith("2026-05-28")           # 1_780_000_000 秒


def test_status_says_which_fields_the_latest_filing_supports(populated):
    s = store_status(populated, ["AAPL", "BKNG", "NVDA"], today=pd.Timestamp("2026-10-12"))
    by = {r["ticker"]: r for r in s["tickers"]}
    assert by["AAPL"]["fields"] == {f: True for f in FIELD_NOTES}
    # BKNG：净资产为负 → 没有 ROE / B/P；不报毛利也不报营业成本 → 没有毛利率
    assert by["BKNG"]["fields"] == {"market_cap": True, "earnings_yield": True, "book_to_price": False,
                                    "roe": False, "gross_margin": False, "asset_growth": True,
                                    "sue": True, "days_since_earnings": True}
    assert by["NVDA"]["fields"] is None                               # 有事实但还没算过指标 → 未知，不猜
    assert s["coverage"]["earnings_yield"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["coverage"]["roe"] == pytest.approx(1 / 3, abs=1e-4)


def test_stale_tickers_are_flagged_and_do_not_count_toward_coverage(populated):
    last = pd.Timestamp("2026-08-04")                                  # BKNG 的最近披露日
    fresh = store_status(populated, ["BKNG"], today=last + pd.Timedelta(days=MAX_STALE_DAYS))
    stale = store_status(populated, ["BKNG"], today=last + pd.Timedelta(days=MAX_STALE_DAYS + 1))
    assert fresh["n_stale"] == 0 and fresh["coverage"]["sue"] == 1.0
    assert stale["n_stale"] == 1 and stale["tickers"][0]["stale"] is True
    assert stale["coverage"]["sue"] == 0.0
    assert stale["tickers"][0]["fields"]["sue"] is True               # 字段本身算得出来，只是过期了


def test_status_makes_no_network_request_and_does_not_touch_the_store(populated, monkeypatch):
    import requests
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: pytest.fail("不该发请求"))
    before = sorted(p.name for p in populated.root.rglob("*") if p.is_file())
    store_status(populated, ["AAPL", "NVDA", "MSFT"], today=pd.Timestamp("2026-10-12"))
    assert sorted(p.name for p in populated.root.rglob("*") if p.is_file()) == before


def test_empty_store_reports_nothing_available(tmp_path):
    s = store_status(FundamentalsStore(tmp_path / "empty"), ["AAPL", "NVDA"], today=pd.Timestamp("2026-10-12"))
    assert s["n_with_facts"] == 0 and set(s["coverage"].values()) == {0.0}
    assert [r["has_facts"] for r in s["tickers"]] == [False, False]


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------

def test_endpoint_lists_every_field_with_its_class_and_coverage(client):
    body = _get(client, "fund_status_us")
    assert body["dataset"] == "fund_status_us" and body["supported"] is True and body["enabled"] is True
    assert [f["name"] for f in body["fields"]] == list(FIELD_NOTES)
    by = {f["name"]: f for f in body["fields"]}
    assert by["market_cap"]["class"] == "valuation" and by["roe"]["class"] == "quarterly"
    assert by["days_since_earnings"]["class"] == "event"
    assert by["earnings_yield"]["coverage"] == pytest.approx(0.5)      # AAPL、BKNG / 4 只
    assert by["roe"]["coverage"] == pytest.approx(0.25)
    assert all(f["note"] for f in body["fields"])
    assert body["n_tickers"] == 4 and body["n_with_facts"] == 3
    assert [t["ticker"] for t in body["tickers"]] == ["AAPL", "BKNG", "NVDA", "MSFT"]
    assert "第一个交易日" in body["visibility_rule"] and str(MAX_STALE_DAYS) in body["visibility_rule"]


def test_endpoint_reports_the_switch_and_identity_state(client, monkeypatch):
    from app.config import settings
    assert _get(client, "fund_status_us")["identity_configured"] is False
    monkeypatch.setattr(settings, "edgar_user_agent", "Someone someone@example.com")
    assert _get(client, "fund_status_us")["identity_configured"] is True
    monkeypatch.setattr(settings, "enable_fundamentals", False)
    off = _get(client, "fund_status_us")
    assert off["enabled"] is False and off["n_with_facts"] == 3        # 关着也如实报本地有什么


def test_endpoint_never_returns_the_identity_itself(client, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "edgar_user_agent", "Someone someone@example.com")
    payload = client.get("/api/datasets/fund_status_us/fundamentals").content.decode("utf-8")
    assert payload.count("someone@example.com") == 0 and payload.count("Someone") == 0
    assert payload.count("identity_configured") == 1


def test_non_us_dataset_is_reported_as_unsupported(client):
    body = _get(client, "fund_status_crypto")
    assert body["supported"] is False and body["enabled"] is False
    assert body["tickers"] == [] and body["n_with_facts"] == 0
    assert all(f["coverage"] is None for f in body["fields"])


def test_unknown_dataset_is_404(client):
    resp = client.get("/api/datasets/no_such_dataset/fundamentals")
    assert resp.status_code == 404


def test_ui_notes_cover_every_fundamental_field():
    assert set(FIELD_NOTES) == set(FUNDAMENTAL_FIELDS)
    assert {field_class(f) for f in FIELD_NOTES} == {"valuation", "quarterly", "event"}


def test_endpoint_judges_staleness_by_the_calendar(client, monkeypatch):
    """同一份本地数据，日子往后推过 200 天，覆盖率要掉下来、过期数要涨上去 —— 钉住"今天"的那个钩子真的被用了。"""
    from app.core.data_engine.fundamentals import service
    assert _get(client, "fund_status_us")["n_stale"] == 0
    monkeypatch.setattr(service, "_today", lambda: pd.Timestamp("2027-06-01"))
    later = _get(client, "fund_status_us")
    assert later["n_stale"] == 3
    assert {f["name"]: f["coverage"] for f in later["fields"]}["earnings_yield"] == 0.0
