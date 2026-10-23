"""
外部审查 2026-10-18 F01：健康门把缺失数据删掉后再评分；成交量、高低价从没送进检查器。

修复后的契约：
  - 按 日期 × 股票池 的完整网格检查（不依赖 pandas 版本的 stack 行为），OHLCV 都送进去；
  - 综合分之外有硬性条件：末日覆盖率、整天缺失的交易日、末日零成交量、末日 OHLC 合法性 ——
    任何一条不满足就不健康，综合分多高都没用；
  - 每日摄取遇到硬性条件不满足 → 拒绝；检查器没给出结果（返回 None）→ 也拒绝（此前按满分放行）。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine import dataset_registry as reg
from app.core.data_engine.dataset_registry import _health_long_panel, check_dataset_health
from app.core.data_engine.health_report import (
    MAX_LAST_DAY_ZERO_VOLUME, MIN_LAST_DAY_COVERAGE, hard_failures,
)

IDX = pd.bdate_range("2025-01-06", periods=60)
COLS = [f"T{i}" for i in range(20)]


def _wide(seed=0):
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (len(IDX), len(COLS))), 0),
                         index=IDX, columns=COLS)
    opn = close.shift(1).fillna(close)
    amp = np.abs(rng.normal(0, 0.005, close.shape))
    return {"open": opn, "high": np.maximum(opn, close) * (1 + amp),
            "low": np.minimum(opn, close) * (1 - amp), "close": close,
            "volume": pd.DataFrame(rng.uniform(1e6, 5e6, close.shape), index=IDX, columns=COLS)}


def _ds(data, name="t", universe=None):
    return SimpleNamespace(name=name, data=data, universe=universe or list(data["close"].columns))


# ---------------------------------------------------------------------------
# 审查的 R10 场景
# ---------------------------------------------------------------------------

def test_the_review_scenario_is_now_unhealthy():
    """100 日、2 只票，只有前 5 日有价，已观测成交量全为零。此前（pandas 2.x）拿 1.0 分。"""
    close = pd.DataFrame(np.nan, index=pd.bdate_range("2025-01-01", periods=100), columns=["AAA", "BBB"])
    close.iloc[:5] = 100.0
    rep = check_dataset_health(_ds({"close": close, "volume": close * 0}), min_score=0.7, warn_only=True)
    assert rep.hard_failures and not rep.is_healthy(0.0)
    assert any("只有 0% 的标的有收盘价" in f for f in rep.hard_failures)
    assert rep.n_dates == 100                                   # 缺失的日期没有从检查里消失
    assert len(rep.zero_volume) == 10                           # 有报价的 5 天 × 2 只，成交量全零
    with pytest.raises(reg.DatasetHealthError, match="硬性条件"):
        check_dataset_health(_ds({"close": close, "volume": close * 0}), min_score=0.7, warn_only=False)


def test_long_panel_keeps_every_missing_observation_regardless_of_pandas_version(monkeypatch):
    """不用 DataFrame.stack：即使 stack 丢 NaN（pandas 2.x 的默认），网格也是完整的。"""
    w = _wide()
    w["close"].iloc[10:20, :5] = np.nan
    monkeypatch.setattr(pd.DataFrame, "stack", lambda *a, **k: pytest.fail("不该依赖 stack"))
    long_df, universe = _health_long_panel(_ds(w))
    assert len(long_df) == len(IDX) * len(COLS)
    assert int(long_df["close"].isna().sum()) == 50
    assert set(long_df.columns) == {"timestamp", "ticker", "close", "open", "high", "low", "volume"}


def test_a_ticker_in_the_universe_with_no_column_at_all_counts_as_missing():
    w = _wide()
    rep = check_dataset_health(_ds(w, universe=COLS + ["GONE1", "GONE2", "GONE3"]), min_score=0.0)
    # 23 只里 20 只有价 = 87% < 90%
    assert any("87%" in f for f in rep.hard_failures)
    assert rep.n_tickers == 23


# ---------------------------------------------------------------------------
# 各条硬性条件与边界
# ---------------------------------------------------------------------------

def test_clean_data_has_no_hard_failures():
    assert hard_failures(_wide(), COLS) == []
    rep = check_dataset_health(_ds(_wide()), min_score=0.7, warn_only=False)
    assert rep.hard_failures == [] and rep.is_healthy(0.7)


def test_last_day_coverage_boundary():
    assert MIN_LAST_DAY_COVERAGE == 0.9
    w = _wide()
    w["close"].iloc[-1, :2] = np.nan                            # 18/20 = 90% → 恰好够
    assert hard_failures(w, COLS) == []
    w["close"].iloc[-1, :3] = np.nan                            # 17/20 = 85%
    (f,) = hard_failures(w, COLS)
    assert "85%" in f and str(IDX[-1].date()) in f


def test_missing_values_before_the_last_day_do_not_trip_the_coverage_rule():
    """历史中段的缺失（新上市、停牌）留给综合分；硬性条件只管今天能不能交易。"""
    w = _wide()
    w["close"].iloc[:30, :10] = np.nan
    assert hard_failures(w, COLS) == []


def test_whole_missing_session_is_caught_against_the_exchange_calendar():
    w = _wide()
    dropped = IDX[25]
    w = {k: v.drop(index=dropped) for k, v in w.items()}
    (f,) = hard_failures(w, COLS, expected_sessions=IDX)
    assert "1 个交易日整天没有任何报价" in f and str(dropped.date()) in f
    assert hard_failures(w, COLS) == []                         # 不给日历就看不出来 —— 这就是要日历的原因


def test_a_session_present_in_the_index_but_empty_counts_as_missing():
    w = _wide()
    w["close"].iloc[25] = np.nan
    (f,) = hard_failures(w, COLS, expected_sessions=IDX)
    assert str(IDX[25].date()) in f


def test_us_datasets_get_the_exchange_calendar_others_do_not(monkeypatch):
    seen = []
    monkeypatch.setattr("app.core.data_engine.market_calendar.trading_days",
                        lambda s, e, exchange="NYSE": seen.append((s, e)) or pd.DatetimeIndex(IDX))
    monkeypatch.setitem(reg._SPECS, "hc_us", reg.DatasetSpec("hc_us", "yfinance", "US", "x", COLS))
    monkeypatch.setitem(reg._SPECS, "hc_crypto", reg.DatasetSpec("hc_crypto", "ccxt_binance", "CRYPTO", "x", COLS))
    check_dataset_health(_ds(_wide(), name="hc_us"), min_score=0.0)
    check_dataset_health(_ds(_wide(), name="hc_crypto"), min_score=0.0)
    assert seen == [(IDX[0], IDX[-1])]


def test_calendar_failure_skips_only_that_rule(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("calendar down")
    monkeypatch.setattr("app.core.data_engine.market_calendar.trading_days", boom)
    monkeypatch.setitem(reg._SPECS, "hc_us", reg.DatasetSpec("hc_us", "yfinance", "US", "x", COLS))
    w = _wide()
    w["volume"].iloc[-1] = 0.0
    rep = check_dataset_health(_ds(w, name="hc_us"), min_score=0.0)
    assert any("成交量为零" in f for f in rep.hard_failures)


def test_last_day_zero_volume_boundary():
    assert MAX_LAST_DAY_ZERO_VOLUME == 0.1
    w = _wide()
    w["volume"].iloc[-1, :2] = 0.0                              # 2/20 = 10% → 恰好允许
    assert hard_failures(w, COLS) == []
    w["volume"].iloc[-1, 2] = np.nan                            # 3/20
    (f,) = hard_failures(w, COLS)
    assert "20 只里 3 只成交量为零或缺失" in f


def test_zero_volume_where_there_is_no_price_is_not_double_counted():
    w = _wide()
    w["close"].iloc[-1, :2] = np.nan
    w["volume"].iloc[-1, :2] = 0.0
    assert hard_failures(w, COLS) == []
    rep = check_dataset_health(_ds(w), min_score=0.0)
    assert len(rep.zero_volume) == 0


def test_missing_volume_field_is_a_hard_failure():
    w = _wide()
    del w["volume"]
    assert hard_failures(w, COLS) == ["没有成交量数据"]


@pytest.mark.parametrize("broken", ["high_below_close", "low_above_open", "nonpositive"])
def test_invalid_ohlc_on_the_last_day_is_caught(broken):
    w = _wide()
    c, o = w["close"].iloc[-1, 0], w["open"].iloc[-1, 0]
    if broken == "high_below_close":
        w["high"].iloc[-1, 0] = min(c, o) * 0.99
    elif broken == "low_above_open":
        w["low"].iloc[-1, 0] = max(c, o) * 1.01
    else:
        w["low"].iloc[-1, 0] = 0.0
    (f,) = hard_failures(w, COLS)
    assert "OHLC 不合法" in f and "T0" in f


def test_high_exactly_equal_to_close_is_valid():
    w = _wide()
    top = np.maximum(w["open"].iloc[-1], w["close"].iloc[-1])
    w["high"].iloc[-1] = top
    w["low"].iloc[-1] = np.minimum(w["open"].iloc[-1], w["close"].iloc[-1])
    assert hard_failures(w, COLS) == []


def test_no_close_at_all_is_a_hard_failure():
    assert hard_failures({}, COLS) == ["没有收盘价数据"]


# ---------------------------------------------------------------------------
# 每日摄取
# ---------------------------------------------------------------------------

def _ingest_with(monkeypatch, report):
    from app.tasks.daily_ingest import DailyIngest
    monkeypatch.setattr(reg, "load_registry_dataset",
                        lambda name, start=None, end=None, health_check=False, with_fundamentals=None:
                        SimpleNamespace(data=_wide()))
    monkeypatch.setattr(reg, "check_dataset_health", lambda ds, min_score=0.7, warn_only=True: report)
    return DailyIngest().ingest("x", "2025-01-06", "2025-03-28", append_pit=False)


def test_ingest_rejects_hard_failures_even_with_a_perfect_score(monkeypatch):
    rep = SimpleNamespace(overall_score=1.0, hard_failures=["最后一个交易日只有 40% 的标的有收盘价"])
    res = _ingest_with(monkeypatch, rep)
    assert res.accepted is False and res.reject_reason.startswith("health_hard_failure")
    assert "40%" in res.reject_reason and res.health_score == 1.0


def test_ingest_rejects_when_the_checker_returns_nothing(monkeypatch):
    """此前：返回 None → score 保持 1.0 → 放行。检查器坏了，门是开着的。"""
    res = _ingest_with(monkeypatch, None)
    assert res.accepted is False and res.reject_reason == "health_unavailable"


def test_ingest_still_accepts_healthy_data(monkeypatch):
    res = _ingest_with(monkeypatch, SimpleNamespace(overall_score=0.95, hard_failures=[]))
    assert res.accepted is True


def test_health_api_reports_hard_failures(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    close = pd.DataFrame(np.nan, index=pd.bdate_range("2025-01-01", periods=30), columns=["AAA", "BBB"])
    close.iloc[:5] = 100.0
    monkeypatch.setattr(reg, "load_registry_dataset",
                        lambda name, start=None, end=None: _ds({"close": close, "volume": close * 0 + 1e6}))
    with TestClient(app, client=("127.0.0.1", 50000)) as c:
        body = c.get("/api/datasets/whatever/health").json()
    assert body["healthy"] is False and any("0%" in f for f in body["hard_failures"])
