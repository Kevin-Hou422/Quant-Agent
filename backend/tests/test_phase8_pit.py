"""
test_phase8_pit.py — Phase 8.1 Point-in-Time 数据存储验收

覆盖验收标准（roadmap §五 Task 8.1）：
  - 修改今日数据不影响昨日 as_of 查询结果（不可变历史 / 双时点语义）
  - load_pit(as_of=None) 返回最新 vintage；load_pit(as_of=T) 返回 T 时点可见视角
  - 追加幂等：同一 (timestamp,ticker,as_of) 重跑不产生重复
  - 与 daily_ingest 集成：通过验收的数据落入 PIT
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine.pit_store import PITStore


def _wide(dates, tickers, values) -> pd.DataFrame:
    return pd.DataFrame(values, index=pd.DatetimeIndex(dates), columns=tickers)


def _panel(dates, tickers, close_vals) -> dict:
    return {"close": _wide(dates, tickers, close_vals)}


@pytest.fixture()
def store(tmp_path):
    return PITStore(tmp_path / "pit")


def test_append_and_latest_query(store):
    dates = ["2026-01-05", "2026-01-06"]
    tickers = ["AAA", "BBB"]
    store.append(_panel(dates, tickers, [[10.0, 20.0], [11.0, 21.0]]),
                 as_of="2026-01-06T21:00:00", name="ds")

    out = store.load_pit(name="ds")
    assert "close" in out
    close = out["close"]
    assert list(close.columns) == tickers
    assert close.loc[pd.Timestamp("2026-01-05"), "AAA"] == 10.0
    assert close.loc[pd.Timestamp("2026-01-06"), "BBB"] == 21.0


def test_revision_does_not_change_earlier_as_of(store):
    """核心验收：对某交易日以更晚 as_of 修订，不改变早先 as_of 的查询结果。"""
    d = ["2026-02-02"]
    t = ["AAA"]
    # vintage 1：as_of=2/2，close=100
    store.append(_panel(d, t, [[100.0]]), as_of="2026-02-02T21:00:00", name="ds")
    # vintage 2：as_of=2/3，同一交易日修订为 105（如数据源事后回填/修正）
    store.append(_panel(d, t, [[105.0]]), as_of="2026-02-03T21:00:00", name="ds")

    # 昨日视角（as_of=2/2）仍看到 100，不受今日修订影响
    at_v1 = store.load_pit(as_of="2026-02-02T23:59:59", name="ds")["close"]
    assert at_v1.loc[pd.Timestamp("2026-02-02"), "AAA"] == 100.0

    # 最新视角（as_of=None）看到修订后的 105
    latest = store.load_pit(as_of=None, name="ds")["close"]
    assert latest.loc[pd.Timestamp("2026-02-02"), "AAA"] == 105.0

    # 显式 as_of=2/3 也看到 105
    at_v2 = store.load_pit(as_of="2026-02-03T21:00:00", name="ds")["close"]
    assert at_v2.loc[pd.Timestamp("2026-02-02"), "AAA"] == 105.0


def test_as_of_before_first_vintage_returns_empty(store):
    store.append(_panel(["2026-03-02"], ["AAA"], [[1.0]]),
                 as_of="2026-03-02T21:00:00", name="ds")
    out = store.load_pit(as_of="2026-03-01T00:00:00", name="ds")
    assert out == {}


def test_append_idempotent_same_vintage(store):
    d, t = ["2026-04-01"], ["AAA"]
    store.append(_panel(d, t, [[50.0]]), as_of="2026-04-01T21:00:00", name="ds")
    store.append(_panel(d, t, [[50.0]]), as_of="2026-04-01T21:00:00", name="ds")  # 重跑
    long_df = store._load_long("ds")
    # 同一 (timestamp,ticker,as_of) 只应有一行
    assert len(long_df) == 1


def test_available_as_of(store):
    store.append(_panel(["2026-05-01"], ["AAA"], [[1.0]]),
                 as_of="2026-05-01T21:00:00", name="ds")
    store.append(_panel(["2026-05-01"], ["AAA"], [[2.0]]),
                 as_of="2026-05-02T21:00:00", name="ds")
    vintages = store.available_as_of("ds")
    assert len(vintages) == 2
    assert vintages[0] < vintages[1]


def test_multi_field_and_year_partition(store):
    dates = ["2025-12-31", "2026-01-02"]  # 跨年分区
    t = ["AAA"]
    data = {
        "close": _wide(dates, t, [[10.0], [12.0]]),
        "volume": _wide(dates, t, [[1000.0], [1200.0]]),
    }
    store.append(data, as_of="2026-01-02T21:00:00", name="ds")
    out = store.load_pit(fields=["close", "volume"], name="ds")
    assert set(out.keys()) == {"close", "volume"}
    assert out["close"].loc[pd.Timestamp("2025-12-31"), "AAA"] == 10.0
    assert out["volume"].loc[pd.Timestamp("2026-01-02"), "AAA"] == 1200.0


def test_field_and_date_filtering(store):
    dates = ["2026-06-01", "2026-06-02", "2026-06-03"]
    t = ["AAA", "BBB"]
    vals = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    store.append({"close": _wide(dates, t, vals)}, as_of="2026-06-03T21:00:00", name="ds")
    out = store.load_pit(start="2026-06-02", end="2026-06-02", tickers=["AAA"], name="ds")
    close = out["close"]
    assert list(close.index) == [pd.Timestamp("2026-06-02")]
    assert list(close.columns) == ["AAA"]
    assert close.iloc[0, 0] == 3.0


def test_nonexistent_dataset_returns_empty(store):
    assert store.load_pit(name="does_not_exist") == {}


def test_nan_rows_dropped_on_append(store):
    d, t = ["2026-07-01"], ["AAA", "BBB"]
    # BBB 全 NaN 应被丢弃
    store.append({"close": _wide(d, t, [[9.0, np.nan]])},
                 as_of="2026-07-01T21:00:00", name="ds")
    long_df = store._load_long("ds")
    assert set(long_df["ticker"]) == {"AAA"}


def test_daily_ingest_appends_to_pit(tmp_path, monkeypatch):
    """集成：DailyIngest 通过验收后应把数据写入 PIT（Task 8.1 hook）。"""
    from app.tasks.daily_ingest import DailyIngest

    monkeypatch.setattr("app.config.settings.pit_store_dir", str(tmp_path / "pit"))

    dates = pd.bdate_range("2026-01-01", periods=40)
    tickers = ["AAA", "BBB", "CCC"]
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.normal(0, 1, (len(dates), len(tickers))), axis=0)

    def _fake_load(name, start, end, health_check=False):
        class _DS:
            data = {
                "open":  _wide(dates, tickers, base),
                "high":  _wide(dates, tickers, base + 1),
                "low":   _wide(dates, tickers, base - 1),
                "close": _wide(dates, tickers, base),
                "volume": _wide(dates, tickers, np.full((len(dates), len(tickers)), 1e6)),
                "vwap":  _wide(dates, tickers, base),
                "returns": _wide(dates, tickers, np.zeros((len(dates), len(tickers)))),
            }
        return _DS()

    monkeypatch.setattr(
        "app.core.data_engine.dataset_registry.load_registry_dataset", _fake_load
    )
    monkeypatch.setattr(
        "app.core.data_engine.dataset_registry.check_dataset_health",
        lambda ds, min_score, warn_only: None,  # None → 视为满分
    )

    res = DailyIngest(min_health=0.7).ingest("ds", "2026-01-01", "2026-03-01")
    assert res.accepted

    store = PITStore(str(tmp_path / "pit"))
    out = store.load_pit(name="ds")
    assert "close" in out and not out["close"].empty
    assert set(out["close"].columns) == set(tickers)
