"""
Smoke test for the data-engine pipeline.

Task 6.3（2026-07-30）重构：原文件是**模块级脚本**（import 即执行），在 pytest
收集期就抛 `KeyError: 'ticker'`，导致整个测试套件收集失败（此前靠 `--ignore` 屏蔽）。
现拆为两个 pytest 函数：
  - Schema / Panel / Universe 三步彼此独立 → 正常通过（真实覆盖）
  - Preprocessor 及其下游（HealthCheck / FeatureStore / DataManager）依赖 Preprocessor，
    而 `Preprocessor.apply` 的 `groupby('ticker')` 在 pandas 升级后 groupby.apply 行为
    变化下丢失了 'ticker' 列 → KeyError。这是**真实数据引擎 bug**（属 Phase 7/8 数据摄取
    路径），此处用 xfail 记录，修复后自动转 xpass 提醒。
"""
import tempfile

import numpy as np
import pandas as pd
import pytest

from app.core.data_engine import (
    DataProvider, STANDARD_COLUMNS,
    SchemaEnforcer, PanelFactory, UniverseFilter,
    Preprocessor, DataHealthChecker,
    ParquetFeatureStore, DataChunker, DataManager,
)


# ---------------------------------------------------------------------------
# 合成数据构造（函数内，不在模块级执行）
# ---------------------------------------------------------------------------

def _build_raw_df():
    dates   = pd.date_range("2023-01-02", periods=60, freq="B")
    tickers = ["AAPL", "MSFT", "DELIST"]
    rng     = np.random.default_rng(42)
    rows    = []
    for ticker in tickers:
        for i, dt in enumerate(dates):
            if ticker == "DELIST" and i >= 30:
                continue
            close  = 100 + rng.normal(0, 1)
            volume = 0 if (ticker == "AAPL" and i == 5) else float(rng.integers(1_000_000, 10_000_000))
            rows.append({
                "timestamp": dt, "ticker": ticker,
                "open": close - 0.5, "high": close + 1.0, "low": close - 1.0,
                "close": close, "volume": volume,
                "adj_factor": 1.2 if (ticker == "AAPL" and i < 10) else 1.0,
            })
    return pd.DataFrame(rows), dates


# ---------------------------------------------------------------------------
# 步骤 1-3：Schema / Panel / Universe（独立，正常通过）
# ---------------------------------------------------------------------------

def test_schema_panel_universe():
    raw_df, dates = _build_raw_df()

    # 1. SchemaEnforcer
    schema_df = SchemaEnforcer().enforce(raw_df)
    assert list(schema_df.columns[:9]) == STANDARD_COLUMNS
    assert schema_df["adj_factor"].dtype == float

    # 2. PanelFactory：3 ticker × 60 天，DELIST 后 30 天为 NaN
    panel = PanelFactory().reindex_to_master([schema_df])
    assert len(panel) == 60 * 3
    delist_late = panel[(panel["ticker"] == "DELIST") & (panel["timestamp"] > dates[29])]
    assert delist_late["close"].isna().all()

    # 3. UniverseFilter（PIT）：as_of 末日应剔除已退市 DELIST
    universe_records = []
    for tk in ["AAPL", "MSFT"]:
        for dt in dates:
            universe_records.append({"date": dt, "ticker": tk, "active": True})
    for dt in dates[:30]:
        universe_records.append({"date": dt, "ticker": "DELIST", "active": True})
    for dt in dates[30:]:
        universe_records.append({"date": dt, "ticker": "DELIST", "active": False})
    uf = UniverseFilter(universe_df=pd.DataFrame(universe_records), strict=True)
    filtered = uf.filter(panel, as_of=str(dates[-1].date()))
    assert "DELIST" not in filtered["ticker"].unique()


# ---------------------------------------------------------------------------
# 步骤 4-7：Preprocessor 及下游（已知 data-engine bug → xfail）
# ---------------------------------------------------------------------------

def test_preprocessor_and_downstream():
    # 修复（2026-07-31）：Preprocessor.apply 的 groupby('ticker') 丢列 bug 已修
    # （改用 groupby[numeric_cols].ffill() 就地赋值），此测试从 xfail 转为正常通过。
    raw_df, dates = _build_raw_df()
    schema_df = SchemaEnforcer().enforce(raw_df)
    panel     = PanelFactory().reindex_to_master([schema_df])

    # 4. Preprocessor（此处触发 KeyError）
    proc, rpt = Preprocessor().run(panel, apply_adj=True, build_synthetic=True, ffill_limit=5)
    assert {"vwap", "returns", "log_returns"}.issubset(proc.columns)

    # 5. DataHealthChecker
    report = DataHealthChecker(spike_threshold=0.5).check(proc)
    assert len(report.zero_volume) >= 1
    assert 0.0 <= report.overall_score <= 1.0

    # 6. ParquetFeatureStore + DataChunker
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ParquetFeatureStore(tmpdir)
        store.save(proc, name="test_ds")
        loaded = store.load("test_ds", tickers=["AAPL", "MSFT"],
                            start="2023-01-01", end="2023-12-31")
        assert set(loaded["ticker"].unique()) == {"AAPL", "MSFT"}
        chunks = list(DataChunker(store, chunk_size=2).iter_chunks(
            "test_ds", ["AAPL", "MSFT", "DELIST"], "2023-01-01", "2023-12-31"))
        assert len(chunks) >= 1

    # 7. DataManager 端到端（MockProvider）
    class MockProvider(DataProvider):
        def __init__(self, data): self._data = data
        def available_fields(self): return list(STANDARD_COLUMNS)
        def fetch(self, tickers, start, end, fields=None): return {}
        def fetch_panel(self, tickers, start, end, fields=None):
            df = self._data[self._data["ticker"].isin(tickers)].copy()
            return df[(df["timestamp"] >= pd.Timestamp(start)) &
                      (df["timestamp"] <= pd.Timestamp(end))]

    dm = DataManager(providers=[MockProvider(schema_df)], cache_to_store=False)
    result_panel, result_health = dm.get_panel(
        tickers=["AAPL", "MSFT"], start_date="2023-01-02", end_date="2023-03-31",
        apply_adj=True, ffill_limit=3, run_health_check=True,
    )
    assert not result_panel.empty and result_health is not None
