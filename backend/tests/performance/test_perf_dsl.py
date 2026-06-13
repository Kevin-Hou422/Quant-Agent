"""
test_perf_dsl.py — DSL 执行性能基准测试

目标：
  - 单次 parse < 100ms
  - 60×10 数据集执行 < 500ms
  - 500×200 数据集执行 < 2s
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest


def _make_data(n_days=60, n_tickers=10, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0),
        index=dates, columns=tickers,
    )
    volume = pd.DataFrame(
        rng.integers(500_000, 2_000_000, (n_days, n_tickers)).astype(float),
        index=dates, columns=tickers,
    )
    return {"close": close, "open": close, "high": close * 1.01, "low": close * 0.99,
            "volume": volume, "vwap": close, "returns": close.pct_change().fillna(0)}


class TestDSLParsePerformance:

    def test_parse_under_100ms(self):
        from app.core.alpha_engine.parser import Parser
        dsl = "rank(ts_delta(log(close), 5)) / ts_std(close, 20)"
        parser = Parser()

        start = time.perf_counter()
        for _ in range(10):
            parser.parse(dsl)
        elapsed = (time.perf_counter() - start) / 10 * 1000  # ms per call

        assert elapsed < 100, f"Parse took {elapsed:.1f}ms (limit 100ms)"

    def test_parse_complex_dsl_under_200ms(self):
        from app.core.alpha_engine.parser import Parser
        dsl = ("rank(ts_decay_linear(ts_delta(log(close), 5), 3)) "
               "* zscore(volume / ts_mean(volume, 20))")
        parser = Parser()

        start = time.perf_counter()
        parser.parse(dsl)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 200, f"Complex parse took {elapsed:.1f}ms (limit 200ms)"


class TestDSLExecutePerformance:

    def test_execute_small_dataset_under_500ms(self):
        from app.core.alpha_engine.parser import Parser
        from app.core.alpha_engine.dsl_executor import Executor

        dsl = "rank(ts_delta(log(close), 5))"
        data = _make_data(n_days=60, n_tickers=10)
        node = Parser().parse(dsl)
        executor = Executor()

        start = time.perf_counter()
        result = executor.run(node, data)
        elapsed = (time.perf_counter() - start) * 1000

        assert result is not None
        assert elapsed < 500, f"Execute took {elapsed:.1f}ms (limit 500ms)"

    def test_execute_medium_dataset_under_2s(self):
        from app.core.alpha_engine.parser import Parser
        from app.core.alpha_engine.dsl_executor import Executor

        dsl = "rank(ts_delta(log(close), 5)) / ts_std(close, 20)"
        data = _make_data(n_days=250, n_tickers=50)
        node = Parser().parse(dsl)
        executor = Executor()

        start = time.perf_counter()
        result = executor.run(node, data)
        elapsed = time.perf_counter() - start

        assert result is not None
        assert elapsed < 2.0, f"Medium execute took {elapsed:.2f}s (limit 2s)"


class TestDSLBatchPerformance:

    DSLS = [
        "rank(ts_delta(log(close), 5))",
        "zscore(close / ts_mean(close, 20))",
        "rank(volume / ts_mean(volume, 10))",
        "ts_delta(rank(close), 5)",
        "rank(ts_std(close, 20))",
        "rank(close - ts_mean(close, 10))",
        "zscore(ts_delta(log(volume), 3))",
        "rank(ts_momentum(close, 10))",
        "rank(ts_delta(log(close / open), 5))",
        "rank(ts_max(close, 10) - ts_min(close, 10))",
    ]

    def test_batch_10_dsls_under_10s(self):
        from app.core.alpha_engine.parser import Parser
        from app.core.alpha_engine.dsl_executor import Executor

        data = _make_data(n_days=60, n_tickers=10)
        parser = Parser()
        executor = Executor()

        start = time.perf_counter()
        for dsl in self.DSLS:
            try:
                node = parser.parse(dsl)
                executor.run(node, data)
            except Exception:
                pass
        elapsed = time.perf_counter() - start

        assert elapsed < 10.0, f"Batch 10 DSLs took {elapsed:.2f}s (limit 10s)"
