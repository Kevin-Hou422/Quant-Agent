"""
test_ml_optimizer.py — Optuna 参数优化器测试

覆盖：SearchSpace 默认值/自定义值、AlphaOptimizer 返回结构、
参数在搜索范围内、OOS 隔离、单次 trial。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 数据工厂
# ---------------------------------------------------------------------------

def _make_dataset(n_days: int = 80, n_tickers: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0),
        index=dates, columns=tickers,
    )
    volume = pd.DataFrame(
        rng.integers(500_000, 2_000_000, (n_days, n_tickers)).astype(float),
        index=dates, columns=tickers,
    )
    high  = close * 1.01
    low   = close * 0.99
    open_ = close * (1 + rng.normal(0, 0.005, (n_days, n_tickers)))
    vwap  = (high + low + close) / 3
    return {
        "close": close, "open": open_, "high": high, "low": low,
        "volume": volume, "vwap": vwap,
        "returns": close.pct_change().fillna(0.0),
    }


def _split(data, oos_ratio=0.3):
    n = len(next(iter(data.values())))
    n_is = int(n * (1 - oos_ratio))
    is_ = {k: v.iloc[:n_is] for k, v in data.items()}
    oos = {k: v.iloc[n_is:] for k, v in data.items()}
    return is_, oos


# ---------------------------------------------------------------------------
# SearchSpace 测试
# ---------------------------------------------------------------------------

class TestSearchSpace:

    def test_defaults(self):
        from app.core.ml_engine.alpha_optimizer import SearchSpace
        ss = SearchSpace()
        assert hasattr(ss, "delay_range")
        assert hasattr(ss, "decay_range")
        lo, hi = ss.delay_range
        assert lo >= 0 and hi > lo

    def test_custom_ranges(self):
        from app.core.ml_engine.alpha_optimizer import SearchSpace
        ss = SearchSpace(delay_range=(1, 3), decay_range=(2, 8))
        assert ss.delay_range == (1, 3)
        assert ss.decay_range == (2, 8)


# ---------------------------------------------------------------------------
# AlphaOptimizer 测试
# ---------------------------------------------------------------------------

class TestAlphaOptimizer:
    """AlphaOptimizer API: AlphaOptimizer(dsl, is_dataset, search_space, n_trials)
    .optimize() → (SimulationConfig, StudySummary)
    """

    @pytest.fixture
    def is_data(self):
        return _make_dataset(n_days=80, n_tickers=8, seed=0)

    def _make_optimizer(self, dsl: str, is_data: dict, n_trials: int = 2):
        from app.core.ml_engine.alpha_optimizer import AlphaOptimizer, SearchSpace
        is_, _ = _split(is_data)
        ss = SearchSpace()
        return AlphaOptimizer(dsl=dsl, is_dataset=is_, search_space=ss, n_trials=n_trials)

    def test_optimizer_returns_tuple(self, is_data):
        dsl = "rank(ts_delta(log(close), 5))"
        opt = self._make_optimizer(dsl, is_data)
        result = opt.optimize()
        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_best_config_is_simulation_config(self, is_data):
        from app.core.ml_engine.alpha_optimizer import SearchSpace
        dsl = "rank(ts_delta(log(close), 5))"
        opt = self._make_optimizer(dsl, is_data)
        sim_config, study_summary = opt.optimize()
        assert sim_config is not None
        assert hasattr(sim_config, "delay")

    def test_best_config_delay_in_range(self, is_data):
        from app.core.ml_engine.alpha_optimizer import SearchSpace
        dsl = "rank(ts_delta(log(close), 5))"
        opt = self._make_optimizer(dsl, is_data)
        sim_config, _ = opt.optimize()
        ss = SearchSpace()
        lo, hi = ss.delay_range
        assert lo <= sim_config.delay <= hi

    def test_n_trials_1_succeeds(self, is_data):
        dsl = "rank(close)"
        opt = self._make_optimizer(dsl, is_data, n_trials=1)
        result = opt.optimize()
        assert result is not None

    def test_oos_isolation_no_oos_attribute(self, is_data):
        """优化器在优化阶段不应有 oos_dataset 属性（防止数据泄露）。"""
        from app.core.ml_engine.alpha_optimizer import AlphaOptimizer, SearchSpace
        is_, _ = _split(is_data)
        ss = SearchSpace()
        opt = AlphaOptimizer(dsl="rank(close)", is_dataset=is_, search_space=ss, n_trials=1)
        assert not hasattr(opt, "oos_dataset")

    def test_study_summary_has_best_value(self, is_data):
        dsl = "rank(ts_delta(log(close), 5))"
        opt = self._make_optimizer(dsl, is_data)
        _, study_summary = opt.optimize()
        if study_summary is not None:
            assert hasattr(study_summary, "best_value") or "best_value" in dir(study_summary)
