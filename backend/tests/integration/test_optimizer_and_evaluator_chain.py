"""
test_phase2.py — Phase 2 高级评估 + Optuna 优化器测试

覆盖：
  1. SearchSpace 和 AlphaOptimizer（Optuna）
  2. AlphaEvaluator（滚动指标、IC Decay、过拟合评分）
  3. /alpha/simulate 和 /alpha/optimize API 端点（FastAPI TestClient）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 测试数据工厂
# ---------------------------------------------------------------------------

def _make_dataset(n_tickers: int = 20, n_days: int = 200, seed: int = 7):
    rng     = np.random.default_rng(seed)
    dates   = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    close   = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0),
        index=dates, columns=tickers,
    )
    volume  = pd.DataFrame(
        rng.integers(500_000, 5_000_000, (n_days, n_tickers)).astype(float),
        index=dates, columns=tickers,
    )
    high    = close * (1 + rng.uniform(0, 0.02, close.shape))
    low     = close * (1 - rng.uniform(0, 0.02, close.shape))
    open_   = close.shift(1).fillna(close)
    vwap    = (high + low + close) / 3
    returns = close.pct_change().fillna(0.0)
    return {
        "close":   close, "open": open_, "high": high,
        "low":     low,   "volume": volume,
        "vwap":    vwap,  "returns": returns,
    }


def _get_is_oos(n_days=200, oos_ratio=0.30):
    from app.core.data_engine.data_partitioner import DataPartitioner
    ds    = _make_dataset(n_days=n_days)
    dates = ds["close"].index
    dp    = DataPartitioner(
        start     = str(dates[0].date()),
        end       = str(dates[-1].date()),
        oos_ratio = oos_ratio,
    )
    part = dp.partition(ds)
    return part.train(), part.test(), ds


DSL = "rank(ts_delta(log(close),5))"


# ===========================================================================
# 1. SearchSpace & AlphaOptimizer
# ===========================================================================

class TestSearchSpace:
    def test_defaults(self):
        from app.core.ml_engine.alpha_optimizer import SearchSpace
        ss = SearchSpace()
        assert ss.delay_range == (0, 5)
        assert ss.decay_range == (0, 10)
        assert "long_short" in ss.portfolio_modes

    def test_custom(self):
        from app.core.ml_engine.alpha_optimizer import SearchSpace
        ss = SearchSpace(delay_range=(1, 3), portfolio_modes=("long_short",))
        assert ss.delay_range == (1, 3)
        assert len(ss.portfolio_modes) == 1


class TestAlphaOptimizer:
    def test_optimize_returns_config(self):
        from app.core.ml_engine.alpha_optimizer import AlphaOptimizer, SearchSpace
        is_data, _, _ = _get_is_oos()
        opt = AlphaOptimizer(dsl=DSL, is_dataset=is_data,
                             search_space=SearchSpace(), n_trials=5, seed=0)
        best_cfg, summary = opt.optimize()
        assert best_cfg is not None
        assert summary.n_trials == 5

    def test_best_config_valid_types(self):
        from app.core.ml_engine.alpha_optimizer import AlphaOptimizer, SearchSpace
        from app.core.alpha_engine.signal_processor import SimulationConfig
        is_data, _, _ = _get_is_oos()
        opt = AlphaOptimizer(dsl=DSL, is_dataset=is_data,
                             search_space=SearchSpace(), n_trials=5, seed=1)
        best_cfg, _ = opt.optimize()
        assert isinstance(best_cfg, SimulationConfig)
        assert isinstance(best_cfg.delay, int)
        assert isinstance(best_cfg.decay_window, int)
        assert 0.0 <= best_cfg.truncation_min_q < 0.5
        assert 0.5 < best_cfg.truncation_max_q <= 1.0

    def test_study_summary_fields(self):
        from app.core.ml_engine.alpha_optimizer import AlphaOptimizer, SearchSpace
        is_data, _, _ = _get_is_oos()
        opt = AlphaOptimizer(dsl=DSL, is_dataset=is_data,
                             search_space=SearchSpace(), n_trials=4, seed=2)
        _, summary = opt.optimize()
        d = summary.to_dict()
        assert "best_value" in d
        assert "best_params" in d
        assert isinstance(d["trial_values"], list)

    def test_oos_isolation_no_attribute(self):
        """AlphaOptimizer 内部不应持有任何 oos_dataset 属性。"""
        from app.core.ml_engine.alpha_optimizer import AlphaOptimizer
        is_data, _, _ = _get_is_oos()
        opt = AlphaOptimizer(dsl=DSL, is_dataset=is_data, n_trials=2, seed=3)
        assert not hasattr(opt, "oos_dataset")
        assert not hasattr(opt, "_oos_dataset")

    def test_fitness_formula(self):
        """best_value 应 > -999（至少有一个 trial 成功）。"""
        from app.core.ml_engine.alpha_optimizer import AlphaOptimizer
        is_data, _, _ = _get_is_oos()
        opt = AlphaOptimizer(dsl=DSL, is_dataset=is_data, n_trials=6, seed=4)
        _, summary = opt.optimize()
        assert summary.best_value > -999.0


# ===========================================================================
# 2. AlphaEvaluator
# ===========================================================================

class TestAlphaEvaluator:
    def _get_reports(self, with_oos=True):
        from app.core.alpha_engine.signal_processor import SimulationConfig
        from app.core.backtest_engine.realistic_backtester import RealisticBacktester
        is_data, oos_data, _ = _get_is_oos()
        cfg = SimulationConfig(delay=1)
        bt  = RealisticBacktester(config=cfg)
        res = bt.run(DSL, is_data, oos_dataset=oos_data if with_oos else None)
        return res, is_data, oos_data

    def test_is_only_evaluation(self):
        from app.core.ml_engine.alpha_evaluator import AlphaEvaluator
        res, is_data, _ = self._get_reports(with_oos=False)
        ev = AlphaEvaluator()
        result = ev.evaluate(
            is_report=res.is_report, is_prices=is_data["close"],
            is_signal=res.processed_signal,
        )
        assert result.is_metrics is not None
        assert result.oos_metrics is None
        assert result.overfitting_score == 0.0
        assert result.is_overfit is False

    def test_is_oos_evaluation(self):
        from app.core.ml_engine.alpha_evaluator import AlphaEvaluator
        from app.core.alpha_engine.signal_processor import SignalProcessor, SimulationConfig
        from app.core.alpha_engine.parser import Parser
        from app.core.alpha_engine.dsl_executor import Executor
        res, is_data, oos_data = self._get_reports(with_oos=True)
        cfg = SimulationConfig(delay=1)
        node      = Parser().parse(DSL)
        raw_oos   = Executor().run(node, oos_data)
        oos_signal = SignalProcessor(cfg).process(raw_oos)

        ev = AlphaEvaluator()
        result = ev.evaluate(
            is_report=res.is_report,   is_prices=is_data["close"],
            is_signal=res.processed_signal,
            oos_report=res.oos_report,  oos_prices=oos_data["close"],
            oos_signal=oos_signal,
        )
        assert result.oos_metrics is not None
        assert 0.0 <= result.overfitting_score <= 1.0

    def test_overfitting_score_range(self):
        from app.core.ml_engine.alpha_evaluator import AlphaEvaluator, EvalMetrics
        ev = AlphaEvaluator(overfit_threshold=0.5)
        is_m  = EvalMetrics(sharpe_ratio=2.0)
        oos_m = EvalMetrics(sharpe_ratio=0.5)   # 退化 75%
        score, is_overfit = ev._overfit_score(is_m, oos_m)
        assert 0.0 <= score <= 1.0
        assert is_overfit is True

    def test_no_overfit_flag(self):
        from app.core.ml_engine.alpha_evaluator import AlphaEvaluator, EvalMetrics
        ev = AlphaEvaluator(overfit_threshold=0.5)
        is_m  = EvalMetrics(sharpe_ratio=1.0)
        oos_m = EvalMetrics(sharpe_ratio=0.9)   # 退化 10%
        score, is_overfit = ev._overfit_score(is_m, oos_m)
        assert score < 0.5
        assert is_overfit is False

    def test_ic_decay_keys(self):
        from app.core.ml_engine.alpha_evaluator import AlphaEvaluator
        res, is_data, _ = self._get_reports(with_oos=False)
        ev = AlphaEvaluator()
        result = ev.evaluate(
            is_report=res.is_report, is_prices=is_data["close"],
            is_signal=res.processed_signal,
        )
        assert "t1" in result.ic_decay
        assert "t5" in result.ic_decay

    def test_summary_no_crash(self):
        from app.core.ml_engine.alpha_evaluator import AlphaEvaluator
        res, is_data, _ = self._get_reports(with_oos=False)
        ev = AlphaEvaluator()
        result = ev.evaluate(
            is_report=res.is_report, is_prices=is_data["close"],
            is_signal=res.processed_signal,
        )
        summary = result.summary()
        assert "IS" in summary or "过拟合" in summary

    def test_to_dict_serializable(self):
        from app.core.ml_engine.alpha_evaluator import AlphaEvaluator
        import json
        res, is_data, _ = self._get_reports(with_oos=False)
        ev = AlphaEvaluator()
        result = ev.evaluate(
            is_report=res.is_report, is_prices=is_data["close"],
            is_signal=res.processed_signal,
        )
        d = result.to_dict()
        # 可序列化为 JSON（不崩溃）
        json_str = json.dumps(d, default=str)
        assert "is_metrics" in json_str


# ===========================================================================
# 3. API 端点（FastAPI TestClient）
# ===========================================================================

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import sys, os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    from app.main import app
    return TestClient(app)


class TestAlphaSimulateAPI:
    def test_simulate_basic(self, client):
        resp = client.post("/api/alpha/simulate", json={
            "dsl":       DSL,
            "config":    {"delay": 1, "decay_window": 0},
            "n_tickers": 15,
            "n_days":    120,
            "oos_ratio": 0.30,
            "seed":      42,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "is_metrics" in body
        assert "oos_metrics" in body
        assert "overfitting_score" in body
        assert "ic_decay" in body
        assert body["best_config"] is None    # simulate 模式无最优参数

    def test_simulate_no_oos(self, client):
        resp = client.post("/api/alpha/simulate", json={
            "dsl":       DSL,
            "config":    {},
            "n_tickers": 15,
            "n_days":    120,
            "oos_ratio": 0.0,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["oos_metrics"] is None
        assert body["overfitting_score"] == 0.0


class TestAlphaOptimizeAPI:
    def test_optimize_returns_best_config(self, client):
        resp = client.post("/api/alpha/optimize", json={
            "dsl":      DSL,
            "search_space": {
                "delay_range":     [0, 2],
                "decay_range":     [0, 5],
                "trunc_min_range": [0.01, 0.10],
                "trunc_max_range": [0.90, 0.99],
                "portfolio_modes": ["long_short"],
            },
            "n_trials":   5,
            "n_tickers":  15,
            "n_days":     120,
            "oos_ratio":  0.30,
            "seed":       0,
        })
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["best_config"] is not None
        assert "delay" in body["best_config"]
        assert body["n_trials_run"] == 5

    def test_optimize_has_oos_metrics(self, client):
        resp = client.post("/api/alpha/optimize", json={
            "dsl":      DSL,
            "search_space": {"portfolio_modes": ["long_short"]},
            "n_trials": 4,
            "n_tickers": 15,
            "n_days":   120,
            "oos_ratio": 0.30,
            "seed": 1,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["oos_metrics"] is not None
        assert "sharpe_ratio" in body["oos_metrics"]
