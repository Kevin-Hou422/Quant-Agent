"""
test_phase6_reproducibility.py — Task 6.5 可复现性验收（2026-07-30）

覆盖：
  R-N1  GP 全链路确定性：相同 seed + 相同数据 → 相同 GP 输出（best_dsl）
  6.5   RunManifest：数据哈希确定性/值敏感、store 往返、verify_dataset
        回测确定性：相同 DSL+数据 两次回测 → 指标逐位一致
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# R-N1 — GP 端到端确定性
# ===========================================================================

class TestGPDeterminism:

    def test_diverse_seeds_reproducible_regardless_of_global_random(self):
        import random
        from app.core.workflows.alpha_workflows import _generate_diverse_seeds
        random.seed(111)
        a = _generate_diverse_seeds("momentum alpha", n_target=8, seed=42)
        random.seed(999)                                  # 污染全局 random
        b = _generate_diverse_seeds("momentum alpha", n_target=8, seed=42)
        assert a == b and len(a) >= 2

    def test_generate_random_alpha_reproducible(self):
        import random
        from app.core.gp_engine import _rng
        from app.core.gp_engine.gp_engine import generate_random_alpha
        _rng.bind_seed(7); x1 = repr(generate_random_alpha())
        random.seed(555)
        _rng.bind_seed(7); x2 = repr(generate_random_alpha())
        assert x1 == x2

    def test_mutation_reproducible(self):
        from app.core.gp_engine import _rng
        from app.core.gp_engine.mutations import point_mutation
        from app.core.alpha_engine.parser import Parser
        node = Parser().parse("rank(ts_delta(close, 5))")
        _rng.bind_seed(3); m1 = repr(point_mutation(node))
        _rng.bind_seed(3); m2 = repr(point_mutation(node))
        assert m1 == m2

    def test_population_evolver_reproducible(self, make_dataset):
        """A4 核心：相同 seed 两次完整 GP 演化 → 相同 best_dsl。"""
        from app.core.gp_engine.population_evolver import PopulationEvolver
        is_data  = make_dataset(n_days=120, n_tickers=8, seed=1)
        oos_data = make_dataset(n_days=60,  n_tickers=8, seed=2)

        def _run():
            ev = PopulationEvolver(is_data=is_data, oos_data=oos_data,
                                   pop_size=8, n_generations=2, seed=123)
            return ev.run(seed_dsls=["rank(ts_delta(close, 5))",
                                     "zscore(ts_mean(returns, 10))"],
                          n_optuna_trials=0)

        r1, r2 = _run(), _run()
        assert r1.best_dsl == r2.best_dsl


# ===========================================================================
# 6.5 — RunManifest 可复现性台账
# ===========================================================================

class TestRunManifest:

    @pytest.fixture
    def dataset(self):
        idx = pd.bdate_range("2023-01-02", periods=10)
        cols = ["A", "B"]
        return {
            "close":  pd.DataFrame(np.arange(20).reshape(10, 2) * 1.0, index=idx, columns=cols),
            "volume": pd.DataFrame(np.ones((10, 2)), index=idx, columns=cols),
        }

    def test_hash_deterministic_and_order_independent(self, dataset):
        from app.db.run_manifest import dataset_sha256
        reordered = {"volume": dataset["volume"], "close": dataset["close"]}
        assert dataset_sha256(dataset) == dataset_sha256(reordered)

    def test_hash_value_sensitive(self, dataset):
        from app.db.run_manifest import dataset_sha256
        h0 = dataset_sha256(dataset)
        perturbed = {"close": dataset["close"].copy(), "volume": dataset["volume"]}
        perturbed["close"].iloc[0, 0] = 999.0
        assert dataset_sha256(perturbed) != h0

    def test_store_roundtrip_and_verify(self, dataset, tmp_path):
        from app.db.run_manifest import RunManifestStore
        store = RunManifestStore(db_url=f"sqlite:///{tmp_path/'rm.db'}")
        mid = store.record("backtest", dataset, seed=42,
                           config={"delay": 1}, summary={"sharpe": 1.23})
        rec = store.get(mid)
        assert rec is not None and rec.seed == 42 and rec.run_type == "backtest"
        # verify：同数据 True，改数据 False
        assert store.verify_dataset(mid, dataset) is True
        perturbed = {"close": dataset["close"].copy(), "volume": dataset["volume"]}
        perturbed["close"].iloc[0, 0] = 999.0
        assert store.verify_dataset(mid, perturbed) is False

    def test_verify_missing_manifest_raises(self, tmp_path, dataset):
        from app.db.run_manifest import RunManifestStore
        store = RunManifestStore(db_url=f"sqlite:///{tmp_path/'rm.db'}")
        with pytest.raises(KeyError):
            store.verify_dataset(99999, dataset)


# ===========================================================================
# 回测确定性（RunManifest 重放的前提）
# ===========================================================================

class TestBacktestDeterminism:

    def test_same_dsl_same_data_identical_metrics(self, make_dataset):
        from app.core.backtest_engine.realistic_backtester import RealisticBacktester
        from app.core.alpha_engine.signal_processor import SimulationConfig
        ds = make_dataset(n_days=150, n_tickers=10, seed=5)
        cfg = SimulationConfig()

        def _sharpe():
            r = RealisticBacktester(config=cfg).run("rank(ts_delta(close, 5))", ds)
            return r.is_report.sharpe_ratio, r.is_report.ann_turnover, r.is_report.max_drawdown

        assert _sharpe() == _sharpe()          # 逐位一致
