"""
test_gp_alpha_pool.py — AlphaPool 去重、相关性过滤、容量管理测试
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core.gp_engine.alpha_pool import AlphaPool, PoolEntry


# ---------------------------------------------------------------------------
# 帮助
# ---------------------------------------------------------------------------

def _entry(
    dsl: str,
    fitness: float = 0.5,
    signal_vec: np.ndarray | None = None,
    sharpe_oos: float = 0.5,
) -> PoolEntry:
    if signal_vec is None:
        rng = np.random.default_rng(abs(hash(dsl)) % 2**31)
        signal_vec = rng.normal(0, 1, 100)
    return PoolEntry(
        dsl=dsl,
        fitness=fitness,
        sharpe_is=sharpe_oos + 0.2,
        sharpe_oos=sharpe_oos,
        turnover=0.5,
        overfitting_score=0.1,
        generation=0,
        signal_vec=signal_vec,
    )


# ---------------------------------------------------------------------------
# 去重测试
# ---------------------------------------------------------------------------

class TestDeduplication:

    def test_add_exact_duplicate_rejected(self):
        """完全相同的 DSL 字符串，第二次 add 应返回 False。"""
        pool = AlphaPool()
        e1 = _entry("rank(close)")
        e2 = _entry("rank(close)", fitness=0.9)  # 相同 DSL，更好的 fitness

        assert pool.add(e1) is True
        assert pool.add(e2) is False   # 应被拒绝
        assert len(pool) == 1

    def test_add_different_dsl_accepted(self):
        pool = AlphaPool()
        rng = np.random.default_rng(0)
        e1 = _entry("rank(close)",  signal_vec=rng.normal(0, 1, 100))
        e2 = _entry("zscore(open)", signal_vec=rng.normal(0, 1, 100))

        pool.add(e1)
        pool.add(e2)
        assert len(pool) == 2


# ---------------------------------------------------------------------------
# 相关性过滤测试
# ---------------------------------------------------------------------------

class TestCorrelationFilter:

    def test_correlated_signal_rejected(self):
        """与已有信号高度相关（r > 0.70）的新条目应被拒绝。"""
        pool = AlphaPool(corr_threshold=0.70)
        base_vec = np.linspace(0, 1, 100)

        e1 = _entry("alpha_A", signal_vec=base_vec.copy())
        # 相关系数接近 1.0
        e2 = _entry("alpha_B", signal_vec=base_vec * 2.0 + 0.5)

        pool.add(e1)
        accepted = pool.add(e2)
        assert accepted is False

    def test_uncorrelated_signal_accepted(self):
        """与已有信号低相关的新条目应被接受。"""
        pool = AlphaPool(corr_threshold=0.70)
        rng = np.random.default_rng(42)

        vec_a = rng.normal(0, 1, 100)
        vec_b = rng.normal(0, 1, 100)   # 独立正态，相关性约 0

        e1 = _entry("alpha_A", signal_vec=vec_a)
        e2 = _entry("alpha_B", signal_vec=vec_b)

        pool.add(e1)
        accepted = pool.add(e2)
        assert accepted is True

    def test_no_signal_vec_always_accepted(self):
        """不带 signal_vec 的条目不进行相关性检查，应被接受。"""
        pool = AlphaPool()
        e1 = _entry("alpha_A")
        e1.signal_vec = None
        e2 = _entry("alpha_B")
        e2.signal_vec = None

        pool.add(e1)
        assert pool.add(e2) is True


# ---------------------------------------------------------------------------
# top_k / best 测试
# ---------------------------------------------------------------------------

class TestTopK:

    def _filled_pool(self, n: int = 5) -> AlphaPool:
        pool = AlphaPool()
        rng = np.random.default_rng(0)
        for i in range(n):
            vec = rng.normal(i, 1, 100)  # 不同向量，低相关
            e = _entry(f"alpha_{i}", fitness=float(i), signal_vec=vec)
            pool.add(e)
        return pool

    def test_top_k_sorted_by_fitness_descending(self):
        pool = self._filled_pool(5)
        top3 = pool.top_k(3)
        assert len(top3) == 3
        fitnesses = [e.fitness for e in top3]
        assert fitnesses == sorted(fitnesses, reverse=True)

    def test_top_k_with_fewer_entries_than_k(self):
        pool = self._filled_pool(3)
        top10 = pool.top_k(10)
        assert len(top10) == 3  # 只有 3 条

    def test_best_returns_highest_fitness(self):
        pool = self._filled_pool(5)
        best = pool.best()
        assert best is not None
        assert best.fitness == max(e.fitness for e in pool.all_entries())

    def test_best_returns_none_on_empty_pool(self):
        pool = AlphaPool()
        assert pool.best() is None


# ---------------------------------------------------------------------------
# population_diagnostics 测试
# ---------------------------------------------------------------------------

class TestPopulationDiagnostics:

    def test_diagnostics_returns_dict(self):
        pool = AlphaPool()
        rng = np.random.default_rng(0)
        for i in range(3):
            e = _entry(f"a_{i}", signal_vec=rng.normal(i, 1, 100))
            pool.add(e)
        diag = pool.population_diagnostics()
        assert isinstance(diag, dict)
        assert "mean_sharpe_oos" in diag
        assert "mean_turnover" in diag
        assert "mean_overfit" in diag

    def test_diagnostics_empty_pool(self):
        pool = AlphaPool()
        diag = pool.population_diagnostics()
        assert diag["mean_sharpe_oos"] == 0.0
        assert diag["mean_turnover"] == 1.0


# ---------------------------------------------------------------------------
# get_orthogonal_signals 测试
# ---------------------------------------------------------------------------

class TestOrthogonalSignals:

    def test_small_pool_returns_original_vectors(self):
        """pool.size < 2 时返回原始向量（无 PCA）。"""
        pool = AlphaPool()
        vec = np.array([1.0, 2.0, 3.0])
        e = _entry("alpha_A", signal_vec=vec)
        e.signal_vec = None  # 触发 <2 分支
        pool._entries.append(e)

        orth = pool.get_orthogonal_signals()
        assert isinstance(orth, dict)

    def test_multi_entry_returns_dict(self):
        pool = AlphaPool()
        rng = np.random.default_rng(99)
        for i in range(4):
            vec = rng.normal(i * 2, 1, 50)
            e = PoolEntry(
                dsl=f"a_{i}", fitness=float(i), sharpe_is=0.5, sharpe_oos=0.4,
                turnover=0.5, overfitting_score=0.1, generation=0, signal_vec=vec,
            )
            pool._entries.append(e)
            pool._seen_dsls.add(f"a_{i}")

        orth = pool.get_orthogonal_signals()
        assert isinstance(orth, dict)
        assert len(orth) == 4


# ---------------------------------------------------------------------------
# 容量管理测试
# ---------------------------------------------------------------------------

class TestCapacityManagement:

    def test_max_size_enforced(self):
        pool = AlphaPool(max_size=3)
        rng = np.random.default_rng(0)
        for i in range(5):
            vec = rng.normal(i * 3, 1, 100)
            e = _entry(f"a_{i}", fitness=float(i), signal_vec=vec)
            pool.add(e)
        assert len(pool) <= 3
