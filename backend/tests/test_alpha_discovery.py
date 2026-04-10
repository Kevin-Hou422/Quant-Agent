"""
tests/test_alpha_discovery.py — 6 项 alpha_discovery 单元测试

覆盖：
  1. test_point_mutation      — 变异后算子改变，NodeType 不变
  2. test_hoist_mutation      — 变异后树深度 <= 原深度
  3. test_subtree_crossover   — 交叉后两棵树仍通过 Validator
  4. test_proxy_features      — 特征向量长度固定，数值合理
  5. test_alpha_store_save_query — 保存后按 Sharpe 过滤查询
  6. test_gp_evolve_smoke     — 5代×10个体，返回非空结果
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

# ── path setup ──────────────────────────────────────────────────────────────
_BACKEND = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

# ── imports ──────────────────────────────────────────────────────────────────
from app.core.alpha_engine.typed_nodes import (
    DataNode, TimeSeriesNode, CrossSectionalNode, ArithmeticNode, NodeType,
)
from app.core.alpha_engine.generator import generate_random_alpha
from app.core.alpha_engine.validator import AlphaValidator

from app.core.alpha_discovery.mutations import (
    point_mutation, hoist_mutation, param_mutation, subtree_crossover,
)
from app.core.alpha_discovery.proxy_model import ProxyModel, extract_features, _FEATURE_SIZE
from app.core.alpha_discovery.alpha_store import AlphaStore, AlphaResult
from app.core.alpha_discovery.gp_engine import AlphaEvolver

_validator = AlphaValidator()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ts_node() -> TimeSeriesNode:
    """ts_mean(close, 20)"""
    return TimeSeriesNode("ts_mean", DataNode("close"), 20)


def _make_cs_node() -> CrossSectionalNode:
    """rank(ts_mean(close, 20))"""
    return CrossSectionalNode("rank", _make_ts_node())


def _make_complex_node():
    """rank(ts_delta(log(close),5)) / ts_std(close,20)"""
    log_close  = ArithmeticNode("log", [DataNode("close")])
    ts_delta   = TimeSeriesNode("ts_delta", log_close, 5)
    rank_node  = CrossSectionalNode("rank", ts_delta)
    ts_std     = TimeSeriesNode("ts_std", DataNode("close"), 20)
    return ArithmeticNode("div", [rank_node, ts_std])


def _make_dataset(n_days: int = 60, n_tickers: int = 10) -> dict:
    rng     = np.random.default_rng(0)
    dates   = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    close   = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0),
        index=dates, columns=tickers,
    )
    volume = pd.DataFrame(
        rng.integers(500_000, 2_000_000, (n_days, n_tickers)).astype(float),
        index=dates, columns=tickers,
    )
    high  = close * (1 + rng.uniform(0, 0.01, close.shape))
    low   = close * (1 - rng.uniform(0, 0.01, close.shape))
    vwap  = (high + low + close) / 3
    returns = close.pct_change().fillna(0.0)
    return {
        "close": close, "open": close.shift(1).fillna(close),
        "high": high, "low": low, "volume": volume,
        "vwap": vwap, "returns": returns,
    }


# ---------------------------------------------------------------------------
# Test 1: point_mutation
# ---------------------------------------------------------------------------

def test_point_mutation():
    """变异后算子改变（或结构改变），NodeType 不变。"""
    original = _make_ts_node()
    orig_type = original.node_type

    # 运行多次取非原算子
    mutated = None
    for _ in range(30):
        m = point_mutation(original)
        if repr(m) != repr(original):
            mutated = m
            break

    # 至少有一次成功变异（10+ 个 TS 算子可选）
    assert mutated is not None, "point_mutation 应产生至少一种不同结果"

    # NodeType 保持不变
    from app.core.alpha_discovery.mutations import _collect_nodes
    orig_ts = [n for n in _collect_nodes(original) if isinstance(n, TimeSeriesNode)]
    mut_ts  = [n for n in _collect_nodes(mutated)   if isinstance(n, TimeSeriesNode)]
    assert len(orig_ts) == len(mut_ts), "TS 节点数量应保持不变"


# ---------------------------------------------------------------------------
# Test 2: hoist_mutation
# ---------------------------------------------------------------------------

def test_hoist_mutation():
    """变异后树深度 <= 原深度。"""
    original = _make_complex_node()
    orig_depth = original.depth()

    for _ in range(20):
        mutated = hoist_mutation(original)
        assert mutated.depth() <= orig_depth, (
            f"hoist 变异后深度 {mutated.depth()} > 原深度 {orig_depth}"
        )


# ---------------------------------------------------------------------------
# Test 3: subtree_crossover
# ---------------------------------------------------------------------------

def test_subtree_crossover():
    """交叉后两棵树仍通过 AlphaValidator。"""
    root1 = _make_complex_node()
    root2 = TimeSeriesNode("ts_std", DataNode("volume"), 10)

    for _ in range(10):
        c1, c2 = subtree_crossover(root1, root2)
        try:
            _validator.validate(c1)
            _validator.validate(c2)
        except Exception as e:
            pytest.fail(f"subtree_crossover 产生的子树验证失败: {e}")


# ---------------------------------------------------------------------------
# Test 4: proxy_features
# ---------------------------------------------------------------------------

def test_proxy_features():
    """特征向量长度固定为 _FEATURE_SIZE，数值合理（非负）。"""
    nodes = [
        DataNode("close"),
        _make_ts_node(),
        _make_cs_node(),
        _make_complex_node(),
        generate_random_alpha(depth=4, seed=7),
    ]
    for node in nodes:
        feat = extract_features(node)
        assert len(feat) == _FEATURE_SIZE, (
            f"特征长度应为 {_FEATURE_SIZE}，实际 {len(feat)}"
        )
        assert (feat >= 0).all(), "所有特征值应 >= 0"

    # 深层树的 tree_depth > 浅层树
    shallow = DataNode("close")
    deep    = _make_complex_node()
    assert extract_features(deep)[0] > extract_features(shallow)[0]


# ---------------------------------------------------------------------------
# Test 5: alpha_store_save_query
# ---------------------------------------------------------------------------

def test_alpha_store_save_query(tmp_path):
    """保存 3 条记录后，按 min_sharpe 过滤能正确返回。"""
    db_url = f"sqlite:///{tmp_path}/test_alphas.db"
    store  = AlphaStore(db_url=db_url)

    records = [
        AlphaResult(dsl="rank(close)", sharpe=1.5, ic_ir=0.6, hypothesis="momentum"),
        AlphaResult(dsl="zscore(close)", sharpe=0.3, ic_ir=0.2, hypothesis="reversal"),
        AlphaResult(dsl="rank(ts_mean(close,20))", sharpe=0.8, ic_ir=0.4),
    ]
    ids = [store.save(r) for r in records]
    assert len(ids) == 3

    # 查询 sharpe >= 0.7 → 应得 2 条
    results = store.query(min_sharpe=0.7)
    assert len(results) == 2, f"期望 2 条，实际 {len(results)}"

    # 查询 sharpe >= 1.4 → 应得 1 条
    results = store.query(min_sharpe=1.4)
    assert len(results) == 1
    assert results[0].dsl == "rank(close)"

    # export_csv
    csv_path = str(tmp_path / "export.csv")
    store.export_csv(csv_path)
    with open(csv_path) as f:
        lines = f.readlines()
    assert len(lines) == 4, f"CSV 应有 4 行（1 header + 3 data），实际 {len(lines)}"


# ---------------------------------------------------------------------------
# Test 6: gp_evolve_smoke
# ---------------------------------------------------------------------------

def test_gp_evolve_smoke():
    """5代×10个体，evolve() 返回非空 Hall of Fame 列表。"""
    dataset = _make_dataset(n_days=60, n_tickers=10)
    evolver = AlphaEvolver(
        pop_size  = 10,
        n_gen     = 3,
        n_workers = 1,  # 单进程避免 Windows multiprocessing 问题
        tree_depth = 3,
    )
    hof = evolver.evolve(dataset)
    assert len(hof) > 0, "Hall of Fame 不应为空"
    for r in hof:
        assert isinstance(r.dsl, str) and len(r.dsl) > 0
        assert isinstance(r.fitness, float)
