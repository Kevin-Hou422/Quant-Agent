"""
Unit tests for the Alpha DSL Engine.

Target expression (appears in multiple tests):
    rank(ts_delta(log(close),5))/ts_std(close,20)

Run with:
    pytest tests/test_dsl_engine.py -v
"""

from __future__ import annotations

import sys
import os

# Make sure the backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from app.core.alpha_engine.typed_nodes import (
    Node, NodeType, ScalarNode, DataNode,
    TimeSeriesNode, CrossSectionalNode, ArithmeticNode,
)
from app.core.alpha_engine.validator import AlphaValidator, ValidationError
from app.core.alpha_engine.parser import Parser, ParseError
from app.core.alpha_engine.dsl_executor import Executor
from app.core.alpha_engine.fast_ops import (
    bn_ts_mean, bn_ts_std, ts_delta, cs_rank,
    cs_zscore, cs_scale, ind_neutralize, signed_power, op_if_else,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TARGET_EXPR = "rank(ts_delta(log(close),5))/ts_std(close,20)"

ASSETS = ["AAPL", "MSFT", "GOOG"]
N_ASSETS = len(ASSETS)
N_DAYS   = 60


def _make_dataset(
    n_days: int = N_DAYS,
    n_assets: int = N_ASSETS,
    assets: list | None = None,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng   = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    cols  = assets or [f"A{i}" for i in range(n_assets)]

    base  = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n_assets)), axis=0))
    close = pd.DataFrame(base, index=dates, columns=cols)
    open_ = pd.DataFrame(base * rng.uniform(0.99, 1.01, base.shape), index=dates, columns=cols)
    high  = pd.DataFrame(base * rng.uniform(1.00, 1.02, base.shape), index=dates, columns=cols)
    low   = pd.DataFrame(base * rng.uniform(0.98, 1.00, base.shape), index=dates, columns=cols)
    vol   = pd.DataFrame(rng.lognormal(12, 0.3, (n_days, n_assets)), index=dates, columns=cols)

    return {
        "close": close,
        "open":  open_,
        "high":  high,
        "low":   low,
        "volume": vol,
    }


@pytest.fixture
def dataset():
    return _make_dataset(assets=ASSETS)


@pytest.fixture
def parser():
    return Parser()


@pytest.fixture
def executor():
    return Executor(validate=True)


# ---------------------------------------------------------------------------
# 1. test_parse_complex
# ---------------------------------------------------------------------------

def test_parse_complex(parser):
    """Parse the target expression without error."""
    node = parser.parse(TARGET_EXPR)
    assert isinstance(node, Node)
    # Root should be an ArithmeticNode (division)
    assert isinstance(node, ArithmeticNode)


# ---------------------------------------------------------------------------
# 2. test_repr_roundtrip
# ---------------------------------------------------------------------------

def test_repr_roundtrip(parser):
    """repr(parse(expr)) should reconstruct a parseable, semantically equal tree."""
    node = parser.parse(TARGET_EXPR)
    r    = repr(node)
    # Parse again from repr — must not raise
    node2 = parser.parse(r)
    # Both trees should produce the same repr (idempotent)
    assert repr(node2) == r


# ---------------------------------------------------------------------------
# 3. test_execute_complex
# ---------------------------------------------------------------------------

def test_execute_complex(parser, executor, dataset):
    """
    Execute rank(ts_delta(log(close),5))/ts_std(close,20) on 3-asset, 60-day data.
    Output shape must be (60, 3).
    First 19 rows of ts_std(close,20) result in NaN, so the signal is NaN there.
    """
    node   = parser.parse(TARGET_EXPR)
    signal = executor.run(node, dataset)

    assert isinstance(signal, pd.DataFrame)
    assert signal.shape == (N_DAYS, N_ASSETS)

    # The std window is 20 — first 19 rows must all be NaN
    assert signal.iloc[:19].isna().all().all(), (
        "Expected NaN for the first 19 rows (ts_std window=20 not met)"
    )
    # At least some non-NaN rows should exist
    assert signal.iloc[20:].notna().any().any()


# ---------------------------------------------------------------------------
# 4. test_memoization
# ---------------------------------------------------------------------------

def test_memoization(executor, dataset):
    """
    An expression with repeated ts_mean(close,20) should compute the
    sub-expression only once (cache hit on second encounter).
    """
    # Build AST manually: ts_mean(close,20) + ts_mean(close,20)
    ts_node  = TimeSeriesNode("ts_mean", DataNode("close"), 20)
    ts_node2 = TimeSeriesNode("ts_mean", DataNode("close"), 20)
    expr     = ArithmeticNode("add", [ts_node, ts_node2])

    # Both ts_node and ts_node2 have the same repr → same cache key
    assert repr(ts_node) == repr(ts_node2)

    # Run via get_cache_keys — should see exactly one entry for ts_mean(close,20)
    # (the two identical nodes share the same cache key, so computed once)
    keys = executor.get_cache_keys(expr, dataset)
    ts_key = repr(ts_node)  # e.g. 'ts_mean(close,20)'
    exact_hits = [k for k in keys if k == ts_key]
    assert len(exact_hits) == 1, (
        f"Expected 1 memoized entry for '{ts_key}', got {len(exact_hits)}: {keys}"
    )


# ---------------------------------------------------------------------------
# 5. test_window_validation
# ---------------------------------------------------------------------------

def test_window_validation():
    """ts_mean(close, -1) must raise ValidationError."""
    validator = AlphaValidator()
    node = TimeSeriesNode.__new__(TimeSeriesNode)
    node.op     = "ts_mean"
    node.child  = DataNode("close")
    node.window = -1
    node.extra_params = {}

    with pytest.raises(ValidationError) as exc_info:
        validator.validate(node)
    assert any("-1" in e or "invalid" in e.lower() for e in exc_info.value.errors)


# ---------------------------------------------------------------------------
# 6. test_lookahead_validation
# ---------------------------------------------------------------------------

def test_lookahead_validation():
    """ts_delay(close, 0) must raise ValidationError (look-ahead bias)."""
    validator = AlphaValidator()
    node = TimeSeriesNode.__new__(TimeSeriesNode)
    node.op     = "ts_delay"
    node.child  = DataNode("close")
    node.window = 0
    node.extra_params = {}

    with pytest.raises(ValidationError) as exc_info:
        validator.validate(node)
    assert any("look-ahead" in e.lower() or "future" in e.lower() for e in exc_info.value.errors)


# ---------------------------------------------------------------------------
# 7. test_depth_validation
# ---------------------------------------------------------------------------

def test_depth_validation():
    """An expression nested 11 levels deep must raise ValidationError."""
    # Build 11-level deep: ts_mean(ts_mean(ts_mean(... close ...)))
    node: Node = DataNode("close")
    for _ in range(11):
        node = TimeSeriesNode("ts_mean", node, 5)

    assert node.depth() == 11

    validator = AlphaValidator()
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(node)
    assert any("depth" in e.lower() for e in exc_info.value.errors)


# ---------------------------------------------------------------------------
# 8. test_cs_type_constraint
# ---------------------------------------------------------------------------

def test_cs_type_constraint():
    """rank(rank(close)) must raise TypeError (CS applied to CS)."""
    inner = CrossSectionalNode("rank", DataNode("close"))
    with pytest.raises(TypeError, match="CrossSectionalNode"):
        CrossSectionalNode("rank", inner)


# ---------------------------------------------------------------------------
# 9. test_nan_policy
# ---------------------------------------------------------------------------

def test_nan_policy():
    """
    With 30 rows and window=20, rows 0–18 must be NaN; row 19+ must be non-NaN
    (assuming no NaN in the input).
    """
    rng = np.random.default_rng(0)
    x = rng.normal(100, 1, (30, 4))  # no NaN in input

    result = bn_ts_std(x, 20)

    # First 19 rows strictly NaN (min_count=window enforced)
    assert np.all(np.isnan(result[:19])), "Expected NaN for rows 0-18"
    # Row 19 onward must be non-NaN (window satisfied)
    assert np.all(~np.isnan(result[19:])), "Expected non-NaN for rows 19+"


# ---------------------------------------------------------------------------
# 10. test_if_else
# ---------------------------------------------------------------------------

def test_if_else(parser, executor, dataset):
    """if_else(close>open,close,open) must produce a valid signal with no NaN."""
    expr   = "if_else((close>open),close,open)"
    signal = executor.run_expr(expr, dataset)

    assert signal.shape == (N_DAYS, N_ASSETS)
    # Result must be >= min(close, open) and <= max(close, open)
    close_arr = dataset["close"].values
    open_arr  = dataset["open"].values
    result    = signal.values

    assert np.all(result >= np.minimum(close_arr, open_arr) - 1e-9)
    assert np.all(result <= np.maximum(close_arr, open_arr) + 1e-9)


# ---------------------------------------------------------------------------
# 11. test_ind_neutralize
# ---------------------------------------------------------------------------

def test_ind_neutralize():
    """
    ind_neutralize(x, groups) must produce group sums close to zero per row.
    """
    rng = np.random.default_rng(7)
    T, N = 20, 6
    x      = rng.normal(0, 1, (T, N))
    groups = np.array([0, 0, 0, 1, 1, 1])  # two groups of 3

    result = ind_neutralize(x, groups)

    # Group 0 sum per row
    g0_sum = np.sum(result[:, :3], axis=1)
    g1_sum = np.sum(result[:, 3:], axis=1)

    np.testing.assert_allclose(g0_sum, 0.0, atol=1e-10,
        err_msg="Group-0 row sums must be ~0 after neutralization")
    np.testing.assert_allclose(g1_sum, 0.0, atol=1e-10,
        err_msg="Group-1 row sums must be ~0 after neutralization")


# ---------------------------------------------------------------------------
# 12. test_signed_power
# ---------------------------------------------------------------------------

def test_signed_power():
    """signed_power(x, 2) must preserve the sign of x."""
    rng = np.random.default_rng(99)
    x   = rng.normal(0, 1, (50, 5))

    result = signed_power(x, 2.0)

    # sign(result) == sign(x), except where x == 0
    nonzero = x != 0
    np.testing.assert_array_equal(
        np.sign(result[nonzero]),
        np.sign(x[nonzero]),
        err_msg="signed_power must preserve the sign of its input",
    )
    # Magnitude: |result| == x^2
    np.testing.assert_allclose(
        np.abs(result),
        x ** 2,
        rtol=1e-12,
        err_msg="|signed_power(x,2)| must equal x^2",
    )
