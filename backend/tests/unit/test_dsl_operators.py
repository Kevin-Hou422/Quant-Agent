"""
test_dsl_operators.py — 全算子族执行覆盖测试

对每个 DSL 算子执行并验证：
  - 输出 shape == 输入 shape
  - NaN 值在预热期后减少（时间序列算子）
  - 值域在合理范围内
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 执行帮助
# ---------------------------------------------------------------------------

def _exec(dsl: str, n_days: int = 60, n_tickers: int = 5, seed: int = 0):
    from app.core.alpha_engine.parser import Parser
    from app.core.alpha_engine.dsl_executor import Executor

    rng = np.random.default_rng(seed)
    dates   = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"T{i}" for i in range(n_tickers)]
    close   = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0),
        index=dates, columns=tickers,
    )
    volume  = pd.DataFrame(
        rng.integers(500_000, 5_000_000, (n_days, n_tickers)).astype(float),
        index=dates, columns=tickers,
    )
    high  = close * (1 + rng.uniform(0, 0.01, (n_days, n_tickers)))
    low   = close * (1 - rng.uniform(0, 0.01, (n_days, n_tickers)))
    open_ = close * (1 + rng.normal(0, 0.005, (n_days, n_tickers)))
    vwap  = (high + low + close) / 3

    data = {
        "close": close, "open": open_, "high": high, "low": low,
        "volume": volume, "vwap": vwap,
        "returns": close.pct_change().fillna(0.0),
    }

    node   = Parser().parse(dsl)
    result = Executor().run(node, data)
    return result, close.shape


# ---------------------------------------------------------------------------
# 时间序列算子
# ---------------------------------------------------------------------------

class TestTimeSeriesOperators:

    def test_ts_mean(self):
        out, shape = _exec("ts_mean(close, 10)")
        assert out.shape == shape
        assert out.iloc[10:].notna().any().any()

    def test_ts_std(self):
        out, shape = _exec("ts_std(close, 10)")
        assert out.shape == shape
        assert (out.iloc[10:].dropna() >= 0).all().all()

    def test_ts_delta(self):
        out, shape = _exec("ts_delta(close, 5)")
        assert out.shape == shape

    def test_ts_momentum_decay(self):
        out, shape = _exec("ts_momentum_decay(close, 10)")
        assert out.shape == shape

    def test_ts_decay_linear(self):
        out, shape = _exec("ts_decay_linear(close, 5)")
        assert out.shape == shape
        assert out.iloc[5:].notna().any().any()

    def test_ts_max(self):
        out, shape = _exec("ts_max(close, 5)")
        assert out.shape == shape
        # ts_max 输出 >= close（窗口期后）
        from app.core.alpha_engine.parser import Parser
        from app.core.alpha_engine.dsl_executor import Executor
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2022-01-03", periods=60)
        tickers = [f"T{i}" for i in range(5)]
        close = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0, 0.01, (60, 5)), axis=0),
            index=dates, columns=tickers,
        )
        data = {"close": close, "open": close, "high": close * 1.01, "low": close * 0.99,
                "volume": pd.DataFrame(1e6 * np.ones((60, 5)), index=dates, columns=tickers),
                "vwap": close, "returns": close.pct_change().fillna(0)}
        node = Parser().parse("ts_max(close, 5)")
        out2 = Executor().run(node, data)
        assert (out2.iloc[5:] >= close.iloc[5:] - 1e-9).all().all()

    def test_ts_min(self):
        out, shape = _exec("ts_min(close, 5)")
        assert out.shape == shape

    def test_ts_delay(self):
        out, shape = _exec("ts_delay(close, 3)")
        assert out.shape == shape
        # 前 3 行应为 NaN
        assert out.iloc[:3].isna().all().all()


# ---------------------------------------------------------------------------
# 截面算子
# ---------------------------------------------------------------------------

class TestCrossSectionalOperators:

    def test_rank(self):
        out, shape = _exec("rank(close)")
        assert out.shape == shape
        # rank 输出 ∈ [0, 1]（或近似）
        valid = out.dropna(how="all").stack().dropna()
        assert valid.between(-0.1, 1.1).all()

    def test_zscore(self):
        out, shape = _exec("zscore(close)")
        assert out.shape == shape
        # 每行均值 ≈ 0
        row_means = out.mean(axis=1).dropna()
        assert (row_means.abs() < 0.5).all()

    def test_scale(self):
        out, shape = _exec("scale(close)")
        assert out.shape == shape
        # 每行绝对值之和 ≈ 1
        row_sums = out.abs().sum(axis=1).dropna()
        assert (row_sums.between(0.9, 1.1)).all()

    def test_group_rank_actually_ranks_within_groups(self):
        """
        group_rank 必须**在组内**排名。原用例只断言 out.shape —— 而旧实现在缺
        'sector' 时凭空捏造 `np.arange(N)%10` 分组，形状当然对，分组全错也照过。
        这里显式提供分组并检查组内秩。
        """
        from app.core.alpha_engine.parser import Parser
        from app.core.alpha_engine.dsl_executor import Executor
        rng = np.random.default_rng(0)
        n_days, tk = 60, [f"T{i}" for i in range(6)]
        idx = pd.bdate_range("2022-01-03", periods=n_days)
        close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, 6)), axis=0),
                             index=idx, columns=tk)
        data = {"close": close, "open": close,
                "high": close * (1 + rng.uniform(0, 0.006, close.shape)),
                "low":  close * (1 - rng.uniform(0, 0.006, close.shape)),
                "volume": close * 1000, "vwap": close,
                "returns": close.pct_change().fillna(0.0),
                "sector": np.array([0., 0., 0., 1., 1., 1.])}
        out = Executor().run(Parser().parse("group_rank(close, 'sector')"), data)
        assert out.shape == close.shape
        # 每组内部的秩取值集合必须一致（各组独立排名，而非全截面排名）
        last = out.iloc[-1]
        g0 = sorted(np.round(last.iloc[:3].values, 6))
        g1 = sorted(np.round(last.iloc[3:].values, 6))
        assert g0 == g1, f"两组的组内秩集合不同 → 未按组独立排名：{g0} vs {g1}"

    def test_group_rank_without_sector_raises(self):
        """缺分组字段必须报错，**不得**凭空编造分组（那会产出看似正常的错数字）。"""
        with pytest.raises(KeyError) as ei:
            _exec("group_rank(close, 'sector')")
        assert "sector" in str(ei.value)


# ---------------------------------------------------------------------------
# 一元数学算子
# ---------------------------------------------------------------------------

class TestUnaryMathOperators:

    def test_log(self):
        out, shape = _exec("log(close)")
        assert out.shape == shape
        assert (out.dropna() > 0).all().all()  # log(价格) > 0 when price > 1

    def test_abs_value(self):
        out, shape = _exec("abs(ts_delta(close, 5))")
        assert out.shape == shape
        assert (out.dropna() >= 0).all().all()

    def test_sign(self):
        out, shape = _exec("sign(ts_delta(close, 1))")
        assert out.shape == shape
        valid = out.dropna(how="all").stack().dropna()
        assert valid.isin([-1.0, 0.0, 1.0]).all()

    def test_signed_power(self):
        out, shape = _exec("signed_power(rank(close), 2)")
        assert out.shape == shape

    def test_sqrt_abs(self):
        out, shape = _exec("sqrt(abs(ts_delta(close, 5)))")
        assert out.shape == shape
        assert (out.dropna() >= 0).all().all()


# ---------------------------------------------------------------------------
# 二元算术算子
# ---------------------------------------------------------------------------

class TestBinaryArithmeticOperators:

    def test_add(self):
        out, shape = _exec("close + open")
        assert out.shape == shape

    def test_subtract(self):
        out, shape = _exec("close - open")
        assert out.shape == shape

    def test_multiply(self):
        out, shape = _exec("close * volume")
        assert out.shape == shape

    def test_divide(self):
        out, shape = _exec("close / open")
        assert out.shape == shape
        # 不应出现 inf（除数非零）
        valid = out.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        assert valid.notna().any().any()

    def test_divide_by_zero_protected(self):
        """当除数可能为零时，结果不应是 inf（被 NaN 替换）。"""
        out, shape = _exec("close / ts_delta(close, 1)")
        assert not np.isinf(out.values).any()


# ---------------------------------------------------------------------------
# 条件算子
# ---------------------------------------------------------------------------

class TestConditionalOperators:

    def test_if_else(self):
        out, shape = _exec("if_else(close > open, close, open)")
        assert out.shape == shape
        # 结果应 ≥ min(close, open) 且 ≤ max(close, open)

    def test_comparison_gt(self):
        out, shape = _exec("close > ts_mean(close, 10)")
        assert out.shape == shape
        valid = out.dropna(how="all").stack().dropna()
        assert valid.isin([0.0, 1.0]).all()

    def test_comparison_lt(self):
        out, shape = _exec("close < ts_mean(close, 10)")
        assert out.shape == shape

    def test_logical_and(self):
        out, shape = _exec("(close > open) && (volume > ts_mean(volume, 10))")
        assert out.shape == shape
