"""
test_backtest_edge_cases.py — 回测引擎边界值测试

场景：全 NaN 价格、零成交量、单标的、单日数据、
全零信号、全同信号、极端信号、价格骤降、中性化边界。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 帮助函数
# ---------------------------------------------------------------------------

def _make_data(n_days=60, n_tickers=10, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_tickers)), axis=0),
        index=dates, columns=tickers,
    )
    volume = pd.DataFrame(
        rng.integers(500_000, 5_000_000, (n_days, n_tickers)).astype(float),
        index=dates, columns=tickers,
    )
    return close, volume, dates, tickers


def _run_backtest(signal, close, volume):
    """执行完整回测流程，返回 BacktestResult 或 None。"""
    from app.core.backtest_engine.backtest_engine import BacktestEngine
    from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio

    constructor = SignalWeightedPortfolio()
    weights = constructor.construct(signal)
    if weights is None or weights.empty:
        return None
    engine = BacktestEngine()
    # 注意：run 的签名是 (weights, prices, volume, signal)。原实现写成
    # run(signal, close, volume) —— 少一个必填参数，**每次都抛 TypeError**，
    # 而所有调用方都用 `except Exception: pass` 吞掉了，导致本文件长期在空跑。
    return engine.run(weights, close, volume, signal)


# ---------------------------------------------------------------------------
# 边界值测试
# ---------------------------------------------------------------------------

class TestAllNaNPrices:

    def test_all_nan_prices_does_not_crash(self):
        """全 NaN 价格不应引发未处理异常。"""
        close, volume, dates, tickers = _make_data()
        nan_close = close.copy()
        nan_close.iloc[:] = np.nan
        signal = close.rank(axis=1)

        # 契约：要么明确拒绝（错误信息须点名 NaN/价格），要么返回**不含 inf/NaN 的**净值。
        # 原用例写的是 `assert ... or True` —— 恒真，等于没有断言。
        try:
            result = _run_backtest(signal, nan_close, volume)
        except Exception as e:
            msg = str(e).lower()
            assert any(k in msg for k in ("nan", "price", "价格", "empty", "空")), (
                f"全 NaN 价格抛出的异常没有说明原因，用户无法定位：{e!r}"
            )
            return
        assert result is not None, "全 NaN 价格返回 None —— 静默失败，调用方无从判断"
        eq = np.asarray(result.equity_curve, dtype=float)
        assert not np.isinf(eq).any(), "全 NaN 价格产出 inf 净值"
        assert not np.isnan(eq).any(), "全 NaN 价格产出 NaN 净值却不报错（会被当成真实结果展示）"


class TestZeroVolume:

    def test_zero_volume_does_not_crash(self):
        """全零成交量不应导致 ADV cap 代码崩溃。"""
        close, _, dates, tickers = _make_data()
        zero_vol = close.copy() * 0.0
        signal = close.rank(axis=1)

        # 零成交量 → ADV=0 → 容量上限为 0 → **必须无持仓**。
        # 若仍产生非零权重，说明流动性约束在 ADV=0 时失效（会在完全不可成交的
        # 标的上下单）。原用例 `except: pass` 让这两种情况都算通过。
        result = _run_backtest(signal, close, zero_vol)
        assert result is not None, "零成交量导致回测返回 None（静默失败）"
        w = np.abs(np.asarray(result.positions, dtype=float))
        assert not np.isnan(w).any(), "零成交量产生 NaN 权重"
        assert w.sum() < 1e-6, (
            f"ADV=0（完全无法成交）却仍持仓 gross={w.sum():.6f} —— 流动性上限在零成交量下失效"
        )


class TestSingleAsset:

    def test_single_asset_shape_preserved(self):
        """单标的组合输出形状应正确（不崩溃）。"""
        close, volume, dates, _ = _make_data(n_tickers=1)
        signal = close.copy()
        signal.iloc[:] = 1.0

        result = _run_backtest(signal, close, volume)
        assert result is not None, "单标的回测返回 None"
        assert np.asarray(result.positions).shape[1] == 1, "单标的输出列数不为 1"
        eq = np.asarray(result.equity_curve, dtype=float)
        assert np.isfinite(eq).all(), "单标的净值含 inf/NaN"


class TestOneDayData:

    def test_one_day_does_not_crash(self):
        """单日数据不应崩溃（某些统计量可能为 NaN 或 0）。"""
        close, volume, dates, tickers = _make_data(n_days=5)
        signal = close.iloc[:1].copy()
        close1 = close.iloc[:1]
        vol1 = volume.iloc[:1]

        # 单日数据无法计算任何收益：契约是**明确拒绝**或返回长度 1 的有限净值，
        # 不能返回一个看起来正常的 Sharpe。
        try:
            result = _run_backtest(signal, close1, vol1)
        except Exception as e:
            assert str(e).strip(), "单日数据抛出空异常信息"
            return
        assert result is not None
        eq = np.asarray(result.equity_curve, dtype=float)
        assert np.isfinite(eq).all(), "单日数据产出非有限净值"
        # 契约二选一（不允许第三种）：单日数据要么**没有** Sharpe（None/NaN），
        # 要么必须是 0 —— 绝不能报出一个看起来正常的比率。
        # 原写法用 if 守卫，Sharpe 是 None 时一条都不检查。
        sr = getattr(result, "sharpe_ratio", None)
        if sr is None or not np.isfinite(sr):
            return                      # 没有 Sharpe：符合契约
        assert abs(float(sr)) < 1e-9, (
            f"仅 1 天数据却报出 Sharpe={sr} —— 无样本的统计量被当成真实结果"
        )


class TestSignalAllZero:

    def test_signal_all_zero_no_positions(self):
        """全零信号应产生全零权重（无持仓）。"""
        from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio

        close, _, _, _ = _make_data()
        signal = close.copy() * 0.0

        constructor = SignalWeightedPortfolio()
        weights = constructor.construct(signal)
        assert weights is not None, "全零信号构建器返回 None"
        assert (weights.abs().sum(axis=1) < 1e-9).all(), "全零信号仍产生持仓"


class TestSignalAllSame:

    def test_signal_all_same_uniform_allocation(self):
        """全相同信号应产生均等分配（或零，因差异为零）。"""
        from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio

        close, _, dates, tickers = _make_data()
        signal = close.copy()
        signal.iloc[:] = 1.0

        constructor = SignalWeightedPortfolio()
        weights = constructor.construct(signal)
        assert weights is not None, "全同信号构建器返回 None"
        row_sums = weights.abs().sum(axis=1)
        assert (row_sums <= 1.0 + 1e-6).all(), f"gross 超 1：max={row_sums.max()}"


class TestExtremeSignalTruncation:

    def test_extreme_signal_truncated(self):
        """极端信号（1e10）经截断后权重应在合理范围内。"""
        from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio
        from app.core.alpha_engine.signal_processor import SignalProcessor, SimulationConfig

        # 注意：分位截断是**逐行截面**操作。若极端值占该行样本的一半，
        # q=0.99 分位数本身就等于极端值，截断在数学上无法生效——所以必须用
        # 足够多标的 + 少量极端值来测，否则测的是一个不成立的期望。
        close, _, _, _ = _make_data(n_tickers=100)
        signal = close.copy()
        signal.iloc[30, 0] = 1e10  # 100 个标的里注入 1 个极端值

        # 先经过信号处理器截断
        cfg = SimulationConfig(truncation_min_q=0.01, truncation_max_q=0.99)
        processor = SignalProcessor(cfg)
        processed = processor.process(signal)

        assert processed is not None, "SignalProcessor 返回 None"
        assert not np.isinf(processed.values).any(), "截断后仍存在 inf"
        # 截断必须真的削掉极端值：1e10 不应原样穿过管道
        assert np.nanmax(np.abs(processed.values)) < 1e9, (
            f"极端值未被截断，max|signal|={np.nanmax(np.abs(processed.values)):.3g}"
        )


class TestPriceDropToZero:

    def test_price_near_zero_no_division_by_zero(self):
        """价格骤降至 0.001 时不应出现除零错误。"""
        from app.core.backtest_engine.backtest_engine import BacktestEngine

        close, volume, dates, tickers = _make_data()
        crash_close = close.copy()
        crash_close.iloc[30:, :3] = 0.001  # 模拟骤跌

        signal = crash_close.rank(axis=1)
        engine = BacktestEngine()

        from app.core.backtest_engine.portfolio_constructor import SignalWeightedPortfolio
        weights = SignalWeightedPortfolio().construct(signal)
        result = engine.run(weights, crash_close, volume, signal)
        assert result is not None, "价格骤降导致回测返回 None"
        eq = np.asarray(result.equity_curve, dtype=float)
        assert np.isfinite(eq).all(), "价格骤降产出 inf/NaN 净值"


class TestSectorNeutralization:

    def test_sector_neutralization_group_sums_zero(self):
        """通过 ind_neutralize DSL 验证行业中性化：截面均值接近 0。"""
        from app.core.alpha_engine.parser import Parser
        from app.core.alpha_engine.dsl_executor import Executor
        import numpy as np, pandas as pd

        rng = np.random.default_rng(0)
        n_days, n_tickers = 60, 10
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
        # sector 走 _AUX_PASSTHROUGH：必须是 (N,) 数值数组（GICS 码），
        # 传字符串 Series 会被 pandas 当日期解析而炸 —— 这是数据契约，不是可选。
        sector = np.array([0.0] * 5 + [1.0] * 5)
        data = {
            "close": close, "open": close, "high": close * 1.01,
            "low": close * 0.99, "volume": volume, "vwap": close,
            "returns": close.pct_change().fillna(0.0),
            "sector": sector,
        }
        # ind_neutralize: removes cross-sectional mean (row mean → 0)
        node = Parser().parse("ind_neutralize(close, 'sector')")
        out = Executor().run(node, data)
        # 真正的行业中性：**每个行业组内**均值为 0（原用例只看全截面均值 <0.5，
        # 那是个极松的上界，行业中性完全失效也能通过）。
        # 真正的行业中性：**每个行业组内**均值为 0。
        # （原用例只断言全截面均值 <0.5 —— cs_zscore 的全截面均值恒为 0，
        #   所以哪怕行业中性完全没生效也能通过，等于没测。）
        for grp in (0.0, 1.0):
            cols = [t for t, sec in zip(tickers, sector) if sec == grp]
            gm = out[cols].mean(axis=1).dropna()
            assert (gm.abs() < 1e-8).all(), (
                f"行业 {grp} 组内均值未归零，max|mean|={gm.abs().max():.3g}。"
                f"根因：parser 把分组参数存为 params['groups_node']，而 "
                f"CrossSectionalNode._compute 读的是 params['groups'] → 恒为 None → "
                f"fast_ops.ind_neutralize 退化为 cs_zscore。行业中性从未生效。"
            )

    def test_ind_neutralize_differs_from_no_sector(self, ):
        """
        判别性探针：给 sector 与不给 sector，ind_neutralize 输出**必须不同**。
        若逐位相同，证明 sector 参数根本没被消费。
        """
        from app.core.alpha_engine.parser import Parser
        from app.core.alpha_engine.dsl_executor import Executor
        rng = np.random.default_rng(0)
        tk = [f"T{i:02d}" for i in range(10)]
        d = pd.bdate_range("2022-01-03", periods=40)
        close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (40, 10)), axis=0),
                             index=d, columns=tk)
        base = {"close": close, "open": close, "high": close * 1.01, "low": close * 0.99,
                "volume": close * 1000, "vwap": close,
                "returns": close.pct_change().fillna(0.0)}
        node = Parser().parse("ind_neutralize(close, 'sector')")
        with_sec = Executor().run(node, {**base, "sector": np.array([0.] * 5 + [1.] * 5)})
        without  = Executor().run(node, base)
        assert not np.allclose(with_sec.values, without.values, equal_nan=True), (
            "给 sector 与不给 sector 的 ind_neutralize 输出逐位相同 —— "
            "分组参数完全未被消费，该算子等价于 cs_zscore。"
        )


class TestSectorFallbackIsNotSilent:
    """
    真实问题探针：dsl_executor 在缺 sector 字段时让 ind_neutralize / group_rank
    **静默退化为全截面版本**（等价于 cs_zscore / cs_rank）。使用者会以为自己做了
    行业中性，实际没有 —— 这是"降级不告知"，与合成数据冒充真实数据同一性质。
    """

    def _data(self, n_tickers=10):
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2022-01-03", periods=40)
        tickers = [f"T{i:02d}" for i in range(n_tickers)]
        close = pd.DataFrame(
            100 * np.cumprod(1 + rng.normal(0, 0.01, (40, n_tickers)), axis=0),
            index=dates, columns=tickers,
        )
        return {"close": close, "open": close, "high": close * 1.01, "low": close * 0.99,
                "volume": close * 1000, "vwap": close,
                "returns": close.pct_change().fillna(0.0)}, tickers

    def test_ind_neutralize_without_sector_must_warn_or_fail(self, caplog):
        from app.core.alpha_engine.parser import Parser
        from app.core.alpha_engine.dsl_executor import Executor
        import logging

        data, _ = self._data()          # 故意不提供 sector
        caplog.set_level(logging.WARNING)
        node = Parser().parse("ind_neutralize(close, 'sector')")
        out = Executor().run(node, data)

        assert out is not None
        warned = any(
            any(k in r.message.lower() for k in ("sector", "group", "行业", "neutral"))
            for r in caplog.records
        )
        assert warned, (
            "缺 sector 字段时 ind_neutralize 静默退化为全截面中性化，"
            "既不报错也不告警 —— 使用者会误以为已做行业中性。"
        )
