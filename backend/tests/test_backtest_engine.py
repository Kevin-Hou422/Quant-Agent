"""
回测引擎 7 项单元测试

pytest tests/test_backtest_engine.py -v
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine import (
    DecilePortfolio,
    SignalWeightedPortfolio,
    NeutralizationLayer,
    CostParams,
    SlippageModel,
    LiquidityConstraint,
    BacktestEngine,
    BacktestResult,
    PerformanceAnalyzer,
    RiskReport,
    BacktestVisualizer,
)


# ---------------------------------------------------------------------------
# 共享 Fixtures
# ---------------------------------------------------------------------------

N_DAYS   = 60
N_ASSETS = 10
TICKERS  = [f"A{i:02d}" for i in range(N_ASSETS)]

@pytest.fixture
def dates():
    return pd.date_range("2023-01-02", periods=N_DAYS, freq="B")

@pytest.fixture
def rng():
    return np.random.default_rng(42)

@pytest.fixture
def signal_df(dates, rng):
    """随机信号矩阵 (60×10)，含 5% NaN。"""
    sig = rng.normal(0, 1, (N_DAYS, N_ASSETS)).astype(float)
    nan_mask = rng.random(sig.shape) < 0.05
    sig[nan_mask] = np.nan
    return pd.DataFrame(sig, index=dates, columns=TICKERS)

@pytest.fixture
def prices_df(dates, rng):
    """随机价格矩阵（对数随机游走）。"""
    log_ret = rng.normal(0.0002, 0.01, (N_DAYS, N_ASSETS))
    prices  = 100 * np.exp(np.cumsum(log_ret, axis=0))
    return pd.DataFrame(prices, index=dates, columns=TICKERS)

@pytest.fixture
def volume_df(dates, rng):
    """随机成交量矩阵。"""
    vol = rng.integers(1_000_000, 10_000_000, (N_DAYS, N_ASSETS)).astype(float)
    return pd.DataFrame(vol, index=dates, columns=TICKERS)

@pytest.fixture
def default_cost():
    return CostParams(fixed_bps=5.0, spread_bps=2.0, impact_coef=0.1,
                      adv_cap_pct=0.10, adv_window=5)

@pytest.fixture
def backtest_result(signal_df, prices_df, volume_df, default_cost):
    """运行完整回测，返回 BacktestResult（供多个测试复用）。"""
    pc      = DecilePortfolio(top_pct=0.2, bottom_pct=0.2)
    weights = pc.construct(signal_df)
    weights = NeutralizationLayer.market_neutral(weights)

    engine = BacktestEngine(cost_params=default_cost, initial_capital=1_000_000)
    return engine.run(weights=weights, prices=prices_df,
                      volume=volume_df, signal=signal_df)


# ===========================================================================
# Test 1 — DecilePortfolio
# ===========================================================================

def test_decile_portfolio(signal_df):
    """Top/Bottom 分位权重符号正确，权重和 ≈ 0（市场中性前）。"""
    pc      = DecilePortfolio(top_pct=0.10, bottom_pct=0.10)
    weights = pc.construct(signal_df)

    assert weights.shape == signal_df.shape

    for t in range(N_DAYS):
        sig_row = signal_df.iloc[t].dropna()
        w_row   = weights.iloc[t]
        valid_w = w_row.dropna()

        # 权重非负的资产应有最高信号
        long_tks  = valid_w[valid_w > 0].index
        short_tks = valid_w[valid_w < 0].index

        if len(long_tks) > 0 and len(short_tks) > 0:
            # 做多组信号均值 > 做空组信号均值
            assert (
                sig_row[long_tks].mean() > sig_row[short_tks].mean()
            ), f"第 {t} 日: 多头信号均值应 > 空头信号均值"

        # 市场中性（等权做多做空）：所有权重绝对值之和 > 0 且权重和 ≈ 0
        row_sum = float(w_row.fillna(0).sum())
        assert abs(row_sum) < 1e-9 or abs(row_sum) < 0.01, \
            f"第 {t} 日权重和偏差过大: {row_sum:.6f}"


# ===========================================================================
# Test 2 — SignalWeightedPortfolio（市场中性后）
# ===========================================================================

def test_signal_weighted(signal_df):
    """权重与信号正相关，市场中性化后权重和 ≈ 0。"""
    pc      = SignalWeightedPortfolio(clip_z=3.0)
    weights = pc.construct(signal_df)
    weights = NeutralizationLayer.market_neutral(weights)

    assert weights.shape == signal_df.shape

    for t in range(N_DAYS):
        s = signal_df.iloc[t].dropna()
        w = weights.iloc[t].reindex(s.index)

        # 权重和 ≈ 0
        row_sum = float(w.sum())
        assert abs(row_sum) < 1e-8, f"第 {t} 日市场中性化后权重和={row_sum:.2e}"

        # 权重与信号方向相关（Pearson > 0）
        if len(s) >= 4:
            corr = np.corrcoef(s.values, w.values)[0, 1]
            assert corr > 0 or np.isnan(corr), \
                f"第 {t} 日权重与信号应正相关，实际 corr={corr:.4f}"


# ===========================================================================
# Test 3 — 行业中性
# ===========================================================================

def test_industry_neutral(signal_df):
    """每个行业组内权重和 ≈ 0。"""
    # 把 10 只资产分成 2 个行业，各 5 只
    ind_map = {t: ("Tech" if i < 5 else "Finance") for i, t in enumerate(TICKERS)}

    pc      = SignalWeightedPortfolio()
    weights = pc.construct(signal_df)
    weights = NeutralizationLayer.industry_neutral(weights, ind_map)

    tech_cols    = [t for t in TICKERS if ind_map[t] == "Tech"]
    finance_cols = [t for t in TICKERS if ind_map[t] == "Finance"]

    for t in range(N_DAYS):
        w_tech    = weights.iloc[t][tech_cols].sum()
        w_finance = weights.iloc[t][finance_cols].sum()
        assert abs(w_tech)    < 1e-8, f"第 {t} 日 Tech 权重和={w_tech:.2e}"
        assert abs(w_finance) < 1e-8, f"第 {t} 日 Finance 权重和={w_finance:.2e}"


# ===========================================================================
# Test 4 — ADV 流动性约束
# ===========================================================================

def test_adv_cap(dates, rng):
    """超 ADV 10% 的持仓必须被截断。"""
    # 构造极端权重（某资产权重 = 0.8），ADV 很小 → 必然触发截断
    N = 5
    tickers = [f"B{i}" for i in range(N)]
    prices  = pd.DataFrame(
        np.full((20, N), 100.0),
        index=pd.date_range("2023-01-02", periods=20, freq="B"),
        columns=tickers,
    )
    volume = pd.DataFrame(
        np.full((20, N), 1_000.0),   # 极低成交量 → ADV 极小
        index=prices.index, columns=tickers,
    )
    raw_weights = pd.DataFrame(
        np.tile([0.8, 0.05, 0.05, 0.05, 0.05], (20, 1)),
        index=prices.index, columns=tickers,
    )

    params = CostParams(adv_cap_pct=0.10, adv_window=5)
    liq    = LiquidityConstraint(params)
    adv    = liq.compute_adv(volume, prices)    # 20日 ADV（USD）

    adj    = liq.apply(raw_weights, adv, portfolio_value=1_000_000)

    # ---- 验证截断效果 ----
    # 1. B0 原始权重 0.8，截断+归一化后应显著降低
    b0_orig = raw_weights["B0"].iloc[-1]         # 0.8
    b0_adj  = adj["B0"].iloc[-1]                 # 截断后（归一化后应 ≤ 原值）
    assert b0_adj < b0_orig, \
        f"B0 权重应被截断：{b0_orig:.3f} → {b0_adj:.3f}"

    # 2. 截断前 B0 在全部权重中的占比必须下降（集中度降低）
    orig_share = raw_weights.abs().div(raw_weights.abs().sum(axis=1), axis=0)
    adj_share  = adj.abs().div(adj.abs().sum(axis=1), axis=0)
    assert (adj_share["B0"] < orig_share["B0"]).all(), \
        "截断后 B0 的权重占比应低于原始占比"

    # 3. 调整后的权重矩阵与原始权重矩阵 L1 和相同（仍归一化）
    adj_l1 = adj.abs().sum(axis=1)
    assert np.allclose(adj_l1.values, 1.0, atol=1e-8), \
        f"调整后 L1 范数应=1，实际均值={adj_l1.mean():.6f}"


# ===========================================================================
# Test 5 — 滑点模型：高波动 > 低波动
# ===========================================================================

def test_slippage_model():
    """平方根法则：高波动/高换手时滑点应 > 低波动场景。"""
    params   = CostParams(spread_bps=2.0, impact_coef=0.5, slippage_model="sqrt")
    slip_mdl = SlippageModel(params)

    N      = 5
    adv    = np.full(N, 1_000_000.0)
    trade  = np.full(N,   100_000.0)   # 固定交易额

    vol_low  = np.full(N, 0.005)   # 0.5% 日波动
    vol_high = np.full(N, 0.030)   # 3.0% 日波动

    slip_low  = slip_mdl.compute(trade, adv, vol_low)
    slip_high = slip_mdl.compute(trade, adv, vol_high)

    assert np.all(slip_high > slip_low), \
        f"高波动滑点应 > 低波动：{slip_high} vs {slip_low}"


# ===========================================================================
# Test 6 — BacktestEngine 完整运行
# ===========================================================================

def test_backtest_run(backtest_result, dates):
    """equity_curve 长度正确，trade_log 非空，基本数值合理。"""
    res = backtest_result

    assert isinstance(res, BacktestResult)

    # 长度
    assert len(res.equity_curve) == N_DAYS, \
        f"equity_curve 长度应={N_DAYS}，实际={len(res.equity_curve)}"

    # trade_log 非空
    assert not res.trade_log.empty, "trade_log 不应为空"
    assert set(["date", "ticker", "direction", "slippage_bps", "cost_usd"]).issubset(
        res.trade_log.columns
    )

    # 净值应为正数
    assert (res.equity_curve > 0).all(), "净值应始终为正"

    # positions shape
    assert res.positions.shape == (N_DAYS, N_ASSETS)

    # turnover ∈ [0, 1]
    assert ((res.turnover >= 0) & (res.turnover <= 1.0 + 1e-9)).all(), \
        "换手率应在 [0, 1] 范围内"

    print(f"\nEquity: {res.equity_curve.iloc[-1]:.4f}")
    print(f"Trades: {len(res.trade_log)}")
    print(f"Mean turnover: {res.turnover.mean():.4f}")


# ===========================================================================
# Test 7 — 绩效指标数值范围验证
# ===========================================================================

def test_performance_metrics(backtest_result, prices_df):
    """IC、Drawdown、VaR 数值在合理范围内。"""
    report = RiskReport.from_result(backtest_result, prices=prices_df)

    # 打印摘要
    print("\n" + report.summary())

    # ---- 年化收益率：合理范围 ----
    assert -5.0 < report.annualized_return < 5.0, \
        f"年化收益率超出合理范围: {report.annualized_return:.4f}"

    # ---- 最大回撤：负数且 ≥ -1 ----
    assert -1.0 <= report.max_drawdown <= 0.0, \
        f"最大回撤应在 [-1, 0]，实际: {report.max_drawdown:.4f}"

    # ---- VaR > 0 ----
    assert report.var_95 >= 0, f"VaR 应 ≥ 0，实际: {report.var_95}"
    assert report.cvar_95 >= 0, f"CVaR 应 ≥ 0，实际: {report.cvar_95}"

    # ---- IC ∈ [-1, 1] ----
    if not np.isnan(report.mean_ic):
        assert -1.0 <= report.mean_ic <= 1.0, \
            f"Mean IC 应在 [-1,1]，实际: {report.mean_ic:.4f}"

    # ---- 分档分析单调性检查（非严格要求，仅校验形状）----
    if report.decile_returns is not None:
        assert len(report.decile_returns) == 10, \
            f"分档数应=10，实际={len(report.decile_returns)}"

    # ---- Visualizer 返回有效 Figure ----
    try:
        import plotly.graph_objects as go
        viz = BacktestVisualizer()
        fig = viz.plot(report)
        assert isinstance(fig, go.Figure), "plot() 应返回 go.Figure"
        fig2 = viz.plot_decile_bar(report)
        assert isinstance(fig2, go.Figure)
        print("Plotly 图表生成: OK")
    except ImportError:
        pytest.skip("plotly 未安装，跳过可视化测试")
