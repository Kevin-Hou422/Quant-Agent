"""
test_golden_backtest.py — 黄金基准测试（Task 6.4，E5/E-N4 核心补齐，2026-07-30）

动机：既有 ~390 项测试以"非 None / 能跑通"为主，缺**对抗性数值断言**——证明的是
"不崩溃"而非"算得对"。本文件对数值关键路径做**手工推导**并断言到高精度：任一
公式发生 1bp 级偏差都会使对应断言失败（见 TestSensitivitySelfCheck 自检）。

覆盖：
  - TransactionCostEngine.compute()：佣金 + √冲击滑点 + 最小票面费（手推逐笔精确值）
  - PerformanceAnalyzer.max_drawdown()：手推峰值/回撤序列
  - project_to_capped_l1()：预算受限时的精确权重与 L1
  - deflated_sharpe_from_returns()：对称零均值 → 恰好 0.5；多重检验单调性 + 回归锚点

所有"手推"值均已用独立算术在注释中给出推导过程。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# 1. TransactionCostEngine —— 手推逐笔成本
# ===========================================================================

class TestGoldenCostEngine:
    """
    输入（CostParams 默认：fixed_bps=5, spread_bps=2, impact_coef=0.1, min_ticket=1）：
      portfolio_val = 1,000,000
      delta_w  = [0.5, -0.3, 0.0]  → trade_usd = [500000, -300000, 0]
      adv      = [5e6, 3e6, 1e6]   → participation = |trade|/adv = [0.1, 0.1, 0]
      daily_vol= [0.02, 0.02, 0.02]

    手推：
      slip_bps = spread/2 + impact·vol·1e4·√participation
               = 1 + 0.1·0.02·1e4·√0.1 = 1 + 20·0.3162277660 = 7.3245553203
      资产0：notional=500000
        fixed = 500000·5e-4 = 250
        slip  = 500000·7.3245553203·1e-4 = 366.2277660
        fee0  = max(250+366.2277660, 1) = 616.2277660
      资产1：notional=300000
        fixed = 150 ; slip = 219.7366596 ; fee1 = 369.7366596
      total_usd = 985.9644256 ; net_cost_w = fee/1e6
    """

    @pytest.fixture
    def result(self):
        from app.core.backtest_engine.transaction_cost import CostParams, TransactionCostEngine
        eng = TransactionCostEngine(CostParams())
        return eng.compute(
            date=None,
            delta_w=np.array([0.5, -0.3, 0.0]),
            prices=np.array([100.0, 50.0, 20.0]),
            adv_usd=np.array([5e6, 3e6, 1e6]),
            daily_vol=np.array([0.02, 0.02, 0.02]),
            portfolio_val=1_000_000.0,
            tickers=["A", "B", "C"],
        )

    def test_fee_asset0_exact(self, result):
        net_cost_w, _, _ = result
        assert net_cost_w[0] * 1e6 == pytest.approx(616.2277660168, abs=1e-8)

    def test_fee_asset1_exact(self, result):
        net_cost_w, _, _ = result
        assert net_cost_w[1] * 1e6 == pytest.approx(369.7366596101, abs=1e-8)

    def test_zero_trade_zero_cost(self, result):
        net_cost_w, _, _ = result
        assert net_cost_w[2] == 0.0

    def test_total_usd_exact(self, result):
        _, total_usd, _ = result
        assert total_usd == pytest.approx(985.9644256269, abs=1e-7)

    def test_slippage_bps_matches_sqrt_law(self):
        from app.core.backtest_engine.transaction_cost import CostParams, SlippageModel
        slip = SlippageModel(CostParams()).compute(
            trade_usd=np.array([500000.0]),
            adv_usd=np.array([5e6]),
            daily_vol=np.array([0.02]),
        )
        expected = 1.0 + 20.0 * np.sqrt(0.1)          # 7.3245553203
        assert slip[0] == pytest.approx(expected, abs=1e-9)

    def test_min_ticket_fee_floor(self):
        """极小交易额 → 成本被最小票面费 $1 托底。"""
        from app.core.backtest_engine.transaction_cost import CostParams, TransactionCostEngine
        eng = TransactionCostEngine(CostParams())
        net_cost_w, total, _ = eng.compute(
            date=None, delta_w=np.array([1e-7]), prices=np.array([100.0]),
            adv_usd=np.array([1e9]), daily_vol=np.array([0.02]),
            portfolio_val=1_000_000.0, tickers=["A"],
        )
        # notional=0.1 → fixed+slip ≈ 5e-5+... ≪ 1 → 托底为 1.0
        assert total == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# 2. PerformanceAnalyzer —— 手推最大回撤
# ===========================================================================

class TestGoldenDrawdown:
    """
    equity = [1.0, 1.2, 0.9, 1.1, 0.99]
    峰值    = [1.0, 1.2, 1.2, 1.2, 1.2]
    回撤    = [0, 0, -0.25, -0.0833.., -0.175]
    → max_drawdown = -0.25（第 3 日，1.2→0.9）
    """

    def test_max_drawdown_exact(self):
        from app.core.backtest_engine.backtest_engine import BacktestResult
        from app.core.backtest_engine.performance_analyzer import PerformanceAnalyzer
        idx = pd.bdate_range("2023-01-02", periods=5)
        eq  = pd.Series([1.0, 1.2, 0.9, 1.1, 0.99], index=idx)
        net = eq.pct_change().fillna(0.0)
        res = BacktestResult(eq, net, net, pd.DataFrame(index=idx), pd.DataFrame(),
                             pd.Series(0.1, index=idx), pd.DataFrame(index=idx),
                             pd.Series(0.0, index=idx))
        dd, *_ = PerformanceAnalyzer(res, rf=0.0).max_drawdown()
        assert dd == pytest.approx(-0.25, abs=1e-12)


# ===========================================================================
# 3. project_to_capped_l1 —— 预算受限精确解
# ===========================================================================

class TestGoldenProjection:
    """
    w=[0.6,-0.3,0.1], cap=0.15 各名 → 预算 Σcap=0.45 < target 1
    正确解：全部触顶 |w_i|=0.15（符号保留），L1=0.45（不虚假放大到 1）
    """

    def test_budget_limited_exact(self):
        from app.core.backtest_engine.transaction_cost import project_to_capped_l1
        w   = np.array([[0.6, -0.3, 0.1]])
        cap = np.array([[0.15, 0.15, 0.15]])
        p   = project_to_capped_l1(w, cap, target=1.0)
        assert np.allclose(np.abs(p), 0.15, atol=1e-12)
        assert np.sign(p).tolist() == [[1.0, -1.0, 1.0]]
        assert np.abs(p).sum() == pytest.approx(0.45, abs=1e-12)

    def test_feasible_reaches_target(self):
        """预算充足 → L1 恰好 = target。"""
        from app.core.backtest_engine.transaction_cost import project_to_capped_l1
        w   = np.array([[0.6, -0.3, 0.1]])
        cap = np.array([[0.9, 0.9, 0.9]])
        p   = project_to_capped_l1(w, cap, target=1.0)
        assert np.abs(p).sum() == pytest.approx(1.0, abs=1e-12)


# ===========================================================================
# 4. Deflated Sharpe —— 解析性质 + 回归锚点
# ===========================================================================

class TestGoldenDeflatedSharpe:

    def test_symmetric_zero_mean_is_exactly_half(self):
        """对称零均值收益（mean 恰为 0）→ n_trials=1 的 PSR = Φ(0) = 0.5。"""
        from app.core.backtest_engine.performance_analyzer import deflated_sharpe_from_returns
        half = np.array([0.01, -0.02, 0.015, -0.005, 0.008,
                         0.011, -0.014, 0.006, -0.009, 0.013])
        sym  = np.concatenate([half, -half])          # mean == 0 精确
        r    = pd.Series(sym, index=pd.bdate_range("2023-01-02", periods=20))
        assert deflated_sharpe_from_returns(r, n_trials=1) == pytest.approx(0.5, abs=1e-12)

    def test_regression_anchors_and_monotonicity(self):
        """固定序列的 DSR 回归锚点 + 多重检验单调不增。"""
        from app.core.backtest_engine.performance_analyzer import deflated_sharpe_from_returns
        r = pd.Series([0.01] * 30 + [-0.005] * 10,
                      index=pd.bdate_range("2023-01-02", periods=40))
        dsr_1   = deflated_sharpe_from_returns(r, n_trials=1)
        dsr_100 = deflated_sharpe_from_returns(r, n_trials=100)
        # 回归锚点（首次计算并锁定；任何公式漂移都会触发）
        assert dsr_1   == pytest.approx(0.9999363723, abs=1e-9)
        assert dsr_100 == pytest.approx(0.9033891882, abs=1e-9)
        # 多重检验校正必然降低显著性
        assert dsr_100 < dsr_1


# ===========================================================================
# 5. 灵敏度自检 —— 证明黄金断言"真的会因偏差而失败"
# ===========================================================================

class TestSensitivitySelfCheck:
    """故意注入偏差，断言黄金值**不再**成立——证明上面的精确断言有鉴别力，
    而非因容差过宽而永远通过。"""

    def test_cost_golden_is_sensitive_to_1bp(self):
        from app.core.backtest_engine.transaction_cost import CostParams, TransactionCostEngine
        # 把 fixed_bps 从 5 改成 5.0001（+0.0001bp）
        eng = TransactionCostEngine(CostParams(fixed_bps=5.0001))
        net_cost_w, _, _ = eng.compute(
            date=None, delta_w=np.array([0.5]), prices=np.array([100.0]),
            adv_usd=np.array([5e6]), daily_vol=np.array([0.02]),
            portfolio_val=1_000_000.0, tickers=["A"],
        )
        # 与原黄金 fee0 不再相等（灵敏）
        assert net_cost_w[0] * 1e6 != pytest.approx(616.2277660168, abs=1e-8)

    def test_projection_golden_is_sensitive(self):
        from app.core.backtest_engine.transaction_cost import project_to_capped_l1
        # cap 抬到 0.20 → 预算 0.60，权重不再是 0.15
        p = project_to_capped_l1(np.array([[0.6, -0.3, 0.1]]),
                                 np.array([[0.20, 0.20, 0.20]]), target=1.0)
        assert not np.allclose(np.abs(p), 0.15, atol=1e-6)
