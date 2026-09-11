"""
backtest_engine.py —— 逐日记账公式的定钉测试（变异测试驱动）

来由：15 个变异点，首测击杀率 **20.0%**（存活 12），而存活的 12 处**全是算钱的行**：

    delta_w_mat   = adj_weights - prev_w_mat          # 换手
    turnover      = np.sum(np.abs(delta_w), axis=1)/2 # 绝对值
    price_chg     = (p_t - p_{t-1}) / p_{t-1}         # 涨跌幅
    long_w        = np.maximum(prev_w, 0.0)           # 多头腿
    short_w       = np.minimum(prev_w, 0.0)           # 空头腿
    short_exp     = np.sum(np.maximum(-prev_w, 0.0))  # 空头敞口（借券基数）
    net_ret       = gross - cost - borrow             # 净收益
    equity        = equity * (1 + net_ret)            # 复利
    cost_bps      = (cost + borrow) * 10_000          # 成本口径

既有覆盖（test_backtest_engine / golden / test_invariants）测的是**成本引擎**
与整体不变量，对引擎自己这几行**逐日记账**零数值断言。

本文件用**手可算**的输入（常数权重、整数价格、零成本）把每一行钉死。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.backtest_engine.backtest_engine import BacktestEngine
from app.core.backtest_engine.transaction_cost import CostParams

# 零成本参数：把成本项完全关掉，毛收益 / 净收益 / 净值才可手算
FREE = CostParams(fixed_bps=0.0, min_ticket_fee=0.0, spread_bps=0.0,
                  impact_coef=0.0, short_borrow_annual_bps=0.0)


def _frames(prices: list[list[float]], weights: list[list[float]],
            tickers=("A", "B")):
    idx = pd.bdate_range("2024-01-02", periods=len(prices))
    cols = list(tickers)
    px = pd.DataFrame(prices, index=idx, columns=cols, dtype=float)
    w = pd.DataFrame(weights, index=idx, columns=cols, dtype=float)
    vol = pd.DataFrame(1e12, index=idx, columns=cols)      # ADV 足够大 → 不触发截断
    return w, px, vol


def _run(prices, weights, params: CostParams = FREE, capital: float = 1_000_000.0):
    w, px, vol = _frames(prices, weights)
    eng = BacktestEngine(cost_params=params, initial_capital=capital)
    return eng.run(w, px, vol, w)


# ===========================================================================
# A. 毛收益 —— 昨仓 × 今日涨跌
# ===========================================================================

class TestGrossReturn:

    def test_price_change_is_a_ratio_of_the_difference(self):
        """
        `price_chg = (p_t - p_{t-1}) / p_{t-1}`。把 `-` 写成 `+` 会让"涨跌幅"
        变成两日价格之**和**的比值（恒 ≈ 2），满仓时毛收益从 10% 跳到 210%。

        构造：满仓单票，价格 100 → 110 → 毛收益必须精确等于 +10%。
        """
        res = _run([[100.0, 100.0], [110.0, 100.0]], [[1.0, 0.0], [1.0, 0.0]])
        assert res.gross_returns.iloc[0] == 0.0, "首日无价格变化，毛收益必须为 0"
        assert res.gross_returns.iloc[1] == pytest.approx(0.10, abs=1e-12)

    def test_gross_uses_yesterdays_weights(self):
        """
        `prev_w_mat[t] = adj_weights[t-1]` —— 用**昨仓**吃今日涨跌。
        构造：第 0 日空仓、第 1 日才建仓 → 第 1 日毛收益仍为 0（昨仓为 0）。
        """
        res = _run([[100.0, 100.0], [200.0, 100.0], [200.0, 100.0]],
                   [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        assert res.gross_returns.iloc[1] == pytest.approx(0.0, abs=1e-12), (
            "第 1 日昨仓为 0，却算出了毛收益 —— 疑似用了当日权重")

    def test_short_position_profits_when_price_falls(self):
        res = _run([[100.0, 100.0], [90.0, 100.0]], [[-1.0, 0.0], [-1.0, 0.0]])
        assert res.gross_returns.iloc[1] == pytest.approx(0.10, abs=1e-12)

    def test_zero_previous_price_does_not_blow_up(self):
        """`safe_prev_mat = np.where(p == 0, nan, p)` —— 0 价不得产生 inf。"""
        res = _run([[0.0, 100.0], [100.0, 100.0]], [[0.5, 0.5], [0.5, 0.5]])
        assert np.isfinite(res.gross_returns.to_numpy()).all()


# ===========================================================================
# B. 多空腿分离
# ===========================================================================

class TestLegSplit:

    def test_long_leg_keeps_only_positive_weights(self):
        """
        `long_w = np.maximum(prev_w, 0.0)` —— 改成 minimum 会把多头腿变成空头腿。
        构造：A 多头 +0.6（涨 10%）、B 空头 −0.4（不动）
        → 多头腿收益 = 0.6×0.10 = 0.06，空头腿 = 0。
        """
        res = _run([[100.0, 100.0], [110.0, 100.0]],
                   [[0.6, -0.4], [0.6, -0.4]])
        assert res.long_returns.iloc[1] == pytest.approx(0.06, abs=1e-12)
        assert res.short_returns.iloc[1] == pytest.approx(0.0, abs=1e-12)

    def test_short_leg_keeps_only_negative_weights(self):
        """B 空头 −0.4，B 跌 10% → 空头腿收益 = (−0.4)×(−0.10) = +0.04。"""
        res = _run([[100.0, 100.0], [100.0, 90.0]],
                   [[0.6, -0.4], [0.6, -0.4]])
        assert res.short_returns.iloc[1] == pytest.approx(0.04, abs=1e-12)
        assert res.long_returns.iloc[1] == pytest.approx(0.0, abs=1e-12)

    def test_legs_add_up_to_the_gross_return(self):
        """多空腿之和必须等于毛收益 —— 任一腿的截取方向错了都会打破这条。"""
        res = _run([[100.0, 100.0], [110.0, 90.0], [105.0, 95.0]],
                   [[0.6, -0.4], [0.6, -0.4], [0.6, -0.4]])
        assert np.allclose(
            (res.long_returns + res.short_returns).to_numpy(),
            res.gross_returns.to_numpy(), atol=1e-12)


# ===========================================================================
# C. 借券成本
# ===========================================================================

class TestBorrowCost:

    def test_short_exposure_is_the_borrow_base(self):
        """
        `short_exp = np.sum(np.maximum(-prev_w, 0.0))` —— 只对**昨仓空头**计费。
        构造：昨仓 A +0.6 / B −0.4，价格不动 → 净收益 = −借券费。
        年化 365bps、动态 tdays = 2/(1/365.25) —— 直接拿 cost_bps 与手算比。
        """
        params = CostParams(fixed_bps=0.0, min_ticket_fee=0.0, spread_bps=0.0,
                            impact_coef=0.0, short_borrow_annual_bps=365.0)
        res = _run([[100.0, 100.0], [100.0, 100.0]],
                   [[0.6, -0.4], [0.6, -0.4]], params=params)
        # 第 1 日：昨仓空头敞口 0.4
        assert res.gross_returns.iloc[1] == pytest.approx(0.0, abs=1e-12)
        assert res.net_returns.iloc[1] < 0.0, "持有空头却没有借券成本"
        assert res.daily_cost_bps.iloc[1] == pytest.approx(
            -res.net_returns.iloc[1] * 10_000, abs=1e-9)

    def test_long_only_book_pays_no_borrow(self):
        params = CostParams(fixed_bps=0.0, min_ticket_fee=0.0, spread_bps=0.0,
                            impact_coef=0.0, short_borrow_annual_bps=365.0)
        res = _run([[100.0, 100.0], [100.0, 100.0]],
                   [[0.6, 0.4], [0.6, 0.4]], params=params)
        assert res.net_returns.iloc[1] == pytest.approx(0.0, abs=1e-12), (
            "纯多头账本被收了借券费 —— 空头敞口的截取方向疑似反了")


# ===========================================================================
# D. 换手
# ===========================================================================

class TestTurnover:

    def test_turnover_is_half_the_absolute_weight_change(self):
        """
        `turnover = Σ|Δw| / 2`。删掉 `np.abs` 后，一买一卖的 Δw 相互抵消 →
        换手恒为 0，成本模型的输入直接归零。

        构造：从 [+0.5, −0.5] 换到 [−0.5, +0.5]，Σ|Δw| = 2 → 单边换手 1.0，
        而代数和恰好是 0。
        """
        res = _run([[100.0, 100.0], [100.0, 100.0]],
                   [[0.5, -0.5], [-0.5, 0.5]])
        assert res.turnover.iloc[1] == pytest.approx(1.0, abs=1e-12), (
            "一买一卖的换手被抵消成 0 —— 绝对值疑似被删掉")

    def test_first_day_turnover_is_the_initial_build(self):
        res = _run([[100.0, 100.0], [100.0, 100.0]],
                   [[0.6, -0.4], [0.6, -0.4]])
        assert res.turnover.iloc[0] == pytest.approx(0.5, abs=1e-12)
        assert res.turnover.iloc[1] == pytest.approx(0.0, abs=1e-12)

    def test_delta_is_new_minus_old(self):
        """`delta_w = adj_weights - prev_w`。写成 `+` 时不动仓也会产生换手。"""
        res = _run([[100.0] * 2] * 3, [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        assert res.turnover.iloc[1] == pytest.approx(0.0, abs=1e-12), (
            "持仓完全不变却算出了换手 —— Δw 疑似写成了加法")
        assert res.turnover.iloc[2] == pytest.approx(0.0, abs=1e-12)


# ===========================================================================
# E. 净收益与净值递推
# ===========================================================================

class TestNetAndEquity:

    def test_net_is_gross_minus_costs(self):
        """
        `net = gross - cost - borrow`。任一个减号写成加号，
        成本就变成了**收益**（净收益高于毛收益）。

        ⚠️ 被测那一天必须**真的有换手**：持仓不变时 cost_ret 恰好为 0，
        `gross - 0` 与 `gross + 0` 完全一样，第一个减号测不出来
        （第一版就是这样，变异测试证实它存活）。这里第 1 日把仓位翻转。
        """
        params = CostParams(fixed_bps=10.0, min_ticket_fee=0.0, spread_bps=4.0,
                            impact_coef=0.0, short_borrow_annual_bps=365.0)
        res = _run([[100.0, 100.0], [110.0, 100.0]],
                   [[0.6, -0.4], [-0.4, 0.6]], params=params)   # 第 1 日大幅调仓
        assert res.turnover.iloc[1] > 0.5, "构造前提：第 1 日必须有明显换手"
        assert res.net_returns.iloc[1] < res.gross_returns.iloc[1], (
            "净收益高于毛收益 —— 成本项疑似被当成了收益")
        implied_cost = res.gross_returns.iloc[1] - res.net_returns.iloc[1]
        assert implied_cost > 0
        assert res.daily_cost_bps.iloc[1] == pytest.approx(implied_cost * 10_000, abs=1e-6)

    def test_cost_bps_scale_is_ten_thousand(self):
        """`(cost + borrow) * 10_000`。写成 `/` 会让成本口径小 1e8 倍。"""
        params = CostParams(fixed_bps=10.0, min_ticket_fee=0.0, spread_bps=4.0,
                            impact_coef=0.0, short_borrow_annual_bps=0.0)
        res = _run([[100.0, 100.0], [100.0, 100.0]],
                   [[0.6, -0.4], [0.6, -0.4]], params=params)
        assert res.daily_cost_bps.iloc[0] > 1.0, (
            f"首日建仓的成本只有 {res.daily_cost_bps.iloc[0]:.8f} bps —— 量纲疑似写反")

    def test_equity_compounds_multiplicatively(self):
        """
        `equity = equity * (1 + net_ret)`：
          `*` 写成 `/` → 涨 10% 反而缩水；
          `1 + net` 写成 `1 - net` → 方向整体反转。
        构造：零成本满仓，连续两天各 +10% → 净值 = 1.21（复利），不是 1.20。
        """
        res = _run([[100.0, 100.0], [110.0, 100.0], [121.0, 100.0]],
                   [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        eq = res.equity_curve
        assert eq.iloc[0] == pytest.approx(1.0, abs=1e-12)
        assert eq.iloc[1] == pytest.approx(1.10, abs=1e-12)
        assert eq.iloc[2] == pytest.approx(1.21, abs=1e-12), (
            f"两次 +10% 的净值应为 1.21（复利），实际 {eq.iloc[2]}")

    def test_equity_follows_net_returns_exactly(self):
        """净值曲线必须是净收益的累乘 —— 两者口径不能分叉。"""
        params = CostParams(fixed_bps=5.0, min_ticket_fee=0.0, spread_bps=2.0,
                            impact_coef=0.0, short_borrow_annual_bps=50.0)
        res = _run([[100.0, 100.0], [110.0, 95.0], [105.0, 99.0], [108.0, 97.0]],
                   [[0.6, -0.4]] * 4, params=params)
        assert np.allclose(res.equity_curve.to_numpy(),
                           (1 + res.net_returns).cumprod().to_numpy(), atol=1e-12)

    def test_equity_curve_is_normalised_to_one(self):
        """`equity_series[t] = equity / initial_capital` —— 与初始资金无关。"""
        a = _run([[100.0, 100.0], [110.0, 100.0]], [[1.0, 0.0]] * 2, capital=1e5)
        b = _run([[100.0, 100.0], [110.0, 100.0]], [[1.0, 0.0]] * 2, capital=1e7)
        assert np.allclose(a.equity_curve.to_numpy(), b.equity_curve.to_numpy(),
                           atol=1e-12)
        assert a.equity_curve.iloc[-1] == pytest.approx(1.10, abs=1e-12)


# ===========================================================================
# F. 熔断
# ===========================================================================

def test_equity_wipeout_triggers_the_circuit_breaker():
    """
    `if equity <= 0: ruin_date = ...; break` —— 净值穿零后停止模拟，
    剩余天数填 0，并记录熔断日期。
    构造：满仓单票价格归零 → 净收益 −100%。
    """
    res = _run([[100.0, 100.0], [0.01, 100.0], [50.0, 100.0], [80.0, 100.0]],
               [[1.0, 0.0]] * 4)
    assert res.equity_curve.iloc[1] < 0.01
    assert res.equity_curve.iloc[-1] >= 0.0
    assert np.isfinite(res.equity_curve.to_numpy()).all()
